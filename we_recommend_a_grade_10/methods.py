from lenskit.algorithms.bias import Bias
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.batch import predict
from lenskit.metrics.predict import user_metric, rmse
from sklearn.neighbors import KNeighborsClassifier as Knn

# this also import all data:
from we_recommend_a_grade_10.vectorization import *


def get_train_test_split(ratings=simple_Ratings, split_count=6.9, train_size=0.7, seed=None):
    ratings_per_user = ratings[['user', 'item']].groupby(['user']).count()
    ratings_per_user = ratings_per_user.reset_index()
    users_more_than = ratings_per_user[ratings_per_user['item'] > split_count]

    # select a sample of int(split_count) ratings from each user in users_more_than,
    #  this is our training set
    # what remains of the ratings is the testing set
    train_set = pd.DataFrame(columns=ratings.columns)
    test_set = pd.DataFrame(columns=ratings.columns)
    for user in users_more_than.user:
        user_ratings = ratings[ratings['user'] == user]
        sample_size = len(user_ratings) * train_size
        sample = user_ratings.sample(int(sample_size), random_state=seed)

        train_set = train_set.append(sample)
        test_set = test_set.append(user_ratings.drop(sample.index))

    train_set['rating'] = train_set['rating'].astype(float)
    test_set['rating'] = test_set['rating'].astype(float)
    return train_set, test_set


def single_metric(preds):
    return user_metric(preds, metric=rmse)


def train_CF(train_set, min_nn=3, max_nn=15):
    # Minimum (3) and maximum (15) number of neighbors to consider
    user_user = UserUser(max_nn, min_nbrs=min_nn)
    recsys = Recommender.adapt(user_user)
    recsys.fit(train_set)
    return recsys

def run_CF(test_set, recsys):
    preds = predict(recsys, test_set)
    return preds

def train_and_run_CF(train_set, test_set, min_nn=3, max_nn=15):
    recsys = train_CF(train_set, min_nn, max_nn)
    return run_CF(test_set, recsys), recsys


"""
For CF-individual:
 "You were recommended this restaurant because X users that liked similar restaurants also liked this restaurant" 
For VB-individual:
 "You were recommended this restaurant because X restaurants you liked were similar to this restaurant in Y (and Z)"
"""

def train_VB(train_set, n=5, metric='euclidean', raw_features=False):
    all_vertices = ratings_vertices.copy()
    if raw_features:
        all_vertices = all_vertices.drop(columns=[x for x in vector_cols])
        all_vertices = all_vertices.rename(columns={x + "-raw": x for x in vector_cols})
    else:
        all_vertices = all_vertices.drop(columns=[x for x in all_vertices.columns if 'raw' in x])

    knn = Knn(n, metric=metric)
    X = all_vertices.loc[train_set.index.intersection(all_vertices.index)][vector_cols]
    Y = all_vertices.loc[train_set.index.intersection(all_vertices.index)]['rating']
    knn.fit(X, Y)
    return knn, all_vertices

def run_VB(test_set, knn, all_vertices):
    test_set2 = all_vertices[vector_cols].loc[test_set.index.intersection(all_vertices.index)]
    proba = knn.predict_proba(test_set2)
    values = proba[:, 0] * 0
    for i in knn.classes_:
        values += proba[:, i] * i
    pred = pd.Series(values, index=test_set2.index)
    preds = test_set.copy()
    preds['prediction'] = pred
    return preds

def train_and_run_VB(train_set, test_set, n=5, metric='euclidean', raw_features=False):
    """
    :param train_set:
    :param test_set:
    :param n:
    :param metric:
    :param raw_features:
    :return: preds, knn
    """
    knn, all_vertices = train_VB(train_set, n, metric, raw_features)
    return run_VB(test_set, knn, all_vertices), knn, all_vertices


def train_CFVB(train_set, gamma=0.5, min_nn=3, max_nn=15,
                       n=5, metric='euclidean', raw_features=False):
    if gamma < 1:
        alg_CF = train_CF(train_set, min_nn, max_nn)
    else:
        alg_CF = -1
    if gamma > 0:
        alg_VB, all_vs = train_VB(train_set, n, metric, raw_features)
    else:
        alg_VB = -1
        all_vs = -1
    return alg_CF, alg_VB, all_vs, gamma

def run_CFVB(test_set, models):
    alg_CF, alg_VB, all_vs, gamma = models
    if alg_CF != -1:
        prd_CF = run_CF(test_set, alg_CF)
    if alg_VB != -1:
        prd_VB = run_VB(test_set, alg_VB, all_vs)
    # prd_CF * (1 - gamma) + gamma * prd_VB
    prd_CFVB = test_set.copy()
    if alg_CF == -1:
        prd_CF = prd_VB
    if alg_VB == -1:
        prd_VB = prd_CF
    prd_CFVB['prediction'] = prd_CF['prediction'] * (1 - gamma) + gamma * prd_VB['prediction']
    return prd_CFVB

def train_and_run_CFVB(train_set, test_set, gamma=0.5, min_nn=3, max_nn=15,
                       n=5, metric='euclidean', raw_features=False):
    models = train_CFVB(train_set, gamma, min_nn, max_nn, n, metric, raw_features)
    return run_CFVB(test_set, models), models

def explanations_CF(user, recsys):
    X = recsys.get_params()['predictor__min_nbrs']
    recommended = pd.DataFrame(recsys.predict_for_user(user, res_Profile.placeID)).dropna()
    if len(recommended) == 0:
        return None

    item_prof = res_Profile.copy()
    item_prof.index = item_prof['placeID']
    item = item_prof.loc[recommended.sort_values('prediction').iloc[-1].name]
    st = item['name']
    return "You were recommended %s because %d users who liked the same restaurants as you also liked %s"%(st, X, st)

def explanations_VB(user, knn, all_vertices):
    vectors = all_vertices[all_vertices.user == user]
    if len(vectors) <= 0:
        return None
    proba = knn.predict_proba(vectors[vector_cols])
    values = proba[:, 0] * 0
    for i in knn.classes_:
        values += proba[:, i] * i
    vectors['prediction'] = values
    item = vectors.sort_values('prediction').iloc[-1]
    matches = list(item[vector_cols].sort_values()[-2:].index)
    x1, x2 = matches[1], matches[0]
    res_res = res_Profile.copy()
    res_res.index = res_Profile.placeID
    s = res_res.loc[item['item']]['name']
    return "You were recommended %s because this restaurant matches your preferences for %s and %s"%(s, x1, x2)