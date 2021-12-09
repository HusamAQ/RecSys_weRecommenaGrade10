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

def train_and_run_CF(train_set, test_set, min_nn = 3, max_nn = 15):
    """
    :param train_set:
    :param test_set:
    :param min_nn:
    :param max_nn:
    :return: preds, recsys
    """
    # Minimum (3) and maximum (15) number of neighbors to consider
    user_user = UserUser(max_nn, min_nbrs=min_nn)
    recsys = Recommender.adapt(user_user)
    recsys.fit(train_set)
    preds = predict(recsys, test_set)
    #metric = user_metric(preds, metric=rmse)
    return preds, recsys

"""
For CF-individual:
 "You were recommended this restaurant because X users that liked similar restaurants also liked this restaurant" 
For VB-individual:
 "You were recommended this restaurant because X restaurants you liked were similar to this restaurant in Y (and Z)"
"""

def train_and_run_VB(train_set, test_set, n=5, metric='euclidean', raw_features=False):
    """
    :param train_set:
    :param test_set:
    :param n:
    :param metric:
    :param raw_features:
    :return: preds, knn
    """
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

    test_set2 = all_vertices[vector_cols].loc[test_set.index.intersection(all_vertices.index)]
    proba = knn.predict_proba(test_set2)
    values = proba[:, 0] * 0
    for i in knn.classes_:
        values += proba[:, i] * i
    pred = pd.Series(values, index=test_set2.index)
    preds = test_set.copy()
    preds['prediction'] = pred
    return preds, knn