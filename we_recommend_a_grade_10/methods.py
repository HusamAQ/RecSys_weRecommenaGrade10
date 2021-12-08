from lenskit.algorithms.bias import Bias
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.batch import predict
from lenskit.metrics.predict import user_metric, rmse
from we_recommend_a_grade_10.data import *


def get_train_test_split(split_count=6.9, train_size=0.7, seed=None):
    ratings_per_user = simple_Ratings[['user', 'item']].groupby(['user']).count()
    ratings_per_user = ratings_per_user.reset_index()
    users_more_than = ratings_per_user[ratings_per_user['item'] > split_count]

    # select a sample of int(split_count) ratings from each user in users_more_than,
    #  this is our training set
    # what remains of the ratings is the testing set
    train_set = pd.DataFrame(columns=simple_Ratings.columns)
    test_set = pd.DataFrame(columns=simple_Ratings.columns)
    for user in users_more_than.user:
        user_ratings = simple_Ratings[simple_Ratings['user'] == user]
        sample_size = len(user_ratings) * train_size
        sample = user_ratings.sample(int(sample_size), random_state=seed)

        train_set = train_set.append(sample)
        test_set = test_set.append(user_ratings.drop(sample.index))

    train_set['rating'] = train_set['rating'].astype(float)
    test_set['rating'] = test_set['rating'].astype(float)
    return train_set, test_set

def train_and_evaluate(train_set, test_set):
    min_nn = 3
    max_nn = 15
    # Minimum (3) and maximum (15) number of neighbors to consider
    user_user = UserUser(max_nn, min_nbrs=min_nn)
    recsys = Recommender.adapt(user_user)
    algo = recsys
    algo.fit(train_set)
    preds = predict(algo, test_set)
    user_metric(preds, metric=rmse)
