"""
    Attributes:
        DATA_FOLDER
        res_Accepts
        res_Cuisine
        res_Hours
        res_Parking
        res_Profile
        user_Cuisine
        user_Payment
        user_Profile
        default_res_Profile
        default_user_Profile
        Ratings
        simple_Ratings
        user_Profile_clean  : all user-profiles minus '?' values
        ratings_close_by    : all simple_Ratings made within 100km of the restaurant
        ratings_minimal     : all simple_Ratings made within 100km from users in user-profile-clean
        ratings_clean       : all simple_Ratings from users in user-profile-clean
        distance_matrix
"""

from typing import List

from IPython.core.display import display
from geopy.distance import distance as geo_distance
import pandas as pd
import numpy as np

# FILE PATHS
DATA_FOLDER = '../kaggle_data'

# Restaurants
RES_ACCEPTS = DATA_FOLDER + '/chefmozaccepts.csv'
RES_CUISINE = DATA_FOLDER + '/chefmozcuisine.csv'
RES_HOURS = DATA_FOLDER + '/chefmozhours4.csv'
RES_PARKING = DATA_FOLDER + '/chefmozparking.csv'
RES_PROFILE = DATA_FOLDER + '/geoplaces2.csv'

# Consumers
USER_CUISINE = DATA_FOLDER + '/usercuisine.csv'
USER_PAYMENT = DATA_FOLDER + '/userpayment.csv'
USER_PROFILE = DATA_FOLDER + '/userprofile.csv'

# User-Item-Rating
RATINGS = DATA_FOLDER + '/rating_final.csv'

# %% loading the data to dataframes

# IMPORTING DATASET

# Restaurants
res_Accepts = pd.read_csv(RES_ACCEPTS)
res_Cuisine = pd.read_csv(RES_CUISINE)
res_Hours = pd.read_csv(RES_HOURS)
res_Parking = pd.read_csv(RES_PARKING)
res_Profile = pd.read_csv(RES_PROFILE)

# Consumers
user_Cuisine = pd.read_csv(USER_CUISINE)
user_Payment = pd.read_csv(USER_PAYMENT)
user_Profile = pd.read_csv(USER_PROFILE)

# User-Item-Rating
Ratings = pd.read_csv(RATINGS)


# %% methods for loading selections of data

def select_data(dataframe: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    if columns is None:
        columns = [x for x in dataframe.columns]
    sub_df = dataframe[[x for x in columns]].copy()
    sub_df = sub_df.replace("?", np.NaN).dropna()
    sub_df = sub_df.reset_index()
    return sub_df


def select_user_data(profile_columns: List[str] = None) -> pd.DataFrame:
    return select_data(user_Profile, profile_columns)


def select_res_data(profile_columns: List[str] = None) -> pd.DataFrame:
    return select_data(res_Profile, profile_columns)


# %% Some default values

DEFAULT_USER_COLUMNS = ['userID', 'latitude', 'longitude', 'drink_level', 'smoker',
                        'dress_preference', 'budget', 'ambience']
DEFAULT_RES_COLUMNS = ['placeID', 'latitude', 'longitude',
                       'name', 'alcohol', 'smoking_area', 'dress_code', 'accessibility', 'price', 'Rambience']

default_user_Profile = select_user_data(DEFAULT_USER_COLUMNS)
default_res_Profile = select_res_data(DEFAULT_RES_COLUMNS)

# simple_ratings is a dataframe that can be directly fed into lenskit for training
# [user; item; rating] columns with a 0-6 rating (sum aggregate)
simple_Ratings = Ratings.copy()
simple_Ratings['sum_column'] = Ratings['rating'] + \
                               Ratings['food_rating'] + Ratings['service_rating']
simple_Ratings = simple_Ratings.rename(
    columns={'userID': 'user', 'placeID': 'item', 'rating': 'general_rating', 'sum_column': 'rating'})
simple_Ratings = simple_Ratings.drop(columns=["general_rating", "food_rating", "service_rating"])


# %% making matrix

def build_matrix():
    res_coords = res_Profile.copy()
    res_coords.index = res_coords.placeID
    res_coords = res_coords.drop(columns=[x for x in res_Profile.columns if x not in ['longitude', 'latitude']])
    user_coords = user_Profile.copy()
    user_coords.index = user_coords.userID
    user_coords = user_coords.drop(columns=[x for x in user_Profile.columns if x not in ['longitude', 'latitude']])

    res_rename = {'latitude': 'res_lat', 'longitude': 'res_lon'}
    user_rename = {'latitude': 'user_lat', 'longitude': 'user_lon'}

    step1 = Ratings.drop(columns=[x for x in Ratings.columns if "rating" in x])
    step2 = pd.merge(step1, res_coords.rename(columns=res_rename), on='placeID', right_index=True).sort_index()
    step3 = pd.merge(step2, user_coords.rename(columns=user_rename), on='userID', right_index=True).sort_index()
    step3['distance'] = [
        geo_distance((step3.loc[i].res_lat, step3.loc[i].res_lon), (step3.loc[i].user_lat, step3.loc[i].user_lon)).km
        for i in step3.index]
    matrix = step3[['userID', 'placeID', 'distance']]
    return matrix


distance_matrix = build_matrix()

#%% Minimizing dataset further for Vector-Based Methods:

user_Profile_clean = user_Profile.replace('?', np.NAN).dropna()
ratings_close_by = simple_Ratings.loc[distance_matrix[distance_matrix.distance < 100].index].copy()
users_dropped = user_Profile.loc[user_Profile.index.drop(user_Profile_clean.index)].userID
ratings_minimal = ratings_close_by.copy()
ratings_clean = simple_Ratings.copy()
for user in users_dropped:
    ratings_minimal['user'] = ratings_minimal['user'].replace(user, np.NAN)
    ratings_clean['user'] = ratings_clean['user'].replace(user, np.NAN)
ratings_minimal = ratings_minimal.dropna()
ratings_clean = ratings_clean.dropna()

# user_Profile_clean : the user-profiles minus '?' values
# ratings_close_by : all simple_Ratings made within 100km of the restaurant
# ratings_minimal : all simple_Ratings made within 100km from users in user-profile-clean
# ratings_clean : all simple_Ratings from users in user-profile-clean
