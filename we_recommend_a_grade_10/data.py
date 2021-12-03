from typing import List

from IPython.core.display import display
import pandas as pd
import numpy as np

# FILE PATHS
DATA_FOLDER = './kaggle_data'

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

DEFAULT_USER_COLUMNS = ['userID', 'latitude', 'longitude', 'smoker', 'drink_level',
                        'dress_preference', 'marital_status', 'hijos', 'birth_year', 'activity', 'budget']
DEFAULT_RES_COLUMNS = ['placeID', 'latitude', 'longitude',
                       'name', 'alcohol', 'smoking_area', 'dress_code', 'price']

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
