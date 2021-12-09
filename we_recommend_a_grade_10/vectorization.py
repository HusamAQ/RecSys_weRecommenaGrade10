"""
	Attributes:
		vector_cols
		res_quantified
		user_quantified
		res_profile_qnt
		user_profile_qnt
		ratings_vertices
"""
from we_recommend_a_grade_10.data import *

# %%
# setting all vector quantification:
vector_cols = ['alcohol', 'smoking_area', 'dress_code', 'accessibility', 'price', 'Rambience']

res_quantified = {
    'alcohol': {
        'No_Alcohol_Served': 0,
        'Wine-Beer': 1,
        'Full_Bar': 1
    },
    'smoking_area': {
        'not permitted': 0,
        'none': 0.33,
        'section': 0.67,
        'only at bar': 0.67,
        'permitted': 1
    },
    'dress_code': {
        'informal': 0,
        'casual': 0.5,
        'formal': 1
    },
    'accessibility': {
        'no_accessibility': 0,
        'partially': 0.5,
        'completely': 1
    },
    'price': {
        'low': 0,
        'medium': 0.5,
        'high': 1
    },
    'Rambience': {
        'quiet': 0,
        'familiar': 1
    }}

user_quantified = {
    'drink_level': {
        'abstemious': 0,
        'casual drinker': 0.5,
        'social drinker': 1
    },
    'smoker': {
        'false': 0,
        'true': 1
    },
    'dress_preference': {
        'informal': 0,
        'no preference': 0.5,
        'elegant': 0.75,
        'formal': 1
    },
    'budget': {
        'low': 0,
        'medium': 0.5,
        'high': 1
    },
    'ambience': {
        'solitary': 0,
        'friends': 0.75,
        'family': 1
    }}
# %%
res_profile_qnt = default_res_Profile.drop(columns=['longitude', 'latitude'])
user_profile_qnt = default_user_Profile.drop(columns=['longitude', 'latitude'])

for key in res_quantified.keys():
    for var in res_quantified[key].keys():
        res_profile_qnt[key] = res_profile_qnt[key].replace(var, res_quantified[key][var])
    res_profile_qnt[key] = res_profile_qnt[key].astype(float)

for key in user_quantified.keys():
    for var in user_quantified[key].keys():
        user_profile_qnt[key] = user_profile_qnt[key].replace(var, user_quantified[key][var])
    user_profile_qnt[key] = user_profile_qnt[key].astype(float)
user_profile_qnt = user_profile_qnt.rename(columns={
    'drink_level': 'alcohol', 'smoker': 'smoking_area', 'dress_preference': 'dress_code',
    'budget': 'price', 'ambience': 'Rambience'})
# %%
# This code combines the user and restaurant quantified vector values into a single value for
#  each dimension; It keeps the raw restaurant value as '%s-raw'%col as well
ratings_vertices = ratings_clean.copy()

if True:
    # user quantified : indexed by ratingID
    user_qnt = user_profile_qnt.copy()
    user_qnt.index = user_qnt['userID']
    user_qnt = user_qnt.drop(columns=['index', 'userID'])
    rtv_user = ratings_vertices.copy()
    rtv_user['index'] = rtv_user.index
    rtv_user.index = rtv_user['user']
    uuu = user_qnt.loc[ratings_vertices.user].copy()
    uuu.index = rtv_user['index']

    # item quantified : indexed by ratingID
    res_qnt = res_profile_qnt.copy()
    res_qnt.index = res_qnt['placeID']
    res_qnt = res_qnt.drop(columns=['index', 'placeID'])
    rtv_res = ratings_vertices.copy()
    rtv_res['index'] = rtv_res.index
    rtv_res.index = rtv_res['item']
    iii = res_qnt.loc[ratings_vertices.item].copy()
    iii.index = rtv_res['index']

    # create combination of user and restaurant vectors:
    for col in vector_cols:
        ratings_vertices[col + '-raw'] = iii[col]
        if col in uuu.columns:
            ratings_vertices[col] = 1 - (uuu[col] - iii[col]).abs()
        else:
            ratings_vertices[col] = iii[col]