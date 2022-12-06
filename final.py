#%%
import pandas as pd
import numpy as np
import xgboost as xgb
import smogn

from utils import median_absolute_error, mean_absolute_percentage_error, acper
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def preprocess_company(company):
    replace_dict = {'㈜':'', '(주)':'', '주식회사': '', '(유)': ''}
    com_list = company.split(',')
    com_list = [replace_all(com,replace_dict).strip() for com in com_list]
    return com_list[0]

def preprocess_grade(grade):
    grade = grade.strip()
    grade_dict = {'전체관람가':0, 
                  '12세관람가': 1, 
                  '12세이상관람가':1, 
                  '15세관람가':2, 
                  '15세이상관람가':2, 
                  '18세관람가':3, 
                  '청소년관람불가':3, 
                  '제한상영가' : 3}
    
    return grade_dict[grade]

def split_actors(x, idx):
    result = x.split(',')
    if len(result) > idx:
        return result[idx]
    else:
        return None
    
def split_date(x):
    result = x.split('-')
    return int(result[1])


def prepare_data():
    df = pd.read_csv('./temp2.csv', names=['movieName', 'director', 'company', 'distributor', 'openDt',
                                        'type1', 'type2', 'genre', 'grade', 'num_actors', 'num_staffs', 'running_time',
                                        'top3_actors', 'num_screen', 'num_seats', 'seoul_seats', 'num_sales','seoul_sales',
                                        'director_prev_movie_num', 'director_prev_movie_seats_num', 'director_prev_max_seats_num', 'actors_prev_seats_num'])
    df.drop(['num_screen', 'num_sales', 'seoul_sales'], axis=1, inplace=True)
    refined_df = df[df['num_seats'] > 10000]

    director_le = LabelEncoder()
    type1_le = LabelEncoder()
    type2_le = LabelEncoder()
    genre_le = LabelEncoder()
    company_le = LabelEncoder()
    distributor_le = LabelEncoder()
    actor1_le = LabelEncoder()
    actor2_le = LabelEncoder()
    actor3_le = LabelEncoder()

    # to reverse, director_le.inverse_transform()
    encoder_dict = {
        'director' : director_le,
        'type1' : type1_le,
        'type2' : type2_le,
        'genre' : genre_le,
        'company' : company_le,
        'distributor' : distributor_le,
        'actor1' : actor1_le,
        'actor2' : actor2_le,
        'actor3' : actor3_le
    }

    refined_df['actor1']= refined_df['top3_actors'].apply(split_actors, idx=0)
    refined_df['actor2']= refined_df['top3_actors'].apply(split_actors, idx=1)
    refined_df['actor3']= refined_df['top3_actors'].apply(split_actors, idx=2)
    refined_df['company'] = refined_df['company'].apply(lambda x : preprocess_company(x))
    refined_df['distributor'] = refined_df['distributor'].apply(lambda x : preprocess_company(x))
    refined_df['grade'] = refined_df['grade'].apply(lambda x : preprocess_grade(x))
    refined_df['openMonth'] = refined_df['openDt'].apply(split_date)
    refined_df.dropna(axis=0, inplace=True)
    refined_df.drop(['top3_actors'], axis=1, inplace=True)

    refined_df['director'] = director_le.fit_transform(refined_df['director'])
    refined_df['type1'] = type1_le.fit_transform(refined_df['type1'])
    refined_df['type2'] = type2_le.fit_transform(refined_df['type2'])
    refined_df['genre'] = genre_le.fit_transform(refined_df['genre'])
    refined_df['company'] = company_le.fit_transform(refined_df['company'])
    refined_df['distributor'] = distributor_le.fit_transform(refined_df['distributor'])
    refined_df['actor1'] = actor1_le.fit_transform(refined_df['actor1'])
    refined_df['actor2'] = actor2_le.fit_transform(refined_df['actor2'])
    refined_df['actor3'] = actor3_le.fit_transform(refined_df['actor3'])

    return refined_df, encoder_dict

def oversampling_with_noise(df, split_size):

    X = df[['director', 'company', 'distributor', 'type1', 'type2', 'genre', 'grade', 'num_actors', 'num_staffs', 'running_time', 
        'actor1', 'actor2', 'actor3', 'openMonth', 'director_prev_movie_num', 'director_prev_movie_seats_num', 
        'director_prev_max_seats_num', 'actors_prev_seats_num', 'openDt' ]]
    y = df[['num_seats']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_size , random_state= 42)
    
    train_df = x_train.join(y_train)
    oversampled_train_df = smogn.smoter(    
        data = train_df,
        samp_method = 'extreme',
        y = 'num_seats')

    return oversampled_train_df, x_test, y_test

#%%
seed = 42
np.random.seed(seed)

df, encoder_dict = prepare_data()
train_df, x_test, y_test = oversampling_with_noise(df, 0.2)

xgb_model = xgb.XGBRegressor(
    n_estimators=10,
    max_depth=7,
    learning_rate=0.1,
    gamma=20,
    subsample=0.9,
    min_child_weight=0.01,
    colsample_bylevel=1.0,
    colsample_bytree=1.0)

xgb_model.fit(train_df.iloc[:,:-2], train_df.iloc[:,-1])
predictions = xgb_model.predict(x_test.iloc[:,:-1])
mae = mean_absolute_error(y_test, predictions)
mdae = median_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)

print(mae)
print(mdae)
print(mape)

# %%
test_df = x_test.join(y_test)

# %%
for genre_name in np.unique(test_df.genre):

    temp = test_df[test_df.genre == genre_name]
    temp_x = temp.iloc[:,:-2]
    temp_y = temp.iloc[:,-1]

    temp_pred = xgb_model.predict(temp_x.iloc[:, :])

    mae = round(mean_absolute_error(temp_y, temp_pred),0)
    mdae = round(median_absolute_error(temp_y, temp_pred), 0)
    mape = round(mean_absolute_percentage_error(temp_y, temp_pred),2)

    acper_result = list(acper(temp_y, temp_pred, threshold=0.5)) 
    true_value = acper_result.count(True)
    acper_value = round(true_value / len(acper_result) * 100, 2)
    genre_realname = encoder_dict['genre'].inverse_transform([genre_name])[0]

    #print(genre_realname, mae, mdae, mape, acper_value, len(temp), max(temp_y), min(temp_y), np.mean(temp_y))
    print(genre_realname, acper_value)

# %%
 
predictions = xgb_model.predict(x_test.iloc[:,:-1])
mae = mean_absolute_error(y_test, predictions)
mdae = median_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)


# %%
xgb.plot_importance(xgb_model)
# %%
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test.iloc[:,:-1])

#%%
shap.summary_plot(shap_values, x_test.iloc[:,:-1], plot_type="bar")
# %%
feature_names = ['director', 'company', 'distributor', 'type1', 'type2', 'genre', 'grade', 'num_actors', 'num_staffs', 'running_time', 
'actor1', 'actor2', 'actor3', 'openMonth', 'director_prev_movie_num', 'director_prev_movie_seats_num', 
'director_prev_max_seats_num', 'actors_prev_seats_num']

shap.summary_plot(shap_values, x_test.iloc[:,:-1], feature_names=feature_names)
# %%
xgb.plot_importance(xgb_model, importance_type='weight')
# %%
xgb.plot_importance(xgb_model, importance_type='cover')
# %%
xgb.plot_importance(xgb_model, importance_type='gain')