import pandas as pd
import numpy as np
import xgboost as xgb
import smogn
import optuna
import joblib

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

def train_xgboost(trial):
    
    cfg = {
        'n_estimaters': trial.suggest_categorical('n_estimaters', [2, 5, 10, 50, 100, 150, 200]),
        'max_depth': trial.suggest_categorical('max_depth', [1, 3, 7, 10, 20, 30]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 0.5]),
        'gamma': trial.suggest_categorical('gamma', [0, 0.5, 2, 10, 20, 30, 40]),
        'subsample': trial.suggest_categorical('subsample', [1.0, 0.9, 0.75, 0.5, 0.3, 0.1]),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [0.01, 0.05, 0.25, 0.75]),
        'colsample_bylevel': trial.suggest_categorical('colsample_bylevel', [0.01, 0.1, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.01, 0.1, 1.0]),
    }

    xgb_model = xgb.XGBRegressor(
        n_estimators=cfg['n_estimaters'],
        max_depth=cfg['max_depth'],
        learning_rate=cfg['learning_rate'],
        gamma=cfg['gamma'],
        subsample=cfg['subsample'],
        min_child_weight=cfg['min_child_weight'],
        colsample_bylevel=cfg['colsample_bylevel'],
        colsample_bytree=cfg['colsample_bytree'])
    
    xgb_model.fit(train_df.iloc[:,:-2], train_df.iloc[:,-1])

    predictions = xgb_model.predict(x_test.iloc[:,:-1])
    #mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return mae

if __name__ == '__main__':

    seed = 42
    np.random.seed(seed)

    df, encoder_dict = prepare_data()
    train_df, x_test, y_test = oversampling_with_noise(df, 0.2)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial : train_xgboost(trial), n_trials=200)
    joblib.dump(study, './optuna_result/xgboost.pkl')