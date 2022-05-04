import pandas as pd
import seaborn as sns
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from pandas.io.json import json_normalize



from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score,precision_score, f1_score, confusion_matrix,classification_report, plot_roc_curve,auc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.over_sampling import SMOTE
import requests
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

import requests

results_2019 = pd.read_csv("/Users/jschmidt345/UM_Masters/data_game_res_2019.csv")
betting_2019 = pd.read_csv("/Users/jschmidt345/UM_Masters/data_2019_betting.csv")
advanced1_2019 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2019_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2019 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2019_ng.csv').rename(columns={'gameId' : 'id'})

# game_results_2018 = pd.read_csv('game_res_2018.csv')
# betting_2018 = pd.read_csv('data_betting_2018.csv')
# adv_stat_2018 = pd.read_csv('advanced1_2018_ng.csv').rename(columns={'gameId' : 'id'})
# more_stat_2018 = pd.read_csv('advanced2_2018_ng.csv').rename(columns={'gameId' : 'id'})

results_2021 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2021.csv')
betting_2021 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2021.csv')
advanced1_2021 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2021.csv').rename(columns={'gameId' : 'id'})
advanced2_2021 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2021.csv').rename(columns={'gameId' : 'id'})

results_2018 = pd.read_csv('/Users/jschmidt345/UM_Masters/2018_results.csv')
betting_2018 = pd.read_csv('/Users/jschmidt345/UM_Masters/2018_betting.csv')
advanced1_2018 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2018_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2018 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2018_ng.csv').rename(columns={'gameId' : 'id'})

results_2017 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2017.csv')
betting_2017 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2017.csv')
advanced1_2017 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2017_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2017 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2017_ng.csv').rename(columns={'gameId' : 'id'})

results_2020 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2020.csv')
betting_2020 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2020.csv')
advanced1_2020 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced_2020_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2020 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2020_ng.csv').rename(columns={'gameId' : 'id'})

results_2015 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2015.csv')
betting_2015 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2015.csv')
advanced1_2015 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2015_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2015 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2015_ng.csv').rename(columns={'gameId' : 'id'})

results_2016 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2016.csv')
betting_2016 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2016.csv')
advanced1_2016 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2016_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2016 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2016_ng.csv').rename(columns={'gameId' : 'id'})


results_2014 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2014.csv')
betting_2014 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2014.csv')
advanced1_2014 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2014_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2014 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2014_ng.csv').rename(columns={'gameId' : 'id'})


results_2013 = pd.read_csv('/Users/jschmidt345/UM_Masters/results_2013.csv')
betting_2013 = pd.read_csv('/Users/jschmidt345/UM_Masters/betting_2013.csv')
advanced1_2013 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced1_2013_ng.csv').rename(columns={'gameId' : 'id'})
advanced2_2013 = pd.read_csv('/Users/jschmidt345/UM_Masters/advanced2_2013_ng.csv').rename(columns={'gameId' : 'id'})





def get_player_lvl_data_home(results, category, query_year, homeaway):
    
    plyr_lbl_qb_szn = []
    
    for week in set(results['week']):
        print(week)
        plyr_lvl_qb_wk = []

        for _id in set(results[results['week'] == week]['id']):

            key = "Bearer PNkaVR4Ti/Mr9d+LMxa875mI9Yvl5fw8EvZRuJDWnC6KN+1O1Cl3g5tB/YLq8hix"
            header = {'Authorization':key}
            params = {'year':query_year, 'week':week, 'categories':'defensive', 'gameId' : _id}


            response = requests.get("http://api.collegefootballdata.com/games/players",
                                    headers=header,
                                    params=params)

            response_json = response.json()

            ie = pd.json_normalize(response_json[0]['teams'])
            player_cats = pd.json_normalize(ie[ie['homeAway'] == homeaway]['categories'].item())

            
            if len(player_cats[player_cats['name'] == category]['types']) > 0:
    
                qb_stats = player_cats[player_cats['name'] == category]['types'].tail(1).item()
            else:
                continue
    

            t_ = []
            
            for frame in qb_stats:
                
                dd = pd.json_normalize(frame['athletes'])
                dd['stat_name'] = frame['name']
                t_.append(dd)

            df=pd.concat(t_)

            df = df.reset_index().groupby(['name', 'stat_name'])['stat'].aggregate('first').unstack().reset_index()
            df['gameId'] = _id
            plyr_lvl_qb_wk.append(df)

        week_combined = pd.concat(plyr_lvl_qb_wk)
        week_combined['week'] = week
        plyr_lbl_qb_szn.append(week_combined)
        
    return pd.concat(plyr_lbl_qb_szn)


def smash_df(category, results, query_year, homeaway):
    if category == 'defensive':
        columns = ['PD', 'QB HUR', 'SACKS', 'SOLO', 'TD', 'TFL', 'TOT', 'gameId']
        ext = '_def'
    elif category == 'rushing':
        columns = ['AVG', 'CAR', 'LONG', 'TD', 'YDS', 'gameId']
        ext = '_rush'
    elif category == 'passing':
        columns = ['gameId', 'AVG', 'C/ATT', 'INT', 'QBR', 'TD', 'YDS']
        ext = '_passing'
    else:
        columns = ['AVG', 'LONG', 'REC', 'TD', 'YDS', 'gameId']
        ext = '_rec'

    # ps = get_player_lvl_data_home(results=results, category=category, query_year=query_year, homeaway=homeaway)
    ps_home = get_player_lvl_data_home(results=results, category=category, query_year=query_year, homeaway='home')
    ps_away = get_player_lvl_data_home(results=results, category=category, query_year=query_year, homeaway='away')
    
    ps_home = ps_home[columns]
    ps_away = ps_away[columns]
    
    ps_home['hA'] = 'home'
    ps_away['hA'] = 'away'
    # ps = ps[columns]

    ps = pd.concat((ps_home, ps_away))

    sm = ps.groupby('gameId').apply(lambda x:pd.DataFrame(x.reset_index().unstack()).transpose()).fillna(0)
    sm = sm.drop('gameId', axis=1)
    sm = sm[columns].add_suffix(ext)
    return sm


def make_player_level_final(results, query_year, homeaway):
    from functools import reduce
    data_frames = []
    for category in ['defensive', 'rushing', 'passing', 'receiving']:
        print('doing: ', category)
        df = smash_df(category=category, results=results, query_year=query_year, homeaway=homeaway)
#         df = df.loc[:,~df.columns.str.startswith('gameId')]
        data_frames.append(df)
    df_merged = reduce(lambda  left, right: pd.merge(left, right, right_index=True, left_index=True,
                                            how='outer'), data_frames)
    return df_merged

query_year = 2018
away_17 = make_player_level_final(results=results_2018, query_year=query_year, homeaway='away')
away_17.to_csv(f"player_level_{query_year}.csv", index=False)