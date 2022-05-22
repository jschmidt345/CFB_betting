import pandas as pd
import warnings
from functools import reduce

warnings.filterwarnings('ignore')
from pandas.io.json import json_normalize
import requests
# now you can import normally from sklearn.impute

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
            try:
                ie = pd.json_normalize(response_json[0]['teams'])
            except:
                continue
            player_cats = pd.json_normalize(ie[ie['homeAway'] == homeaway]['categories'].item())
            tm = ie[ie['homeAway'] == homeaway]['school']

            
            if len(player_cats[player_cats['name'] == category]['types']) > 0:
    
                qb_stats = player_cats[player_cats['name'] == category]['types'].tail(1).item()
            else:
                continue
    

            t_ = []
            
            for frame in qb_stats:
                dd = pd.json_normalize(frame['athletes'])
                
                # if category == 'defensive':

                #     cls=['PD', 'QB HUR', 'SACKS', 'SOLO', 'TD', 'TFL', 'TOT']
                #     so = [False, False, False, False, False, False, False]
                #     dd = dd.sort_values(by=cls, ascending=so)

                dd['stat_name'] = frame['name']
                t_.append(dd)

            df=pd.concat(t_)

            df = df.reset_index().groupby(['name', 'stat_name'])['stat'].aggregate('first').unstack().reset_index()

            if category == 'defensive':

                cls=['PD', 'QB HUR', 'SACKS', 'SOLO', 'TD', 'TFL', 'TOT']
                so = [False, False, False, False, False, False, False]
                df = df.sort_values(by=cls, ascending=so)
            

            df['gameId'] = _id
            df['team'] = tm
            
            df = df.head(15) 
            if category == 'passing':
                df['COMPLETIONS'] = df['C/ATT'].apply(lambda x: x.split('/')[0]).astype(int)
                df['ATT'] = df['C/ATT'].apply(lambda x: x.split('/')[-1]).astype(int)
                df['COMPPCT'] = df['COMPLETIONS'] / df['ATT']
                df = df.drop("C/ATT", axis=1)

            plyr_lvl_qb_wk.append(df)

        week_combined = pd.concat(plyr_lvl_qb_wk)
        week_combined['week'] = week
        plyr_lbl_qb_szn.append(week_combined)
        
    return pd.concat(plyr_lbl_qb_szn)


def smash_df(category, results, query_year, homeaway):
    if category == 'defensive':
        columns = ['PD', 'QB HUR', 'SACKS', 'SOLO', 'TD', 'TFL', 'TOT', 'gameId', 'team']
        ext = '_def'
    elif category == 'rushing':
        columns = ['AVG', 'CAR', 'LONG', 'TD', 'YDS', 'gameId', 'team']
        ext = '_rush'
    elif category == 'passing':
        columns = ['gameId', 'AVG', 'COMPLETIONS', 'ATT', 'COMPPCT', 'INT', 'QBR', 'TD', 'YDS', 'team']
        ext = '_passing'
    else:
        columns = ['AVG', 'LONG', 'REC', 'TD', 'YDS', 'gameId', 'team']
        ext = '_rec'

    ps_home = get_player_lvl_data_home(results=results, category=category, query_year=query_year, homeaway='home')
    ps_away = get_player_lvl_data_home(results=results, category=category, query_year=query_year, homeaway='away')
    
    ps_home = ps_home[columns]
    ps_away = ps_away[columns]
    
    ps_home['hA'] = 'home'
    ps_away['hA'] = 'away'
    
    ps_away['gameId'] = ps_away['gameId'].astype(str) + '_'

    ps = pd.concat((ps_home, ps_away))
    
    sm = ps.groupby('gameId').apply(lambda x:pd.DataFrame(x.reset_index().unstack()).transpose()).fillna(0)
    sm = sm[columns].add_suffix(ext)
    return sm


# def make_player_level_final(results, query_year, homeaway):
#     from functools import reduce
#     data_frames = []
#     for category in ['defensive', 'rushing', 'passing', 'receiving']:
#         print('doing: ', category)
#         df = smash_df(category=category, results=results, query_year=query_year, homeaway=homeaway)
# #         df = df.loc[:,~df.columns.str.startswith('gameId')]
#         df = df[[col for col in df.columns if 'Id' not in col]]
#         data_frames.append(df)
#     df_merged = reduce(lambda  left, right: pd.merge(left, right, right_index=True, left_index=True,
#                                             how='outer'), data_frames)
#     return df_merged


def make_player_level_final(results, query_year, homeaway, category):
    print('doing: ', category)
    df = smash_df(category=category, results=results, query_year=query_year, homeaway=homeaway)
    df = df[[col for col in df.columns if 'Id' not in col]]
    return df

def fix_dataframe(df_merged):
    df_merged = df_merged.reset_index()
    df_merged.columns = df_merged.columns.map('|'.join).str.strip('|')
    df_merged = df_merged.drop([col for col in df_merged.columns if 'team' in col][2:], axis=1)
    df_merged = df_merged.drop([col for col in df_merged.columns if 'Id' in col][1:], axis=1)
    return df_merged

for year in [2021, 2016]:
    query_year = year

    key = "Bearer PNkaVR4Ti/Mr9d+LMxa875mI9Yvl5fw8EvZRuJDWnC6KN+1O1Cl3g5tB/YLq8hix"
    header = {'Authorization':key}
    params = {'year':query_year}


    response = requests.get("http://api.collegefootballdata.com/games",
                            headers=header,
                            params=params)

    response_json = response.json()

    results = pd.json_normalize(response_json)

    df_list = []
    for category in ['defensive', 'passing', 'receiving', 'rushing']:
        df = make_player_level_final(results=results, query_year=query_year, homeaway='away', category=category)
        df = fix_dataframe(df)
        df_list.append(df)

# df_merged = reduce(lambda  left, right: pd.merge(left, right, right_index=True, left_index=True,
#                     how='outer'), df_list)


# df_merged.to_csv(f"player_level_{query_year}_.csv", index=False)


# results_2018 = results_2018[results_2018['week'] <= 1]
# away_17 = make_player_level_final(results=results_2018, query_year=query_year, homeaway='away')

# away_17 = away_17.drop([col for col in away_17.columns if 'Id' in col][2:], axis=1)       
# away_17 = away_17.reset_index()
# away_17.columns = away_17.columns.map('|'.join).str.strip('|')
# away_17 = away_17.drop([col for col in away_17.columns if 'team' in col][2:], axis=1)
# away_17 = away_17.drop([col for col in away_17.columns if 'Id' in col][1:], axis=1)
# away_17.to_csv(f"player_level_{query_year}_.csv", index=False)


