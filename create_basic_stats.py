import pandas as pd
import requests
import numpy as np
from datetime import datetime


def get_data(year, url):
    key = "Bearer PNkaVR4Ti/Mr9d+LMxa875mI9Yvl5fw8EvZRuJDWnC6KN+1O1Cl3g5tB/YLq8hix"
    header = {'Authorization':key}
    params = {'year':year}


    response = requests.get(f"http://api.collegefootballdata.com/{url}",
                            headers=header,
                        params=params)

    response_json = response.json()

    df = pd.json_normalize(response_json)
    return df

results_2010 = get_data(2010, 'games')
results_2011 = get_data(2011, 'games')
results_2012 = get_data(2012, 'games')
results_2013 = get_data(2013, 'games')
results_2014 = get_data(2014, 'games')
results_2015 = get_data(2015, 'games')
results_2016 = get_data(2016, 'games')
results_2017 = get_data(2017, 'games')
results_2018 = get_data(2018, 'games')
results_2019 = get_data(2019, 'games')
results_2020 = get_data(2020, 'games')
results_2021 = get_data(2021, 'games')

def make_home_json(json,i):
    s=pd.json_normalize(json).set_index("category").T

    s.reset_index(inplace=True)
    s.index = [response_json[i]['id']]
    s['number_of_pen'] = s['totalPenaltiesYards'].apply(lambda x: x.split('-')[0]).replace("", 0).astype(int).item()
    s['penaltyYards'] = s['totalPenaltiesYards'].apply(lambda x: x.split('-')[1]).replace("", 0).astype(int).item()
 
    s = s.drop("totalPenaltiesYards",axis=1)

    s['numberCompletions'] = s['completionAttempts'].apply(lambda x: x.split('-')[0]).astype(int).item()
    s['numberAttempts'] = s['completionAttempts'].apply(lambda x: x.split('-')[1]).astype(int).item()
    s['completionPct'] = s['numberCompletions'] / s['numberAttempts']

    s = s.drop("completionAttempts", axis=1)

    s['fourthDownSuc'] = s['fourthDownEff'].apply(lambda x: x.split('-')[0]).astype(int).item()
    s['fourthDownAtt'] = s['fourthDownEff'].apply(lambda x: x.split('-')[1]).astype(int).item()
    s['fourthDownSucPct'] = s['fourthDownSuc'] / s['fourthDownAtt']

    s = s.drop("fourthDownEff", axis=1)

    s['thirdDownSuc'] = s['thirdDownEff'].apply(lambda x: x.split('-')[0]).astype(int).item()
    s['thirdDownAtt'] = s['thirdDownEff'].apply(lambda x: x.split('-')[1]).astype(int).item()
    s['thirdDownSucPct'] = s['thirdDownSuc'] / s['thirdDownAtt']
    
    s = s.drop("thirdDownEff", axis=1)
    return s

def make_away_json(json,i):
    s=pd.json_normalize(json).set_index("category").T
    s.reset_index(inplace=True)

    s.index = [response_json[i]['id']]

    s['number_of_pen'] = s['totalPenaltiesYards'].apply(lambda x: x.split('-')[0]).replace("", 0).astype(int).item()
    s['penaltyYards'] = s['totalPenaltiesYards'].apply(lambda x: x.split('-')[1]).replace("", 0).astype(int).item()

    s = s.drop("totalPenaltiesYards",axis=1)

    s['numberCompletions'] = s['completionAttempts'].apply(lambda x: x.split('-')[0]).astype(int).item()
    s['numberAttempts'] = s['completionAttempts'].apply(lambda x: x.split('-')[1]).astype(int).item()
    s['completionPct'] = s['numberCompletions'] / s['numberAttempts']

    s = s.drop("completionAttempts", axis=1)

    s['fourthDownSuc'] = s['fourthDownEff'].apply(lambda x: x.split('-')[0]).astype(int).item()
    s['fourthDownAtt'] = s['fourthDownEff'].apply(lambda x: x.split('-')[1]).astype(int).item()
    s['fourthDownSucPct'] = s['fourthDownSuc'] / s['fourthDownAtt']

    s = s.drop("fourthDownEff", axis=1)

    s['thirdDownSuc'] = s['thirdDownEff'].apply(lambda x: x.split('-')[0]).astype(int).item()
    s['thirdDownAtt'] = s['thirdDownEff'].apply(lambda x: x.split('-')[1]).astype(int).item()
    s['thirdDownSucPct'] = s['thirdDownSuc'] / s['thirdDownAtt']

    s = s.drop("thirdDownEff", axis=1)
    return s

# algo works, need to establish which columns bc they not only cary in order but in amt


#use 2016 and on
trainset = []
testset = []

for year_, results in zip([2016, 2017, 2018, 2019, 2020, 2021], [results_2016, results_2017, results_2018, results_2019, results_2020, results_2021]):
    main_home = []
    main_away = []
    
    for year, result in zip([year_], [results]):
        
        big_home = []
        big_away = []

        for i in range(1, np.max(result['week'])):
            key = "Bearer PNkaVR4Ti/Mr9d+LMxa875mI9Yvl5fw8EvZRuJDWnC6KN+1O1Cl3g5tB/YLq8hix"
            header = {'Authorization':key}
            params = {'year':year, 'week':i, 'categories':'defensive'}
            response = requests.get("http://api.collegefootballdata.com/games/teams",
                                    headers=header,
                                    params=params)
            response_json = response.json()
            
            h_l = []
            a_l = []
            
            c = 0
            
            for matchup in response_json:
                
                sample = pd.json_normalize(matchup['teams'])
                
                aw = sample[sample['homeAway'] == 'away']
                ho = sample[sample['homeAway'] == 'home']
                hom_Df = make_home_json(ho['stats'].item(),c)
                awa_Df = make_away_json(aw['stats'].item(),c)
                hom_Df['team'] = ho['school'].item()
                awa_Df['team'] = aw['school'].item()
                hom_Df['week'] = i
                awa_Df['week'] = i
                h_l.append(hom_Df)
                a_l.append(awa_Df)
                
                c+=1
                
            big_home.append(pd.concat(h_l))
            big_away.append(pd.concat(a_l))
            
        HOME=pd.concat(big_home)
        AWAY=pd.concat(big_away)
        main_home.append(HOME)
        main_away.append(AWAY)


    HOME = pd.concat(main_home)
    AWAY = pd.concat(main_away)

    HOME_features = HOME.drop("index", axis=1)
    AWAY_features = AWAY.drop("index", axis=1)

    # random_cols = ['kickReturnYards', 'kickReturnTDs', 'kickReturns', 'kickingPoints', 'fumblesRecovered', 'totalFumbles', 'fumblesLost',
    # 'puntReturnYards', 'puntReturnTDs', 'puntReturns', 'interceptionYards', 'interceptionTDs']

    # HOME_features = HOME_features.drop(random_cols, axis=1)
    # AWAY_features = AWAY_features.drop(random_cols, axis=1)

    HOME_features = HOME_features[HOME_features['possessionTime'].isna() == False]
    AWAY_features = AWAY_features[AWAY_features['possessionTime'].isna() == False]

    def time_to_sec(time_str):
        return sum(x * int(t) for x, t in zip([1, 60, 3600], reversed(time_str.split(":"))))

    HOME_features['possessionTime'] = HOME_features['possessionTime'].apply(lambda x: time_to_sec(x)) / 60
    AWAY_features['possessionTime'] = AWAY_features['possessionTime'].apply(lambda x: time_to_sec(x)) / 60

    home_imputed_features = HOME_features.fillna(0)
    away_imputed_features = AWAY_features.fillna(0)

    home_imputed_df = pd.DataFrame(home_imputed_features, columns=HOME_features.columns, index=HOME_features.index)
    away_imputed_df = pd.DataFrame(away_imputed_features, columns=AWAY_features.columns, index=AWAY_features.index)

    save_df = pd.concat((home_imputed_df, away_imputed_df))
    print(save_df.reset_index().columns)
    print(save_df.shape, f"saving {year}...")
    save_df.to_csv(f"basic_team_stats_{year_}.csv")