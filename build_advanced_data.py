import pandas as pd
import requests
import numpy as np

week = int(input("What is the current week of CFB: "))

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

def create_betting(year, url):
    spreads = []
    ou = []
    
    betting = get_data(year, url)
    
    spreads = []
    ou = []
    for i in range(len(betting)):
        try:
            spreads.append(pd.json_normalize(betting['lines'][i])[['spread']].astype(float).median().item())
        except:
            spreads.append(pd.NA)
        try:
            ou.append(pd.json_normalize(betting['lines'][i])[['overUnder']].astype(float).median().item())
        except:
            ou.append(pd.NA)

    betting['spread'] = spreads
    betting['ou'] = ou

    return betting

def create_advanced(input_df, results, week):
    input_df = input_df[input_df['week'] < week]
    
    input_df = input_df.drop(['conference', 'season'], axis=1)
    
    input_df.iloc[:, 2:] = input_df.iloc[:, 2:].astype(float)
    
    latest_data = input_df.sort_values(by=['gameId', 'team']).drop("gameId", axis=1)

    latest_data_teams = latest_data.team

    latest_data = latest_data.groupby("team").transform(lambda x: x.expanding().mean() if x.dtype != object else x)
    
    latest_data = pd.merge(latest_data_teams, latest_data, right_index=True, left_index=True)
    
    latest_data = latest_data.sort_values(by=['team', 'week']).groupby("team").tail(1)
    return latest_data

def build_advanced(input_dataframe, results, func):
    large_pandas = []
    for i in range(2, np.max(results['week'])):
        twentyone_TEST = func(input_dataframe, results, i)

#         print(twentyone_TEST.shape)
        large_scaled = []
        large_probs = []
        week = i

        h_list = []
        a_list = []
        id_list = []

        results_week = results[results['week'] == week]
        for home_team, away_team in zip(results_week['home_team'], results_week['away_team']):

            game_index = results[(results['week'] == week) & (results['home_team'] == home_team) 
                                      & (results['away_team'] == away_team)]['id']

            home__team = twentyone_TEST[twentyone_TEST['team'] == home_team].sort_index().tail(1)
            away__team = twentyone_TEST[twentyone_TEST['team'] == away_team].sort_index().tail(1)

            h_list.append(home__team)
            a_list.append(away__team)
            id_list.append(game_index)

        home_df = pd.concat(h_list)
        away_df = pd.concat(a_list)
        # home_df.index = [i.values[0] for i in id_list]
        # away_df.index = [i.values[0] for i in id_list]
        away_df = pd.merge(away_df.rename(columns={'team' : 'away_team'}), results_week[['away_team', 'id']], on='away_team').set_index('id')\
        .rename(columns={'away_team' : 'team'})

        home_df = pd.merge(home_df.rename(columns={'team' : 'home_team'}), results_week[['home_team', 'id']], on='home_team').set_index('id')\
        .rename(columns={'home_team' : 'team'})

        totl = pd.merge(home_df, away_df, left_index=True, right_index=True, how='inner')

        game_id_index = totl.index

        conference_dataframe = results[['home_team', 'home_conference']].rename(columns={'home_team' : 'team',
                                                                                              'home_conference' : 'conference'})


        home = results.groupby("home_team")['home_conference'].\
        last().reset_index().rename(columns={'home_conference' : 'conference', 'home_team' : 'team'})

        home_conf_dummies = pd.get_dummies(conference_dataframe[conference_dataframe['team'].isin(home['team'])]['conference'])

        away = results.groupby("away_team")['away_conference'].\
        last().reset_index().rename(columns={'away_conference' : 'conference', 'away_team' : 'team'})

        away_conf_dummies = pd.get_dummies(conference_dataframe[conference_dataframe['team'].isin(away['team'])]['conference'])


        all_conf = pd.concat((home, away))

        total_conference = pd.concat((all_conf['team'], pd.get_dummies(all_conf['conference'])), axis=1)

        total_conference = total_conference.groupby("team").first().reset_index()

        home_teams = total_conference[total_conference['team'].isin(totl['team_x'])].rename(columns={'team' : 'team_x'})

        away_teams = total_conference[total_conference['team'].isin(totl['team_y'])].rename(columns={'team' : 'team_y'})

        new_names = [(i,i+'_away') for i in away_teams.columns[1:]]

        away_teams.rename(columns = dict(new_names), inplace=True)

        totl['id'] = game_id_index
        totl = pd.merge(totl, home_teams, on='team_x', how='inner')

        totl = pd.merge(totl, away_teams, on='team_y', how='inner')

#         totl['id'] = game_id_index
        totl['week'] = i
        large_pandas.append(totl) 
    return pd.concat(large_pandas)

def clean_advanced(advanced):
    advanced = advanced.drop(['team_x', 'team_y', 'week_x', 'week_y', 'week'], axis=1)
    return advanced

results_2022 = get_data(2021, 'games')
betting_2022 = create_betting(2021, 'lines')
advanced1_2022 = get_data(2021, 'ppa/games')
advanced2_2022 = get_data(2021, 'stats/game/advanced')

advanced_2022 = pd.merge(advanced2_2022.drop(["week", "opponent"], axis=1), 
                         advanced1_2022.drop("opponent", axis=1), on=['gameId', 'team'])


ttwo_advanced = build_advanced(input_dataframe = advanced_2022, results=results_2022, func=create_advanced)

ttwo_advanced_clean = clean_advanced(ttwo_advanced)

ttwo_advanced_clean.to_csv(f"weekly_data/advanced_2022_{week}.csv", index=False)

