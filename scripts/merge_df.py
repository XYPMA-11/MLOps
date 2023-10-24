import pandas as pd
import os

df_games = pd.read_csv('/home/xypma/project/MLOps/data/raw/games.csv')
df_teams = pd.read_csv('/home/xypma/project/MLOps/data/raw/teams.csv')
df_teamstats = pd.read_csv('/home/xypma/project/MLOps/data/raw/teamstats.csv')


merged = pd.merge(df_games, df_teamstats, on=['season', 'gameID', 'date'], how='left')

s = merged.groupby(['gameID']).agg(total_goals = ('goals','sum'), shots = ('shots','sum'), corners = ('corners','sum'),  fouls = ('fouls','sum'), yellowCards = ('yellowCards','sum'), redCards = ('redCards','sum')).reset_index()

merged = merged.drop(columns=['goals', 'shots', 'corners', 'fouls', 'yellowCards', 'redCards'])
merged = merged.drop_duplicates(subset=['gameID'])
merged = pd.merge(merged, s, on=['gameID'], how='inner')

c = df_teams.rename(columns = {"teamID" : "homeTeamID"})
merged = pd.merge(merged, c, on=['homeTeamID'], how='left')

merged = merged.rename(columns = {"name" : "homeTeam"})
c = df_teams.rename(columns = {"teamID" : "awayTeamID"})
merged = pd.merge(merged, c, on=['awayTeamID'], how='left')
merged = merged.rename(columns = {"name" : "awayTeam"})

os.makedirs(os.path.join("data", "stage1"), exist_ok=True)
merged.to_csv('/home/xypma/project/MLOps/data/stage1/train.csv', encoding='utf-8')

