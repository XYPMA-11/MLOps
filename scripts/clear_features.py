import pandas as pd
import os


df = pd.read_csv('/home/xypma/project/MLOps/data/stage1/train.csv')

df = df.drop(columns=['gameID', 'teamID', 'season', 'date', 'homeTeamID', 'awayTeamID', 'location', 'homeGoalsHalfTime', 'awayGoalsHalfTime', 'homeProbability', 'drawProbability', 'awayProbability', 'B365H', 'B365D', 'leagueID', 'B365A', 'homeGoals', 'awayGoals'])

os.makedirs(os.path.join("data", "stage2"), exist_ok=True)
df.to_csv('/home/xypma/project/MLOps/data/stage2/train.csv', encoding='utf-8')
