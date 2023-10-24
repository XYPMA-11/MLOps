import pandas as pd
import os

df = pd.read_csv('/home/xypma/project/MLOps/data/stage2/train.csv')

#df['result'] = df['result'].astype('int')
check = df
check.loc[check["result"] == "W", "result"] = 1
check.loc[check["result"] == "L", "result"] = 2
check.loc[check["result"] == "D", "result"] = 0
check['result'] = check['result'].astype('int')
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)
check.to_csv('/home/xypma/project/MLOps/data/stage3/train.csv', encoding='utf-8')

