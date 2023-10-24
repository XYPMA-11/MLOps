import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/xypma/project/MLOps/data/stage3/train.csv')

params = yaml.safe_load(open("params.yaml"))["split"]

p_split_ratio = params["split_ratio"]

train = df.drop(columns=['xGoals'])
y = df['xGoals']


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=p_split_ratio, shuffle=True, random_state=42)

os.makedirs(os.path.join("data", "stage4"), exist_ok=True)

pd.concat([y_train, X_train], axis=1).to_csv('/home/xypma/project/MLOps/data/stage4/train.csv', encoding='utf-8', index=None)
pd.concat([y_test, X_test], axis=1).to_csv('/home/xypma/project/MLOps/data/stage4/test.csv', encoding='utf-8', index=None)
