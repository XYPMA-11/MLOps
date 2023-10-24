import pandas as pd
import yaml
import os
import pickle

df = pd.read_csv('/home/xypma/project/MLOps/data/stage4/train.csv')

x = df.drop(columns=['xGoals'])
y = df['xGoals']

params = yaml.safe_load(open("params.yaml"))["train"]

p_cv = params["cv"]
p_n_jobs = params["n_jobs"]
p_verbose = params["verbose"]

cat_columns = []
num_columns = []

for column_name in x.columns:
    if (x[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', MinMaxScaler())
])


categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent', )),
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

preprocessors = ColumnTransformer(transformers=[
    ('num', numerical_pipe, num_columns),
    ('cat', categorical_pipe, cat_columns)
])

preprocessors.fit(x)

x = preprocessors.transform(x)

from sklearn.tree import DecisionTreeRegressor

g = DecisionTreeRegressor(max_depth=6)
g.fit(x, y)

os.makedirs(os.path.join("models"), exist_ok=True)
f = os.path.join("models", "model")

with open(f, "wb") as fd:
	pickle.dump(g, fd)

