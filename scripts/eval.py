import os
import pickle
import json
import pandas as pd


input_file_test = os.path.join("data", "stage4", "test.csv")
input_file_model = os.path.join("models", "model.pkl")
output_file = os.path.join("metrics", "evaluation.json")
os.makedirs(os.path.join("metrics"), exist_ok=True)


df = pd.read_csv(input_file_test)

x = df.drop(columns=['xGoals'])
y = df['xGoals']

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

metr=0

with open(input_file_model, "rb") as fd:
	unpickler = pickle.Unpickler(fd)
	tree = unpickler.load()
	scr = tree.score(x, y)
	metr = scr
	
with open(output_file, "w") as f:
	res = { "score": metr }
	json.dump(res, f)	
