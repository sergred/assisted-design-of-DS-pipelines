import openml

from utils import base

df = openml.datasets.list_datasets(tag='OpenML-CC18', output_format='dataframe')
df.to_csv(base / 'openml-datasets-CC18.csv')
