import openml

from utils import base


to_save = base / 'flows.csv'
flows = openml.flows.list_flows(output_format='dataframe')
flows.to_csv(to_save, index=False)
