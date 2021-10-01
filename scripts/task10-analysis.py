from collections import namedtuple, defaultdict
from itertools import product
from pathlib import Path
import dill as pickle
import pandas as pd
import numpy as np
import openml
import sys


from utils import base, FLOWS_TO_REMOVE

runs = base / 'runs-CC18/'
datasets = pd.read_csv(base / 'openml-datasets-CC18.csv')
flows = pd.read_csv(base / 'flows.csv')

tasks = openml.tasks.list_tasks(output_format='dataframe')
tasks.to_csv(base / 'tasks.csv', index=False)
runs_info = pd.read_csv(base / 'runs-list.csv')


if not (base / 'full.csv').exists:
    res = defaultdict(list)
    for r in runs_info.run_id:
        if not (runs / f'{r}.pkl').exists():
            continue
        with open(runs / f'{r}.pkl', 'rb') as g:
            run = pickle.load(g)
        res['run_id'].append(r)
        res['flow_id'].append(run.flow_id)
        res['dataset_id'].append(run.dataset_id)
        res['user_id'].append(run.uploader)
        res['task_id'].append(run.task_id)
        flow = run.flow.name if run.flow else run.flow_name
        res['flow'].append(flow)
        res['task'].append(run.task_type)
        res['auc_roc'].append(run.evaluations.get('area_under_roc_curve', None))
        res['accuracy'].append(run.evaluations.get('predictive_accuracy', None))
        res['rmse'].append(run.evaluations.get('root_mean_squared_error', None))
    pd.DataFrame(res).to_csv(base / 'full.csv', index=False)


res = defaultdict(list)
percentiles = np.arange(0, 105, 5)

df = pd.read_csv(base / 'full.csv')
for (flow_id, did), group in df.groupby(['flow_id', 'dataset_id']):
    res['run_id'].append(":".join(map(str, group.run_id)))
    res['flow_id'].append(flow_id)
    # res['rflow_id'].append(mapping.get(flow_id, None))
    # rflow_id = group['id'].astype(int).min()
    # res['rflow_id'].append(rflow_id)
    res['dataset_id'].append(did)
    res['user_id'].append(":".join(map(str, group['user_id'].unique())))
    res['task_id'].append(":".join(map(str, group['task_id'].unique())))
    res['flow'].append(group['flow'].iloc[0])
    res['task'].append(group['task'].iloc[0])
    res['auc_roc_mean'].append(np.nanmean(group['auc_roc']))
    res['auc_roc_std'].append(np.nanstd(group['auc_roc']))
    for metric in ['auc_roc', 'accuracy', 'rmse']:
        tmp = np.nanpercentile(group[metric], percentiles)
        for i, val in zip(percentiles, tmp):
            res[f'{metric}_{i}'].append(val)
pd.DataFrame(res).to_csv(base / f'short.csv', index=False)


flows = pd.read_csv(base / 'flows.csv')

sklearn_in_version = flows.external_version.apply(lambda x: 'sklearn' in str(x))
rflows = flows[sklearn_in_version]
rflows = rflows[rflows.uploader != 6138] # No Felix
rflows = rflows[~rflows.id.isin(FLOWS_TO_REMOVE)]
print(flows.shape, rflows.shape)

res = defaultdict(list)
mg = lambda x: ":".join(map(str, x))
mapping = {}
for name, group in rflows.groupby('name'):
    rflow_id = group['id'].astype(int).min()
    res['flow_id'].append(mg(group['id'].unique()))
    res['rflow_id'].append(rflow_id)
    res['name'].append(name)
    res['uploader'].append(mg(group['uploader'].unique()))
    res['num_duplicates'].append(group.shape[0])
    to_ignore = 0 if not 'TEST' in name else 1
    res['ignore'].append(to_ignore)
    if not to_ignore:
        for fid in group['id']:
            mapping[int(fid)] = rflow_id


df = pd.DataFrame(res)
df.to_csv(base / 'rflows.csv', index=False)

with open(base / 'duplicate_flows.pkl', 'wb') as file:
    pickle.dump(mapping, file)


df['rflow_id'] = df['rflow_id'].astype(int)
print(df)

df[df.rflow_id > 0].to_csv('runs.csv', index=False)


Run = namedtuple('Run', 'flow_id dataset_id user_id auc_roc')

df = pd.read_csv(base / 'short.csv')
tmp = df[['flow_id', 'dataset_id', 'user_id', 'auc_roc_mean']].drop_duplicates(['flow_id', 'dataset_id', 'user_id'])
print(tmp)
print(tmp[tmp.isna().any(axis=1)])

runs = [Run(r.flow_id, r.dataset_id, r.user_id, r.auc_roc_mean) for _, r in tmp.iterrows()]
res = defaultdict(list)
for one, another in product(runs, runs):
    if one == another:
        continue
    if one.dataset_id != another.dataset_id:
        continue
    res['dataset_id'].append(one.dataset_id)
    more = one if one.auc_roc > another.auc_roc else another
    less = another if more == one else one
    res['compared'].append(less.flow_id)
    res['preferred'].append(more.flow_id)
    res['less_metric'].append(less.auc_roc)
    res['more_metric'].append(more.auc_roc)

pd.DataFrame(res).to_csv(base / 'interaction_table.csv', index=False)
