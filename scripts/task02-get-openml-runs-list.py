import dill as pickle
import pandas as pd
import openml

from utils import base


def check(dependency, pattern):
    return not dependency.startswith(pattern)


flows = pd.read_csv(base / 'flows.csv')
to_save = base / 'runs-list.csv'

run_chunks = []
sklearn_in_version = flows.external_version.apply(lambda x: 'sklearn' in str(x))
reduced_flows = flows[sklearn_in_version]
reduced_flows = reduced_flows[reduced_flows.uploader != 6138] # No Felix
print(flows.shape, reduced_flows.shape)

for idx, flow in reduced_flows.iterrows():
    not_supported = False
    if flow.external_version is not None:
        for dep in flow.external_version.split(','):
            if not (check(dep, 'R_')
                    and check(dep, 'Weka_')
                    and check(dep, 'Moa_')):
                not_supported = True
                break
    if not_supported:
        continue

    runs_per_flow = openml.runs.list_runs(flow=[flow.id], output_format='dataframe')
    number_of_runs = runs_per_flow.shape[0]
    if number_of_runs == 0:
        continue

    run_chunks.append(runs_per_flow)

runs = pd.concat(run_chunks)
runs.to_csv(to_save, index=False)
