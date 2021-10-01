import dill as pickle
import pandas as pd

from utils import base, timeit


class ExpDB:
    def __init__(self):
        with open(base / 'duplicate_flows.pkl', 'rb') as f:
            self.duplicate_flows = pickle.load(f)
        self.runs = pd.read_csv(base / 'runs.csv')
        rflows = pd.read_csv(base / 'rflows.csv')
        self.rflows = rflows[rflows.ignore == 0]
        self.interaction_table = pd.read_csv(base / 'interaction_table.csv')
        with open(base / 'kdtree.pkl', 'rb') as f:
            self.kdtree = pickle.load(f)
        with open(base / 'openml_id_mapping.pkl', 'rb') as f:
            self.openml_did_mapping = pickle.load(f)
        with open(base / 'bktree.pkl', 'rb') as f:
            self.bktree = pickle.load(f)

    @timeit
    def bktree_get_k(self, item, limit, discarded=[]):
        # bktree API suports the tolerance parameter n (i.e., how similar
        # pipelines should be) instead of k for kNN search, workaround
        def dedup(v):
            return self.duplicate_flows.get(v, v)

        n, res = 2*limit, []
        while len(res) < limit:
            res = list(set([(k, dedup(v)) for k, v in self.bktree.find(item, n)
                            if (k != 0.0) and (dedup(v) not in discarded)]))
            n *= 2
        return res[:limit]

    def get_similar_data(self, data_profile,
                         limit_datasets=3,
                         return_distances=True):
        dist, indices = self.kdtree.query(data_profile, limit_datasets + 1)
        similar_dids = [self.openml_did_mapping[i] for i in indices]
        dist, indices = dist[1:], indices[1:]
        return (similar_dids, dist) if return_distances else similar_dids

    def get_flow_name(self, flow_id):
        return self.runs[self.runs['rflow_id'] == flow_id].flow.iloc[0]

    def get_relevant_runs(self, similar_dids, task):
        is_data = self.runs['dataset_id'].isin(similar_dids)
        is_task = self.runs['task'] == task
        return self.runs[is_data & is_task]
