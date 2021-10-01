from nds.ndomsort import non_domin_sort
from collections import defaultdict, namedtuple
import networkx as nx
import dill as pickle

from experiment_database import ExpDB
from utils import base, timeit

Candidate = namedtuple('Candidate', ['fid', 'flow', 'score'])


class RecommendationEngine:
    def __init__(self):
        self.expdb = ExpDB()

    def get_cperf(self, flow_id, did):
        return list(self.expdb.runs[(self.expdb.runs['rflow_id'] == flow_id)
                                    & (self.expdb.runs['dataset_id'] == did)].auc_roc_mean.unique())

    def get_slice_perf(self, flow_id, slice):
        preferred = slice[slice['preferred'] == flow_id].shape[0]
        compared = slice[slice['compared'] == flow_id].shape[0]
        cperf = 1. * preferred / (preferred + compared) if (preferred + compared) != 0 else .0
        return cperf

    # @timeit
    def item_distance(self, one, another):
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.optimize_graph_edit_distance.html#networkx.algorithms.similarity.optimize_graph_edit_distance
        folder = base / 'flows-as-graphs/'
        one_graph = pickle.load(open(folder / f'flow_{one}.pkl', 'rb'))
        another_graph = pickle.load(open(folder / f'flow_{another}.pkl', 'rb'))
        for v in nx.optimize_graph_edit_distance(one_graph, another_graph):
            return v

    @timeit
    def suggest(self, pool, config, offset=None, previous=defaultdict(int), discarded=[]):
        it = self.expdb.interaction_table
        similar_data = config.get('ddist', None)
        score = lambda x: np.mean(list(map(lambda y: y[0]*y[1], x)))

        if not offset:
            rs = defaultdict(list)
            for d in similar_data:
                it_slice = it[it['dataset_id'] == d.did]
                for candidate in pool:
                    if candidate in discarded:
                        continue
                    c = self.expdb.duplicate_flows.get(candidate, candidate)
                    cperf = self.get_slice_perf(c, it_slice)
                    rs[c].append((cperf, 1. - d.dist ))
            candidates = [Candidate(k, [score(v), self.get_slice_perf(k, it)],
                                    self.expdb.get_flow_name())
                                    for k, v in rs.items()]
        else:
            similar_pipelines = self.expdb.bktree_get_k(offset, 20, discarded)

            it_slice = it[it['dataset_id'].isin([d.did for d in similar_data])]
            candidates, max_pdist = [], 0

            for dist, candidate in similar_pipelines:
                c = self.expdb.duplicate_flows.get(candidate, candidate)
                if dist == 0.: # in case pipelines match
                    continue
                cperf = self.get_slice_perf(c, it[it['dataset_id'] == d.did])
                simd_cperf = score([(cperf, 1. - d.dist) for d in similar_data])
                pdist = [d for d, p in similar_pipelines if p == c][0] / max_pdist
                max_pdist = pdist if np.abs(pdist) > max_pdist else max_pdist
                flow = self.expdb.rflows[self.expdb.rflows['rflow_id'] == c]
                flow = flow.name.iloc[0] if not flow.empty else None
                candidates.append(
                    Candidate(c,
                              [-1 * previous.get(c, 0),
                              simd_cperf, self.get_slice_perf(c, it),
                              -1. * pdist], flow, None))
        fronts = non_domin_sort(candidates, get_objectives=lambda x: x.score)
        res = [candidate for fr in fronts for candidate in fronts[fr]][::-1]
        return res[:config.get('num_candidates', -1)]
