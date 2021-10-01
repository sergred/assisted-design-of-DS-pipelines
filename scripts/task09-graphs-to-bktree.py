from collections import namedtuple
from pathlib import Path
from pybktree import BKTree
import networkx as nx
import pandas as pd
import dill as pickle
import traceback
import time
import sys

sys.setrecursionlimit(10**6)

from utils import timeit, base, FLOWS_TO_REMOVE


Item = namedtuple('Item', 'id graph')
folder = base / 'flows-as-graphs/'


# @timeit
def item_distance(one, another):
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.optimize_graph_edit_distance.html#networkx.algorithms.similarity.optimize_graph_edit_distance
    one_graph = pickle.load(open(folder / f'flow_{one}.pkl', 'rb'))
    another_graph = pickle.load(open(folder / f'flow_{another}.pkl', 'rb'))
    for v in nx.optimize_graph_edit_distance(one_graph, another_graph):
        return v


graphs = [g for g in folder.listdir()
          if int(g.stem.split('_')[-1]) not in FLOWS_TO_REMOVE]
to_save = base / 'bktree.pkl'

k = 2
start = time.time()
bktree = BKTree(item_distance, graphs[:k])
for idx, graph in enumerate(graphs[k:]):
    st = time.time()
    bktree.add(graph)
    print(idx, time.time() - st)
print("Calculated in", (time.time() - start) / 3600)

try:
    with open(to_save, 'wb') as f:
        pickle.dump(bktree, f)
except:
    traceback.print_exc(1)
