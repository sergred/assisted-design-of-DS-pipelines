from scipy.stats import percentileofscore as whatpercentile
from collections import namedtuple, defaultdict
from itertools import product, repeat
from pathlib import Path
from enum import Enum
import dill as pickle
import networkx as nx
import pandas as pd
import numpy as np
import traceback
import time
import sys

from utils import timeit
from experiment_database import ExpDB
from data_summarizer import DataSummarizer
from recommendation_engine import RecommendationEngine


Dataset = namedtuple('Dataset', ['did', 'data'])
Query = namedtuple('Query', ['did', 'task', 'offset', 'eval'])

class Eval(Enum):
    Manual = 1


class Task(Enum):
    Classification = "Supervised Classification"


@timeit
def run_query(user_query: Query, config: dict = {}):
    recsys = RecommendationEngine()
    expdb = recsys.expdb

    failed_exec = ([int(l.split(',')[0]) for l in open('failed_exec.txt').read().splitlines() if int(l.split(',')[-1]) == user_query.did]
                   + [int(l.split(',')[0]) for l in open('screwed.txt').read().splitlines() if int(l.split(',')[2]) == user_query.did])
    run_exec = self.runs[self.runs['dataset_id'] == user_query.did].flow_id.unique()
    failed_exec = list(set(failed_exec) - set(run_exec))

    dprofile = DataSummarizer().compute_for_did(user_query.did)
    similar_dids, dist = expdb.get_similar_data(dprofile)
    pool_of_candidates = expdb.get_relevant_runs(similar_dids, user_query.task)
    config['ddist'] = [Dataset(i, d) for i, d in zip(similar_dids, dist)]

    suggestions = recsys.suggest(pool_of_candidates, config)
    for candidate in suggestions:
        print(candidate.fid, candidate.flow)


if __name__ == "__main__":
    user_query = Query(did=1049,
                       task=Task.Classification,
                       offset=None,
                       eval=Eval.Manual)
    config = {
        'num_candidates': 5,
    }
    run_query(user_query, config)
