#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import signal
import dabl
import time
import sys
import dill as pickle

from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures

from utils import base, get_logger, FEATURE_ORDER

folder = base / 'datasets/'
logger = get_logger('log/datasets.log')


def handler(signum, frame):
    raise TimeoutError("Takes too long")


signal.signal(signal.SIGALRM, handler)

res = defaultdict(list)
for i, folder in enumerate(sorted(folder.iterdir(), key=lambda x: int(x.stem))):
    if not folder.exists():
        continue
    if not (folder / 'features.csv').exists():
        continue
    if not (folder / 'target.csv').exists():
        continue
    if not (folder / 'name').exists():
        continue
    did = int(folder.stem)

    dataset_name = open(folder / 'name').read().strip()
    X = pd.read_csv(folder / 'features.csv')
    X.infer_objects()
    numerics = (X.dtypes != 'object').to_numpy()
    strings = (X.dtypes == 'object').to_numpy()
    y = pd.read_csv(folder / 'target.csv')
    types = dabl.detect_types(X, near_constant_threshold=1.1)
    types = types[['categorical', 'free_string', 'useless']].to_numpy()
    categorical = np.logical_or.reduce(types, axis=1)
    categorical = dict(zip(types.index, categorical))
    try:
        signal.alarm(5*60)
        start = time.time()
        features = calculate_all_metafeatures(X, y.to_numpy().reshape((-1,)),
                                              categorical, dataset_name, logger,
                                              calculate=FEATURE_ORDER)
        delta = time.time() - start
        with open(base / 'exec_times_metafeatures.txt', 'a') as file:
            file.write(f'{idx},{delta}\n')
        with open(folder / f'metafeatures.pkl', 'wb') as f:
            pickle.dump(features, f)
        res['dataset_id'].append(int(idx))
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        print(did, traceback.print_exc(1))
        pass
