from collections import defaultdict
from scipy.spatial import kdtree
from pathlib import Path
import dill as pickle
import pandas as pd
import numpy as np
import sys

from utils import base, FEATURE_ORDER

to_save = base / 'openml-datasets-CC18.csv'

# PATCH: module-level attribute to enable pickle to work
kdtree.node = kdtree.KDTree.node
kdtree.leafnode = kdtree.KDTree.leafnode
kdtree.innernode = kdtree.KDTree.innernode

openml_id_mapping = []
metafeatures = []

files = list(Path(base / 'datasets/').glob('**/metafeatures.pkl'))
print(len(files))
print(sum([1 for f in files if f.stat().st_size != 0]))

res = defaultdict(list)
for f in files:
    if f.stat().st_size == 0:
        continue
    openml_id_mapping.append(int(f.parents[0].stem))
    with open(f, 'rb') as g:
        data = pickle.load(g)
    metafeatures.append([data.metafeature_values[k].value for k in FEATURE_ORDER])

# pd.DataFrame(res).to_csv(base / 'data_profiles.csv', index=False)
print(np.logical_or.reduce(pd.isna(metafeatures), axis=0))

# BUG: NaNs replaced with 0?
from sklearn.preprocessing import normalize
normalized, norms = normalize(np.nan_to_num(metafeatures), norm='max', axis=0, return_norm=True)       

tree = kdtree.KDTree(normalized)
with open(base / 'kdtree.pkl', 'wb') as f:
    pickle.dump(tree, f)

with open(base / 'norms.pkl', 'wb') as f:
    pickle.dump(norms, f)

with open(base / 'openml_id_mapping.pkl', 'wb') as f:
    pickle.dump(openml_id_mapping, f)
