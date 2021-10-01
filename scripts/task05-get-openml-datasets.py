from openml import datasets, tasks, runs, flows, exceptions
from pathlib import Path
import pandas as pd
import traceback
import logging
import sys

from utils import base

# list_of_datasets = datasets.list_datasets()
# print(len(list_of_datasets), "datasets")
# datasets_info = pd.DataFrame.from_dict(list_of_datasets, orient='index')

datasets_info = pd.read_csv(base / 'task01-get-openml-flows/openml-datasets-CC18.csv')

base = base / 'source01-openml/datasets'
base.mkdir(parents=True, exist_ok=True)

# openml_screwed = [int(line) for line in open('data/openml-screwed.txt', 'r').read().split('\n')[:-1]]
openml_screwed = []

for idx, d in datasets_info.iterrows():
    if d.did in openml_screwed:
        continue
    try:
        folder = base / str(d.did)

        if (folder / 'name').exists() and (folder / 'features.csv') and (folder / 'target.csv').exists():
            print(d.did)
            continue

        folder.mkdir(parents=True, exist_ok=True)

        odata = datasets.get_dataset(d.did)
        target = odata.default_target_attribute

        if not (folder / 'name').exists():
            with open(folder / "name", 'w') as f:
                f.write(odata.name)

        if not (folder / 'features.csv').exists():
            X, y, _, attribute_names = odata.get_data(target=target)
            df = pd.DataFrame(X, columns=attribute_names)
            df.infer_objects()
            df.to_csv(folder / "features.csv", index=False)
        if not (folder / 'target.csv').exists():
            X, y, _, attribute_names = odata.get_data(target=target)
            pd.DataFrame(y, columns=[target]).to_csv(folder / "target.csv", index=False)

        print(d.did)
    except KeyboardInterrupt:
        sys.exit(0)
    except exceptions.OpenMLServerException:
        with open('openml-screwed.txt', 'a') as f:
            f.write(f'{d.did}\n')
    except:
        traceback.print_exc(1)
