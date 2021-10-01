import dill as pickle
import pandas as pd
import traceback
import openml
import time

from utils import base


step = 1000
to_save = base / 'runs-CC18'
to_save.mkdir(parents=True, exist_ok=True)
runs = pd.read_csv(base / 'runs-list.csv')
datasets = pd.read_csv(base / 'openml-datasets-CC18.csv').did.to_list()
run_ids = [r for r in runs.run_id if not (to_save / f'{r}.pkl').exists()]

print(runs.shape, len(run_ids))

count = 0
while count <= len(run_ids) // step + 1:
    start = time.time()
    repeat = True
    while repeat:
        try:
            n = step*(count+1) if step*(count+1) <= len(run_ids) else len(run_ids)
            batch = openml.runs.get_runs(run_ids[step*count:n])
            for r in batch:
                if r.dataset_id in datasets:
                    with open(to_save / f'{r.run_id}.pkl', 'wb') as f:
                        pickle.dump(r, f)

            print("Batch", count+1, "done in", (time.time() - start) / 60, "min.")
            repeat = False
            count += 1
        except:
            traceback.print_exc(1)
            time.sleep(60)
