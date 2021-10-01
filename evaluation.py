import warnings
warnings.simplefilter("ignore", UserWarning)

from scipy.stats import percentileofscore as whatpercentile
from collections import namedtuple, defaultdict
from itertools import product, repeat
from pathlib import Path
from nds import ndomsort
import dill as pickle
import networkx as nx
import pandas as pd
import numpy as np
import traceback
import signal
import time
import sys
import ast

from utils import base, get_logger, timeit, pass_on_failure, repeat_on_failure


def handler(signum, frame):
    raise Exception('Takes too long.')


signal.signal(signal.SIGALRM, handler)


class EvaluationSuite:
    def __init__(self, recommendation_engine):
        self.engine = recommendation_engine
        self.runs = self.engine.expdb.runs

    @timeit
    def compute_run_shortcut(self, flow_id, did):
        all_flows = list(set([flow_id] + [k for k, v in self.engine.expdb.duplicate_flows.items() if v == flow_id]))
        task_ids = list(self.runs[(self.runs['flow_id'].isin(all_flows)) & (self.runs['dataset_id'] == did)].task_id.astype(int).unique())
        tasks = openml.tasks.get_tasks(task_ids)
        times = []
        done = open('done.txt').read().splitlines()
        for t, fid in product(tasks, all_flows):
            try:
                flow = openml.flows.get_flow(fid)
                model = openml.extensions.sklearn.SklearnExtension().flow_to_model(flow=flow, strict_version=False)
                start = time.time()
                run = openml.runs.run_model_on_task(model, t)
                times.append(time.time() - start)
                break
            except:
                print(f"Computing flow {fid} for dataset {did} failed.", traceback.format_exc().splitlines()[-1])
                _t = []
                for l in done:
                    if not l.startswith(f'{fid},'):
                        continue
                    if len(l.split(',')) != 4:
                        continue
                    fid, tid, _did, _time = l.split(',')
                    fsize = Path(f'intermediates/source01-openml/datasets/{_did}/features.csv').stat().st_size
                    print(fid, _did, _time, fsize, float(_time) / fsize)
                    _t.append(float(_time) / fsize)
                if _t:
                    dsize = Path(f'intermediates/datasets/{did}/features.csv').stat().st_size
                    times.append(dsize * np.nanmedian(_t))
        return np.nanmedian(times) if times else np.nan

    @timeit
    def compute_run(self, flow_id, did):
        all_flows = list(set([flow_id] + [k for k, v in self.engine.expdb.duplicate_flows.items() if v == flow_id]))
        task_ids = list(self.runs[(self.runs['flow_id'].isin(all_flows)) & (self.runs['dataset_id'] == did)].task_id.astype(int).unique())
        res = []
        if not task_ids:
            try:
                task = openml.tasks.create_task(openml.tasks.TaskType.SUPERVISED_CLASSIFICATION, did, 1, target_name=open(base / f'source01-openml/datasets/{did}/target.csv', 'r').readline().strip())
                task.publish()
            except openml.exceptions.OpenMLServerException as e:
                # Error code for 'task already exists'
                if e.code == 614:
                    task_id = ast.literal_eval(e.message.split("matched id(s):")[-1].strip())[0]                # Lookup task
            tasks = [openml.tasks.get_task(task_id)]
        else:
            tasks = openml.tasks.get_tasks(task_ids)
        for t, fid in product(tasks, all_flows):
            dep = None
            try:
                flow = openml.flows.get_flow(fid)
                dep = [f for f in flow.dependencies.split('\n') if f.startswith('sklearn==')][0]
                model = openml.extensions.sklearn.SklearnExtension().flow_to_model(flow=flow, strict_version=False)
                run = openml.runs.run_model_on_task(model, t)
                run.publish()
                print(run.evaluations)
                res.append(run.evaluations.get('area_under_roc_curve', None))
            except:
                if dep:
                    with open(dep, 'a') as file:
                        file.write(f'{fid},{did},{t.task_id}\n')
                print(f"Computing flow {fid} for dataset {did} failed.", traceback.format_exc().splitlines()[-1])
                continue
        return res

    def execute(self, config):
        res = {}
        for f in (base / 'task07-analysis/runs-per-dataset/').iterdir():
            if f.stem.startswith('full_'):
                continue
            if f.stem.startswith('short_'):
                continue
            #if int(f.stem) in [23517, 40670, 40966, 40668, 40975, 41027, 40701]:
            #    continue
            if not (base / f'source01-openml/datasets/{f.stem}/metafeatures.pkl').exists():
                continue
            signal.alarm(40)
            print(f.stem)
            res[int(f.stem)] = self.run_query(int(f.stem), config)
            signal.alarm(0)
        return res


    def eval_suggestions(self, user_query, suggestions):
        print("Suggested candidates: ", [c.fid for c in suggestions])
        failed_exec = list(set(([int(l.split(',')[0]) for l in open('failed_exec.txt').read().splitlines() if int(l.split(',')[-1]) == user_query.did]
                      + [int(l.split(',')[0]) for l in open('screwed.txt').read().splitlines() if int(l.split(',')[2]) == user_query.did])))
        #print(failed_exec)
        res = []
        eval_perf = self.runs[(self.runs['dataset_id'] == user_query.did) & (self.runs['task'] == user_query.task)].drop_duplicates().auc_roc_mean
        for idx, candidate in enumerate(suggestions):
            #if candidate.fid in failed_exec:
            #    continue
            c = self.engine.expdb.duplicate_flows.get(candidate.fid, None)
            if not c:
                all_flows = list(set([candidate.fid] + [k for k, v in self.engine.expdb.duplicate_flows.items() if v == candidate.fid]))
                task_ids = list(self.runs[(self.runs['flow_id'].isin(all_flows)) & (self.runs['dataset_id'] == user_query.did)].task_id.astype(int).unique())
                run_dict = openml.runs.list_runs(flow=[candidate.fid], task=task_ids)
                runs = openml.runs.get_runs(list(run_dict.keys()))
                cperf = np.nanmean([r.evaluations.get('area_under_roc_curve', None) for r in runs])
                with open('failed_exec.txt', 'a') as file:
                    file.write(f'{candidate.fid},{user_query.did}\n')
            else:
                cperf = self.engine.get_cperf(c, user_query.did)
            if cperf:
                cpercentile = whatpercentile(eval_perf, cperf[0]) # candidate relative performance
                print("#%d %d ROC AUC %.4f, %2.2f percentile, %d/%d place.\n%s\n" % (idx + 1, candidate.fid, cperf[0], cpercentile, eval_perf[eval_perf > cperf[0]].shape[0] + 1, eval_perf.shape[0], candidate.flow))
                approx = self.compute_run_shortcut(candidate.fid, user_query.did)
                res.append((c, cpercentile, eval_perf[eval_perf > cperf[0]].shape[0] + 1, eval_perf.shape[0], approx, cperf))
            else:
                res.append((c, np.nan, 0, 0, np.nan, np.nan))
                # print("Compute failed.\n")
                # print(f'No data about candidate {candidate.fid} performance on dataset {user_query.did}.')
                print(candidate.fid, candidate.flow)
                with open('screwed.txt') as file:
                    report = file.read().splitlines()
                issues = [l for l in report if (int(l.split(',')[0]) == candidate.fid) and (int(l.split(',')[2]) == user_query.did)]
                #if issues:
                print("\n".join(issues))
                with open('failed_exec.txt', 'a') as file:
                    file.write(f'{candidate.fid},{user_query.did}\n')
                continue
        return res


def main():
    suite = EvaluationSuite(recommendation_engine)
#    for i in range(5):
#        for j in range(5):
    for strategy in ['a-star', 'breadth-first', 'depth-first', 'a-star', 'random']:
        config = {
            'num_candidates': 4,
            'global_limit': 20,
            'depth_limit': 2,
            'search-strategy': strategy,
        }

        with open(f'result_final_{config.get("search-strategy")}.pkl', 'wb') as file:
             pickle.dump(EvaluationSuite(dorian).execute(config), file)


if __name__ == "__main__":
    main()
