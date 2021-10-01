from redbaron import RedBaron
from pathlib import Path
from uuid import uuid4
import dill as pickle
import networkx as nx
import pandas as pd
import traceback
import openml
import sys

from baron.render import RenderWalker
from baron.render import node_types

from ast import literal_eval
from utils import base

def dumps(tree):
    return Dumper().dump(tree.fst())


def uid():
    return str(uuid4())[:6] + ":"


class Dumper(RenderWalker):
    def before_node(self, node, key):
        mapping = {
            'name': lambda n: n['value'],
            'dot': lambda n: '.',
            'atomtrailers': lambda n: None,
            'call': lambda n: 'CALL',
            'call_argument': lambda n: 'TARGET',
            'comma': lambda n: 'COMMA',
        }

        type = node.get('type', None)
        res = mapping.get(type, lambda n: n)(node)
        if res:
            if res in 'CALL':
                self.dump.add_node(self.cur)
                if len(self.root):
                    self.dump.add_edge(self.root[-1], self.cur)
                self.root.append(self.cur)
                self.cur = uid()
            elif res in 'COMMA':
                self.dump.add_node(self.cur)
                if len(self.root):
                    self.dump.add_edge(self.root[-1], self.cur)
                self.cur = uid()
            elif res == 'TARGET':
                pass
            else:
                self.cur += res

    def dump(self, tree):
        self.dump = nx.DiGraph()
        self.root = []
        self.cur = uid()
        self.walk(tree)
        if not self.cur.endswith("::"):
            self.dump.add_node(self.cur)
            if self.root:
                self.dump.add_edge(self.root[-1], self.cur)
        return self.dump


def is_dict(v):
    try:
        evald = literal_eval(v)
        return True if isinstance(evald, dict) else False
    except ValueError:
        return False


folder = base / 'flows-as-graphs'
folder.mkdir(parents=True, exist_ok=True)

flows = pd.read_csv(base / 'flows.csv')
sklearn_in_version = flows.external_version.apply(lambda x: 'sklearn' in str(x))
flows = flows[sklearn_in_version]
flows = flows[flows.uploader != 6138] # No Felix
for idx, fl in flows.drop_duplicates('name').iterrows():
    flow = openml.flows.get_flow(fl.id)
    if (folder / f'flow_{fl.id}.pkl').exists():
        continue
    try:
        tree = RedBaron(flow.name)
        G = dumps(tree)
        params = [(k, v) for k, v in flow.parameters.items()
                  if (v is not None) and ('oml-python' not in v)
                  and (k not in ['steps', 'memory', 'verbose'])
                  and not is_dict(v)]
        if params:
            print(flow.name)
            print(params)
            root = [ n for n, d in G.in_degree() if d==0 ][0]
            _uid = root[:6]
            G.add_node(f'{_uid}:params')
            G.add_edge(root, f'{_uid}:params')
        for idx, (k, v) in enumerate(params):
            # G.add_node(f'param_{idx}')
            G.add_node(f'{_uid}:{idx}:key:{k}')
            G.add_node(f'{_uid}:{idx}:val:{v}')
            # G.add_edge('params', f'param_{idx}')
            G.add_edge(f'{_uid}:params', f'{_uid}:{idx}:key:{k}')
            G.add_edge(f'{_uid}:{idx}:key:{k}', f'{_uid}:{idx}:val:{v}')
        with open(folder / f'flow_{fl.id}.pkl', 'wb') as f:
            pickle.dump(G, f)
        print(fl.id)
    except:
        traceback.print_exc(1)
        continue
