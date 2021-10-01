import dill as pickle

from utils import base, timeit, FEATURE_ORDER


class DataSummarizer:
    def __init__(self):
        with open(base / 'norms.pkl', 'rb') as f:
            self.norms = pickle.load(f)
        self.feature_order = FEATURE_ORDER

    @timeit
    def compute_for_did(self, did, reload=False):
        f = base / f'datasets/{did}/metafeatures.pkl'
        # TODO: copy/paste a version that computes on the fly
        if reload or not f.exists():
            folder = f.parents[0]
            assert folder.exists()
            assert (folder / 'features.csv').exists()
            assert (folder / 'target.csv').exists()
            assert (folder / 'name').exists()

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
            start = time.time()
            features = calculate_all_metafeatures(
                X, y.to_numpy().reshape((-1,)), categorical,
                dataset_name, logger, calculate=FEATURE_ORDER)
            delta = time.time() - start
            with open(base / 'exec_times_metafeatures.txt', 'a') as file:
                file.write(f'{idx},{delta}\n')
            with open(f, 'wb') as file:
                pickle.dump(features, file)
        else:
            with open(f, 'rb') as file:
                features = pickle.load(file)
        return [features.metafeature_values[k].value / n
                for (n, k) in zip(self.norms, self.feature_order)]
