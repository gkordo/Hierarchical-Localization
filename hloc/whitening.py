import argparse
from pathlib import Path
import h5py
import numpy as np

from . import logger


class PCALayer:
    def __init__(self, file_name=None, n_components=None, eps=1e-5):
        super(PCALayer, self).__init__()
        self.dims = n_components
        self.mean = None
        self.d = None
        self.V = None
        self.DVt = None
        self.eps = eps
        if file_name is not None:
            self.load(file_name)

    def save(self, file_name):
        np.savez_compressed(file_name, mean=self.mean.cpu().numpy(), d=self.d, V=self.V)

    def load(self, file_name):
        white = np.load(file_name)
        self.init_params(white["mean"], white["d"], white["V"])

    def init_params(self, mean, d, V, dtype=np.float32):
        self.d = d
        self.V = V

        eps = d.max() * self.eps
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        idx = np.argsort(d)[::-1][:self.dims]
        d = d[idx]
        V = V[:, idx]

        d = np.diag(1. / np.sqrt(d))

        self.mean = mean.astype(dtype)
        self.DVt = np.dot(d, V.T).astype(dtype)

    def fit(self, x):
        dtype = x.dtype
        mean = x.mean(axis=0)
        x = x - mean
        Xcov = np.dot(x.T, x)
        d, V = np.linalg.eigh(Xcov.astype(np.float32))
        self.init_params(mean, d, V, dtype)

    def transform(self, x):
        x = x - self.mean
        x = np.dot(self.DVt, x.T).T
        x = x / np.linalg.norm(x, axis=-1)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def main(feature_path, n_components=None, key='global_descriptor'):
    logger.info('Load extracted descriptors')

    white_feature_path = Path(str(feature_path).replace('.h5', '_white.h5'))

    descs = []
    with h5py.File(str(feature_path), 'r') as fd:
        for k in fd['db'].keys():
            descs.append([fd['db'][k][key][:]])
    descs = np.concatenate(descs, 0)

    logger.info('Fit PCA Layer')
    pca = PCALayer(n_components=n_components)
    pca.fit(descs)

    logger.info('Store whitened descriptors')
    with h5py.File(str(feature_path), 'r') as fdr:
        with h5py.File(str(white_feature_path), 'w') as fdw:
            def visit_fn(name, obj):
                if isinstance(obj, h5py.Dataset):
                    value = obj[:]
                    if value.shape[0] > 2:
                        value = pca.transform(value)
                    fdw.create_dataset(name, data=value)
                else:
                    fdw.create_group(name)
            fdr.visititems(visit_fn)

    return white_feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=Path, required=True)
    parser.add_argument('--n_components', type=int, default=None)
    parser.add_argument('--desc_name', type=str, default='global_descriptor')
    args = parser.parse_args()
    main(args.feature_path, args.desc_name)
