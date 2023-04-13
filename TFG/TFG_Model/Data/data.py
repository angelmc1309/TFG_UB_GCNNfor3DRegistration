import os
from random import shuffle
from DataReader.read import DataReader
from math import floor
from scipy import sparse
from scipy.spatial.transform import Rotation as R

from time import time
import tensorflow as tf
import tensorflow_graphics as tfg
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *
from values import *

reader = DataReader()


class Data:
    # Reduce I/O operations at the cost of RAM
    _use_buffer = True
    _T_buffer = {}
    _F_buffer = {}
    _E_buffer = {}
    _L_buffer = {}

    _f2o = {  # fabric to one hot
        'cotton': [0, 0, 0, 1],
        'denim': [0, 0, 1, 0],
        'silk': [0, 1, 0, 0],
        'leather': [1, 0, 0, 0]
    }

    def __init__(self, txt, batch_size=1, mode='train'):
        # txt: path to .txt file with list of samples
        # batch_size: batch size
        # mode: 'train' for shuffle

        # Read sample list
        self._txt = txt
        samples = self._read_txt()
        # batch size
        self._batch_size = batch_size
        # Shuffle
        if mode == 'train':
            shuffle(samples)
        self._mode = mode
        self._samples = samples
        # Init
        self._idx = 0

    def _read_txt(self):
        # samples in txt as: "[SAMPLE_PATH]\t[N.FRAME]\n"
        with open(self._txt, 'r') as f:
            T = f.readlines()
        return [t.replace('\n', '').split('\t') for t in T]

    # reset when all samples read
    def _reset(self):
        self._idx = 0
        if self._mode == 'train': shuffle(self._samples)

    def _read_outfit(self, sample, info):
        path = SRC + sample + '/'
        path_preprocess = SRC_PREPROCESS + sample + '/'
        # read outfit garments in order
        with open(path_preprocess + 'outfit_verts.txt', 'r') as f:
            outfit = [t.replace('\n', '').split('\t') for t in f.readlines()]
        garments = [o[0] for o in outfit]
        subindices = [0] + [int(o[1]) for o in outfit]
        # read each garment vertices and concatenate
        """
        Read the garments from the .obj file
        """
        reader = DataReader()
        V = np.concatenate([readOBJ(os.path.join(reader.SRC, sample, garment + '.obj'))[0] for garment in garments],
                           axis=0)

        # V = np.concatenate([readPC2Frame(path + g + '.pc16', n, True) for g in garments], axis=0)
        # template
        if path_preprocess in self._T_buffer:
            T = self._T_buffer[path_preprocess]
        else:
            T = readPC2(path_preprocess + 'rest.pc16', True)['V'][0]
            if self._use_buffer: self._T_buffer[path_preprocess] = T
        # faces
        if path_preprocess in self._F_buffer:
            F = self._F_buffer[path_preprocess]
        else:
            F = readFaceBIN(path_preprocess + 'faces.bin')
            if self._use_buffer: self._F_buffer[path_preprocess] = F
        # edges

        if path_preprocess in self._E_buffer:
            E = self._E_buffer[path_preprocess]
        else:
            E = readEdgeBIN(path_preprocess + 'edges.bin')
            if self._use_buffer: self._E_buffer[path_preprocess] = E

        # laplacian
        if path_preprocess in self._L_buffer:
            L = self._L_buffer[path_preprocess]
        else:
            L = sparse.load_npz(path_preprocess + 'laplacian.npz').tocoo()
            if self._use_buffer: self._L_buffer[path_preprocess] = L
        # fabric
        fabric = self._fabric_to_one_hot(info, garments, subindices)
        return T, V, F, E, L

    # encode fabric as per-vertex one-hot
    def _fabric_to_one_hot(self, info, garments, subindices):
        _F = None
        for i in range(1, len(subindices)):
            s, e = subindices[i - 1], subindices[i]
            fabric = np.tile(
                np.float32(self._f2o[info['outfit'][garments[i - 1]]['fabric']]).reshape((1, 4)),
                [e - s, 1]
            )
            if _F is None:
                _F = fabric
            else:
                _F = np.concatenate((_F, fabric), axis=0)
        return _F

    # Get next outfit
    def next_sample(self):
        # read idx and increase
        idx = self._idx
        self._idx += 1
        # reset if all samples read
        if self._idx + 1 > len(self._samples): self._reset()
        # Read sample
        sample, nframe = self._samples[idx]
        nframe = int(nframe)
        # Load info
        info = loadInfo(SRC + sample + '/info.mat')

        # shape&gender
        S = info['shape']
        G = info['gender']
        # weights prior (for unsupervised)
        #W = np.load(SRC_PREPROCESS + sample + '/weights.npy')
        #W = None
        # Load data
        T, V, F, E, L = self._read_outfit(sample, info)
        tightness = np.float32(info['tightness'])


        """ Human """

        
        gender = 'm' if info['gender'] else 'f'
        shape = info['shape']
        P = np.zeros((1, 24, 3), np.float32) + 0.000000001
        P[:, 0, 0] = np.pi / 2
        P[:, 1, 2] = 0.15
        P[:, 2, 2] = -0.15
        B, J = reader.smpl[gender].set_params(pose=P, beta=shape)
        B -= J[0:1]

        return V, T, L, F, E, None, G, S, P, B

    # Get a batch of outfits
    def next(self):
        # return self.next_sample()
        samples = [self.next_sample() for _ in range(self._batch_size)]
        return self._merge_samples(samples)

    # Merges meshes into single one
    def _merge_samples(self, samples):
        V, indices = self._merge_verts(s[0] for s in samples)
        #T, _ = self._merge_verts(s[1] for s in samples)
        L = self._merge_laplacians([s[2] for s in samples], indices)
        F = self._merge_topology([s[3] for s in samples], indices)
        F_S = [s[3] for s in samples]
        E = self._merge_topology([s[4] for s in samples], indices)
        #W, _ = self._merge_verts([s[5] for s in samples])
        G = np.stack([s[6] for s in samples])
        S = np.stack([s[7] for s in samples])
        P = np.stack([s[8] for s in samples])
        B = np.stack([s[9] for s in samples])

        """
        P = np.stack([s[5] for s in samples])
        S = np.stack([s[6] for s in samples])
        G = np.stack([s[7] for s in samples])
        W, _ = self._merge_verts([s[8] for s in samples])
        fabric, _ = self._merge_verts(s[9] for s in samples)
        tightness = np.stack([s[10] for s in samples])
        outfits = [s[11] for s in samples]
        subindices = [s[12] for s in samples]
        """
        return {
            'vertices': tf.convert_to_tensor(V),
            #'template': tf.convert_to_tensor(T),
            'laplacians': L,
            'faces': F,
            'edges': E,
            #'weights_prior': W,
            'indices': indices,
            'genders': G,
            'poses': P,
            'shapes': S,
            'faces_split': F_S,
            'bodies': B
        }

    def _merge_verts(self, Vs):
        V = None
        indices = [0]
        for v in Vs:
            if V is None:
                V = v
            else:
                V = np.concatenate((V, v), axis=0)
            indices += [V.shape[0]]
        return V, indices

    def _merge_laplacians(self, Ls, indices):
        idx, data = None, None
        shape = [indices[-1]] * 2
        for i, l in enumerate(Ls):
            if idx is None:
                idx = np.mat([l.row, l.col]).transpose()
            else:
                _idx = np.mat([l.row, l.col]).transpose()
                _idx += indices[i]
                idx = np.concatenate((idx, _idx))
            if data is None:
                data = l.data
            else:
                data = np.concatenate((data, l.data))
        return tf.SparseTensor(idx.astype(np.float32), data.astype(np.float32), shape)

    def _merge_topology(self, Fs, indices):
        F = None
        for i, f in enumerate(Fs):
            if F is None:
                F = f
            else:
                F = np.concatenate((F, f + indices[i]))
        return F
