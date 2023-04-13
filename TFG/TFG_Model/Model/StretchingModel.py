import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smpl.smpl_np import SMPLModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from Layers import *
from values import rest_pose

class StretchingModel:
	# Builds models and initializes SMPL
	def __init__(self, psd_dim, checkpoint=None, rest_pose=rest_pose):
		# psd_dim: dimensionality of blend shape matrices (PSD)
		# checkpoint: path to pre-trained model
		# with_body: will compute SMPL body vertices
		# rest_pose: SMPL rest pose for the dataset (star pose in CLOTH3D; A-pose in TailorNet; ...)
		self._psd_dim = psd_dim
		self._build()
		self._best = float('inf') # best metric obtained with this model
		
		smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/smpl/'
		self.SMPL = {
			0: SMPLModel(smpl_path + 'model_f.pkl', rest_pose),
			1: SMPLModel(smpl_path + 'model_m.pkl', rest_pose)
		}
		# load pre-trained
		if checkpoint is not None:
			print("Loading pre-trained model: " + checkpoint)
			self.load(checkpoint)

		"""
		# USED COEFFS
		self.vertices_reg_coff = 3.0e-1
		self.edges_reg_coff = 2.0e+0
		self.collision_reg_coff = 7.5e+2
		self.normal_coff = 1.0e+3
		self.distance_coff = 1.0e-5
		"""
		self.vertices_reg_coff = 3.0e-1
		self.edges_reg_coff = 2.0e+0
		self.collision_reg_coff = 7.5e+2
		self.normal_coff = 1.0e+3
		self.distance_coff = 1.0e-5



	# Builds model
	def _build(self):
		# Phi

		self._GCCN = [
			GraphConvolution((3, 32), act=tf.nn.relu, name='gcn_0'),
			GraphConvolution((32, 64), act=tf.nn.relu, name='gcn_1'),
			GraphConvolution((64, 128), act=tf.nn.relu, name='gcn_2'),
			GraphConvolution((128, 256), act=tf.nn.relu, name='gcn_3'),
			GraphConvolution((256, 256), act=tf.nn.relu, name='gcn_4')
		]
		self._FC = [
			FullyConnected((256, 3), act=None, name='fc_0')
		]


		
	# Returns list of model variables
	def gather(self):
		vars = []
		for lay in self._GCCN:
			vars += lay.gather()
		for lay in self._FC:
			vars += lay.gather()
		return vars
	
	# loads pre-trained model (checks differences between model and checkpoint)
	def load(self, checkpoint):
		# checkpoint: path to pre-trained model
		# list vars
		vars = self.gather()
		# load vars values
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		values = np.load(checkpoint, allow_pickle=True)[()]
		# assign
		_vars = set([v.name for v in vars])
		_vars_chck = set(values.keys()) - {'best'}
		_diff = sorted(list(_vars_chck - _vars))
		if len(_diff):
			print("Model missing vars:")
			for v in _diff: print("\t" + v)
		_diff = sorted(list(_vars - _vars_chck))
		if len(_diff):
			print("Checkpoint missing vars:")
			for v in _diff: print("\t" + v)
		for v in vars: 
			try: v.assign(values[v.name])
			except:
				if v.name not in values: continue
				else: 
					print("Mismatch in variable shape:")
					print("\t" + v.name)
		if 'best' in values: self._best = values['best']
		
	def save(self, checkpoint):
		# checkpoint: path to save the pre-trained model
		print("\tSaving checkpoint: " + checkpoint)
		# get vars values
		values = {v.name: v.numpy() for v in self.gather()}
		if self._best is not float('inf'): values['best'] = self._best
		# save weights
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		np.save(checkpoint, values)
	


	
	def _ones(self, T_shape):
		return tf.ones((T_shape[0], 1), tf.float32)

	def _convolve(self, X, L):
		# X: template outfit verts
		# L: template outfit laplacian
		for lay in self._GCCN: X = lay(X, L)
		for lay in self._FC: X = lay(X)
		return X

	def _weights(self, X, L):
		# X: template outfit descriptors
		for l in self._GCCN: X = l(X,L)
		# normalize weights to sum 1
		X = X / (tf.reduce_sum(X, axis=-1, keepdims=True) + 1e-7)
		return X

	def __call__(self, V, L):
		# V: Vertices
		# L: Laplacians
		#self.W = self._weights(V, L)

		return self._convolve(V, L)
