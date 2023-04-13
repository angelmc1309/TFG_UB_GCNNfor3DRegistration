import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
from scipy.spatial import cKDTree
from DataReader.read import DataReader

reader = DataReader()

def L2_loss(x, y, indices, subindices=None):
	# x: predicted outfits
	# y: ground truth
	# indices: to split batch into single outfits
	# subindices: to split outfits into garments, if provided, computes error garment-wise
	D = tf.sqrt(tf.reduce_sum((x - y) ** 2, -1))
	loss = tf.reduce_sum(D)
	err = []
	for i in range(1, len(indices)):
		s, e = indices[i - 1], indices[i]
		if subindices is None:
			err += [tf.reduce_mean(D[s:e])]
		else:
			_D = D[s:e]
			for j in range(1, len(subindices[i - 1])):
				_s, _e = subindices[i - 1][j - 1], subindices[i - 1][j]
				err += [tf.reduce_mean(_D[_s:_e])]
	err = tf.reduce_mean(err)
	return loss, err
	
def edge_loss(x, y, E):
	# x: predicted outfits
	# y: template outfits
	# E: Nx2 array of edges
	x_e = tf.gather(x, E[:,0], axis=0) - tf.gather(x, E[:,1], axis=0)
	x_e = tf.reduce_sum(x_e ** 2, -1)
	x_e = tf.sqrt(x_e)
	y_e = tf.gather(y, E[:,0], axis=0) - tf.gather(y, E[:,1], axis=0)
	y_e = tf.reduce_sum(y_e ** 2, -1)
	y_e = tf.sqrt(y_e)
	d_e = y_e - x_e
	err = tf.reduce_mean(tf.abs(d_e))
	loss = tf.reduce_sum(d_e ** 2)
	return loss, err

def bend_loss(x, F, L):
	# x: predicted outfits
	# F: faces
	# L: laplacian
	VN = tfg.geometry.representation.mesh.normals.vertex_normals(x, F)
	bend = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
		VN, L, sizes=None,
		edge_function=lambda x,y: (x - y)**2,
		reduction='weighted',
		edge_function_kwargs={}
	)
	bend_dist = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
		VN, L, sizes=None,
		edge_function=lambda x,y: 1 - tf.einsum('ab,ab->a', x, y)[:,None],
		reduction='weighted',
		edge_function_kwargs={}
	)
	bend_dist = tf.clip_by_value(bend_dist, 0, 2)
	return tf.reduce_sum(bend), tf.reduce_mean(bend_dist)

def collision_loss(x, B, B_F, indices, thr=.005):
	# x: predicted outfits
	# B: posed human bodies
	# B_F: human body faces
	# indices: to split batch into single outfits
	# thr: collision threshold
	loss = 0
	vmask = np.zeros(x.shape[0], np.float32)
	vcount = []
	for i in range(1, len(indices)):
		s, e = indices[i - 1], indices[i]
		_x = x[s:e]
		_B = np.float32(B[i - 1])
		# build body KDTree
		tree = cKDTree(_B)
		_, idx = tree.query(_x.numpy(), n_jobs=-1)
		# to nearest
		D = _x - _B[idx]
		# body normals
		B_vn = tfg.geometry.representation.mesh.normals.vertex_normals(_B, B_F)
		# corresponding normals
		VN = tf.gather(B_vn, idx, axis=0)
		# dot product
		dot = tf.einsum('ab,ab->a', D, VN)
		vmask[s:e] = tf.cast(tf.math.less(dot, thr), tf.float32)
		_vmask = tf.cast(tf.math.less(dot, 0), tf.float32)
		vcount += [tf.reduce_sum(_vmask) / _x.shape[0]]
		# collision if dot < 0  --> -dot > 0
		loss += tf.reduce_sum(tf.minimum(dot - thr, 0) ** 2)
	return loss, np.array(vcount).mean(), vmask


def vertices_regularizer(predicted, garment):

	return tf.reduce_sum(tf.square(predicted - garment))


def normal_loss(V, F, L):

  """
  N = tfg.geometry.representation.mesh.normals.vertex_normals(V, F)

  laplacians = L.indices.numpy() 
  edge_index = laplacians[0][0]

  normal_values = np.zeros((N.shape[0],4),dtype=int)
  for edge in laplacians:
      edges_list = normal_values[edge[0]]
      index = np.argmin(edges_list)
      edges_list[index] = edge[1]
  
  loss = tf.convert_to_tensor(0., dtype="float32")

  get_normal_x = lambda x : N[x,0].numpy()
  get_normal_y = lambda x : N[x,1].numpy()
  get_normal_z = lambda x : N[x,2].numpy()

  normals_x = np.vectorize(get_normal_x)(normal_values)

  loss += tf.reduce_mean(tf.math.reduce_std(tf.convert_to_tensor(normals_x), 1))

  normals_y = np.vectorize(get_normal_y)(normal_values)

  loss += tf.reduce_mean(tf.math.reduce_std(tf.convert_to_tensor(normals_y), 1))

  normals_z = np.vectorize(get_normal_z)(normal_values)

  loss += tf.reduce_mean(tf.math.reduce_std(tf.convert_to_tensor(normals_z), 1))


  return loss

  # SECOND IMPLEMENTATION
  loss = tf.convert_to_tensor(0., dtype="float32")
  

  laplacians = L.indices.numpy() 
  edge_index = laplacians[0][0]

  normal_values = []
  for edge in laplacians:
    if edge_index == edge[0]:
      normal_values.append(N[edge[1]].numpy().tolist())
    else:
      edge_index = edge[0]

      #print("normal_values", normal_values)
      std = np.std(np.array(normal_values), axis=0)
      loss += np.sum(np.square(std)) 
      normal_values = []
      normal_values.append(N[edge[1]].numpy().tolist())
      #print("N", N[edge[1]])
      #print("N numpy", N[edge[1]].numpy())



  print(V.shape)
  return loss / V.shape[0]
  """

  N = tfg.geometry.representation.mesh.normals.vertex_normals(V, F)

  #DOT IMPLEMENTATION
  D = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
  N, L, sizes=None,
  edge_function= lambda x,y: tf.einsum("ij,ij->ij", x, y),
  reduction='weighted',
  edge_function_kwargs={}
  )

  return tf.reduce_mean(1 - tf.reduce_sum(D, axis=1))
  
def distance_loss(pred, shape, gender, pose):
  
  G = 'm' if gender else 'f'
  body, J = reader.smpl[G].set_params(pose=pose, beta=shape)
  body -= J[0:1]
  garment = tf.reshape(pred, (pred.shape[0], 1 , pred.shape[1]))
  body_reshape = tf.reshape(tf.convert_to_tensor(body, dtype="float32"), (1, body.shape[0], body.shape[1]))

  d = tf.reduce_sum(tf.square(1 - tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(garment - body_reshape), -1)), -1)))
  del garment
  del body_reshape
  return d




def chamfer_distance(B,G):
  """
  return tfg.nn.loss.chamfer_distance.evaluate(B,G)
  """
  point_set_a = tf.convert_to_tensor(value=B, dtype='float32')
  point_set_b = tf.convert_to_tensor(value=G, dtype='float32')

  # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
  # dimension D).
  difference = (
      tf.expand_dims(point_set_a, axis=-2) - tf.expand_dims(point_set_b, axis=-3))
  # Calculate the square distances between each two points: |ai - bj|^2.
  square_distances = tf.einsum("...i,...i->...", difference, difference)

  minimum_square_distance_a_to_b = tf.reduce_min(
      input_tensor=square_distances, axis=-1)
  minimum_square_distance_b_to_a = tf.reduce_min(
      input_tensor=square_distances, axis=-2)

  return (
      tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
      tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))[0]

def register_vectors_loss(stretched, predicted, L):
  register = stretched - predicted

  #DOT IMPLEMENTATION
  D = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
  register, L, sizes=None,
  edge_function= lambda x,y: tf.einsum("ij,ij->ij", x, y),
  reduction='weighted',
  edge_function_kwargs={}
  )

  return tf.reduce_mean(10 - tf.reduce_sum(D, axis=1))

def edge_similarity_loss(x, E):
	# x: predicted outfit
	# E: Nx2 array of edges

  # Compute edge distances
	x_e = tf.gather(x, E[:,0], axis=0) - tf.gather(x, E[:,1], axis=0)
	x_e = tf.reduce_sum(x_e ** 2, -1)
	x_e = tf.sqrt(x_e)

	return tf.math.reduce_std(x_e)


