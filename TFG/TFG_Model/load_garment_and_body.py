import sys
import tensorflow as tf
import values
from DataReader.read import DataReader

reader = DataReader()

""" Utils for this notebook """
import os
from random import choice
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from Losses import chamfer_distance

# Display utils used on this notebook require triangulated faces
def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)

# Display mesh
def display(V, F, C):
    if F.shape[1] != 3: F = quads2tris(F)
    fig = go.Figure(data=[
        go.Mesh3d(
            x=V[:,0],
            y=V[:,1],
            z=V[:,2],
            # i, j and k give the vertices of triangles
            i = F[:,0],
            j = F[:,1],
            k = F[:,2],
            vertexcolor = C,
            showscale=True
        )
    ])
    fig.show()

samples = os.listdir('/content/drive/MyDrive/UB/TFG/TFG_Model/Samples/')

sample = '03543'#choice(samples) # sample/sequence name
print(sample)
print('----------------')

info = reader.read_info(sample)

# Display dict utility
from pprint import PrettyPrinter
printer = PrettyPrinter(indent=4)
#printer.pprint(info)

frame = 0 # frame to visualize
from IO import readOBJ

""" Human """
gender = 'm' if info['gender'] else 'f'
shape = info['shape']
pose = np.zeros((1,24,3), np.float32) + 0.000000001
pose[:,0,0] = np.pi / 2
pose[:,1,2] = 0.15
pose[:,2,2] = -0.15
V, J = reader.smpl[gender].set_params(pose=pose, beta=shape)
V -= J[0:1]
F = np.array(reader.smpl[gender].faces)
C = np.array([[255,255,255]]*V.shape[0], np.uint8)

""" Garments """
garments = list(info['outfit'].keys())
for i,garment in enumerate(garments):
    _V, _F, _, _ = readOBJ(os.path.join(reader.SRC, sample, garment + '.obj'))
    _F = quads2tris(_F)
    # Vertex colors
    _C = np.array([[255*i,0,255*(i-1)]]*_V.shape[0], np.uint8)

    V = V
    loss = chamfer_distance(V, _V)

    print("Chamfer loss", loss)
    # Merge human and garment meshes into one (for visualization purposes)
    F = np.concatenate((F,_F + V.shape[0]),0)
    V = np.concatenate((V,_V),0)
    C = np.concatenate((C,_C),0)
    del _V
    del _F

""" DISPLAY """
display(V, F, C)