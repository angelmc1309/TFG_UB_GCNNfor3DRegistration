import sys
from scipy import sparse
from util import display
from util import display_batch
from values import SRC
from values import SRC_PREPROCESS
from IO import *
from Model.StretchingModel import StretchingModel
from Model.RegisterModel import RegisterModel
import tensorflow as tf 
from Data.data import Data

train = '//content//drive//MyDrive//UB//TFG//TFG_Model//Data//sample_loader.txt'
data = Data(train)

checkpoint = sys.argv[1]

batch = data.next()

#pipeline to read V and L

model = StretchingModel(128, checkpoint)

C = np.array([[255, 0, 0]] * batch['vertices'].shape[0], np.uint8)


display_batch(batch['vertices'].numpy(), batch['faces_split'], batch['bodies'], model.SMPL[0].faces, batch['indices'])

pred = model(batch['vertices'], batch['laplacians'])

display_batch(pred.numpy(), batch['faces_split'], batch['bodies'], model.SMPL[0].faces, batch['indices'])
