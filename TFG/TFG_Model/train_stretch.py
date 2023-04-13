import os
from time import time
from datetime import timedelta
from math import floor
from Data.data import Data
from Model.DeePSD import DeePSD
from Model.StretchingModel import StretchingModel
from util import model_summary
from util import display
from util import display_batch
from Losses import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" ARGS """
# gpu_id: GPU slot to run model
# name: name under which model checkpoints will be saved
# checkpoint: pre-trained model (must be in ./checkpoints/ folder)

gpu_id = sys.argv[1]  # mandatory
name = sys.argv[2]  # mandatory
checkpoint = None
if len(sys.argv) > 3:
    checkpoint = sys.argv[3]

""" GPU """
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 10
if name == 'test': stdout_steps = 1  # show progress continuously (instead of every 100 steps)

""" TRAIN PARAMS """
batch_size = 1
virtual_batch_size = 1
num_epochs = 101  # if fine-tuning, fewer epochs

""" MODEL """
print("Building model...")
model = StretchingModel(128, checkpoint)
if checkpoint is not None and checkpoint.split('/')[1] != name: model._best = float('inf')
tgts = model.gather()  # model weights
model_summary(tgts)
print("*" * 25)
print("Model Best: ", model._best)
print("*" * 25)
optimizer = tf.optimizers.Adam()

""" DATA """
print("Reading data...")
tr_txt = '/content/drive/MyDrive/UB/TFG/TFG_Model/Data/sample_loader.txt'
#tr_txt = '/content/drive/MyDrive/UB/TFG/TFG_Model/Data/train_full.txt'
val_txt = '/content/drive/MyDrive/UB/TFG/TFG_Model/Data/val.txt'
tr_data = Data(tr_txt, batch_size=batch_size)
val_data = Data(val_txt, batch_size=batch_size)

print("TRAIN DATA: ", tr_data._samples)
tr_steps = floor(len(tr_data._samples) / batch_size)
val_steps = floor(len(val_data._samples) / batch_size)
next_save = 0
f = open("stretch_losses.txt","w")
for epoch in range(num_epochs):
    if (epoch + 1) % 2 == 0: virtual_batch_size *= 2
    print("")
    print("Epoch " + str(epoch + 1))
    print("--------------------------")
    """ TRAIN """
    print("Training...")
    total_time = 0
    error = 0
    cgrds = None
    start = time()
    for step in range(tr_steps):
        """ I/O """
        batch = tr_data.next()
        C = np.array([[255, 0, 0]] * batch['vertices'].shape[0], np.uint8)
        """ Train step """
        with tf.GradientTape() as tape:
            pred = model(
                batch['vertices'],
                batch['laplacians'],
            )

            # Loss & Error
            if (next_save) % 20 == 0 and next_save != 0:
                display_batch(pred.numpy(), batch['faces_split'], batch['bodies'], model.SMPL[0].faces, batch['indices'])
                #display(pred, batch['faces'], C)
                next_save = 1
                model.save('/content/drive/MyDrive/UB/TFG/TFG_Model/checkpoints/' + name + "_" + str(epoch)+ "_" +str(step))
            else:
              next_save += 1
              
            normal_loss_ = normal_loss(pred, batch['faces'], batch['laplacians'])


            distance_loss_ = tf.convert_to_tensor(0., dtype="float32")

            # for each body
            for i in range(1,len(batch['indices'])):
                #generate body with smpl here
                distance_loss_ += distance_loss(pred[batch['indices'][i-1]:batch['indices'][i]], batch['shapes'][i-1], batch['genders'][i-1], batch['poses'][i-1])
            vertices_reg_ = vertices_regularizer(pred, batch['vertices'])
            collision_loss_, _, _ = collision_loss(pred,
                                                   batch['bodies'],
                                                   model.SMPL[0].faces,
                                                   batch['indices'])
            edge_loss_, _ = edge_loss(pred, batch['vertices'], batch['edges'])


            
            print("---NORMAL LOSS---")
            print(normal_loss_ * model.normal_coff)
            print("-----------------")

            print("---VERT LOSS---")
            print(vertices_reg_* model.vertices_reg_coff)
            print("-----------------")

            print("---COLL LOSS---")
            print(collision_loss_ * model.collision_reg_coff)
            print("-----------------")

            print("---EDGE LOSS---")
            print(edge_loss_ * model.edges_reg_coff)
            print("-----------------")

            print("---DIST LOSS---")
            print(distance_loss_ * (model.distance_coff / batch_size))
            print("-----------------")

            f.write(str(normal_loss_))
            f.write(str(vertices_reg_))
            f.write(str(collision_loss_))
            f.write(str(edge_loss_))
            f.write(str(distance_loss_))
            f.write('\n')



            loss = model.collision_reg_coff * collision_loss_ + \
                  model.vertices_reg_coff * vertices_reg_ + \
                  model.edges_reg_coff * edge_loss_ + \
                  model.normal_coff * normal_loss_  + \
                  (model.distance_coff / batch_size) * distance_loss_
            
            print("---TOTAL LOSS---")
            print(loss)
            print("-----------------")
            
            """
            if epoch == 0 and not checkpoint:
                ww = 10 ** (1 - step / tr_steps)
                loss += ww * tf.reduce_sum((model.W - batch['weights_prior']) ** 2)
            """
        """ Backprop """
        grads = tape.gradient(loss, tgts)

        print("----------------------")
        print("GRADS")
        print("----------------------")
        # print(grads)
        optimizer.apply_gradients(zip(grads, tgts))
        """
        if virtual_batch_size is not None:
            if cgrds is None: cgrds = grads
            else: cgrds = [c + g for c,g in zip(cgrds,grads)]
            if (step + 1) % virtual_batch_size == 0:
                optimizer.apply_gradients(zip(cgrds, tgts))
                cgrds = None
        else:
            optimizer.apply_gradients(zip(grads, tgts))
        """

        """ Progress """
        # error += E_L2.numpy()
        total_time = time() - start
        ETA = (tr_steps - step - 1) * (total_time / (1 + step))
        if (step + 1) % stdout_steps == 0:
            sys.stdout.write('\r\tStep: ' + str(step + 1) + '/' + str(tr_steps) + ' ... '
                             + 'Err: {:.2f}'.format(1000 * error / (1 + step))
                             + ' ... ETA: ' + str(timedelta(seconds=ETA)))
            sys.stdout.flush()
    """ Epoch results """
    error /= (step + 1)
    print("")
    print("Total error: {:.5f}".format(1000 * error))  # in millimeters
    print("Total time: " + str(timedelta(seconds=total_time)))
    print("")

    if (False):
        """ VALIDATION """
        print("Validating...")
        total_time = 0
        error = 0
        start = time()
        for step in range(val_steps):
            """ I/O """
            batch = val_data.next()
            """ Forward pass """
            pred, body = model(
                batch['template'],
                batch['laplacians'],
                batch['poses'],
                batch['shapes'],
                batch['genders'],
                batch['fabric'],
                batch['tightness'],
                batch['indices'],
                with_body=False
            )
            """ Metrics """
            _, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
            """ Progress """
            error += E_L2.numpy()
            total_time = time() - start
            ETA = (val_steps - step - 1) * (total_time / (1 + step))
            if (step + 1) % stdout_steps == 0:
                sys.stdout.write('\r\tStep: ' + str(step + 1) + '/' + str(val_steps) + ' ... '
                                 + 'Err: {:.2f}'.format(1000 * error / (1 + step))
                                 + ' ... ETA: ' + str(timedelta(seconds=ETA)))
                sys.stdout.flush()
        """ Epoch results """
        error /= (step + 1)
        print("")
        print("Total error: {:.5f}".format(1000 * error))
        print("Total time: " + str(timedelta(seconds=total_time)))
        print("")
        """ Save checkpoint """
        if error < model._best:
            model._best = error
            model.save('checkpoints/' + name)
        print("")
        print("BEST: ", model._best)
        print("")

f.close()
