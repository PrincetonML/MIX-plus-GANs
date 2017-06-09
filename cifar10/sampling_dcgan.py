import sys
import os
import shutil
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
from lasagne.layers import dnn
from lasagne.init import Normal
sys.path.insert(0, '../')
from cifar10_data import unpickle, load_cifar_data
import time
import nn
import scipy
import scipy.misc
from theano.sandbox.rng_mrg import MRG_RandomStreams

''' settings '''
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/gan0')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-python')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--advloss_weight', type=float, default=1.) # weight for adversarial loss
parser.add_argument('--condloss_weight', type=float, default=1.) # weight for conditional loss
parser.add_argument('--entloss_weight', type=float, default=10.) # weight for entropy loss
parser.add_argument('--labloss_weight', type=float, default=1.) # weight for entropy loss
parser.add_argument('--gen_lr', type=float, default=0.0001) # learning rate for generator
parser.add_argument('--disc_lr', type=float, default=0.0001) # learning rate for discriminator
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=50000)
args = parser.parse_args()
print(args)

rng = np.random.RandomState(args.seed) # fixed random seeds
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

''' input tensor variables '''
y_1hot = T.matrix()
x = T.tensor4()
y = T.ivector()
meanx = T.tensor3()
lr = T.scalar() # learning rate
                   
''' specify generator G0, gen_x = G0(z0, h1) '''
"""
z0 = theano_rng.uniform(size=(args.batch_size, 50)) # uniform noise
gen0_layers = [LL.InputLayer(shape=(args.batch_size, 50), input_var=z0)] # Input layer for z0
gen0_layers.append(nn.batch_norm(LL.DenseLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[0], num_units=128, W=Normal(0.02), nonlinearity=nn.relu)),
                  num_units=128, W=Normal(0.02), nonlinearity=nn.relu))) # embedding, 50 -> 128
gen0_layer_z_embed = gen0_layers[-1] 

gen0_layers.append(LL.InputLayer(shape=(args.batch_size, 10), input_var=y_1hot)) # Input layer
gen0_layer_fc3 = gen0_layers[-1]

gen0_layers.append(LL.ConcatLayer([gen0_layer_fc3,gen0_layer_z_embed], axis=1)) # concatenate noise and fc3 features
gen0_layers.append(LL.ReshapeLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[-1], num_units=256*5*5, W=Normal(0.02), nonlinearity=T.nnet.relu)),
                 (args.batch_size,256,5,5))) # fc
gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,256,10,10), (5,5), stride=(2, 2), padding = 'half',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,128,14,14), (5,5), stride=(1, 1), padding = 'valid',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv

gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,128,28,28), (5,5), stride=(2, 2), padding = 'half',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
gen0_layers.append(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,3,32,32), (5,5), stride=(1, 1), padding = 'valid',
                 W=Normal(0.02),  nonlinearity=T.nnet.sigmoid)) # deconv

gen_x_pre = LL.get_output(gen0_layers[-1], deterministic=False)
gen_x = gen_x_pre - meanx
"""

''' define the sampling function '''
samplefun = th.function(inputs=[meanx, y_1hot], outputs=gen_x) # to generate images conditioned on labels 

weights_toload = np.load('logs/baseline_test_test/gen_x_params_epoch188.npz')
weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
LL.set_all_param_values(gen0_layers, weights_list_toload)

''' load mean img '''
meanimg = np.load('data/meanimg.npy')

samples = []
for i in range(args.num_samples // args.batch_size):
  refy = np.zeros((args.batch_size,), dtype=np.int)
  for i in range(args.batch_size):
      refy[i] =  i%10
      refy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
      refy_1hot[np.arange(args.batch_size), refy] = 1

  imgs = samplefun(meanimg, refy_1hot)
  imgs = imgs + meanimg
  imgs = np.transpose(np.reshape(imgs, (args.batch_size, 3, 32, 32)), (0, 2, 3, 1))
  samples.append(imgs)

samples = np.concatenate(samples, 0) 
np.save('sampled_imgs/baseline_test_test_DCGAN_samples_for_inception.npy', samples)

''' sample images by DCGAN '''
imgs = [imgs[i] for i in range(100)]
rows = []
for i in range(10):
    rows.append(np.concatenate(imgs[i::10], 1))
imgs = np.concatenate(rows, 0)
scipy.misc.imsave("cifar_samples.png", imgs)


