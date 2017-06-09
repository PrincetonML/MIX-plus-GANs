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


for epoch in range(0,106,20):
  ''' input tensor variables '''
  y_1hot = T.matrix()
  x = T.tensor4()
  y = T.ivector()
  meanx = T.tensor3()
  lr = T.scalar() # learning rate
                     
  # specify generator, gen_x = G(z, real_pool3)
  z = theano_rng.uniform(size=(args.batch_size, 50)) # uniform noise
  # y_1hot = T.matrix()
  gen_x_layer_z = LL.InputLayer(shape=(args.batch_size, 50), input_var=z) # z, 20
  # gen_x_layer_z_embed = nn.batch_norm(LL.DenseLayer(gen_x_layer_z, num_units=128), g=None) # 20 -> 64

  gen_x_layer_y = LL.InputLayer(shape=(args.batch_size, 10), input_var=y_1hot) # conditioned on real fc3 activations
  gen_x_layer_y_z = LL.ConcatLayer([gen_x_layer_y,gen_x_layer_z],axis=1) #512+256 = 768
  gen_x_layer_pool2 = LL.ReshapeLayer(nn.batch_norm(LL.DenseLayer(gen_x_layer_y_z, num_units=256*5*5)), (args.batch_size,256,5,5))
  gen_x_layer_dconv2_1 = nn.batch_norm(nn.Deconv2DLayer(gen_x_layer_pool2, (args.batch_size,256,10,10), (5,5), stride=(2, 2), padding = 'half',
                   W=Normal(0.02),  nonlinearity=nn.relu))
  gen_x_layer_dconv2_2 = nn.batch_norm(nn.Deconv2DLayer(gen_x_layer_dconv2_1, (args.batch_size,128,14,14), (5,5), stride=(1, 1), padding = 'valid',
                   W=Normal(0.02),  nonlinearity=nn.relu))

  gen_x_layer_dconv1_1 = nn.batch_norm(nn.Deconv2DLayer(gen_x_layer_dconv2_2, (args.batch_size,128,28,28), (5,5), stride=(2, 2), padding = 'half',
                   W=Normal(0.02),  nonlinearity=nn.relu))
  gen_x_layer_x = nn.Deconv2DLayer(gen_x_layer_dconv1_1, (args.batch_size,3,32,32), (5,5), stride=(1, 1), padding = 'valid',
                   W=Normal(0.02),  nonlinearity=T.nnet.sigmoid)
  # gen_x_layer_x = dnn.Conv2DDNNLayer(gen_x_layer_dconv1_2, 3, (1,1), pad=0, stride=1, 
  #                  W=Normal(0.02), nonlinearity=T.nnet.sigmoid)

  print(gen_x_layer_x.output_shape)

  gen_x_layers = [gen_x_layer_z, gen_x_layer_y, gen_x_layer_y_z, gen_x_layer_pool2, gen_x_layer_dconv2_1, 
      gen_x_layer_dconv2_2, gen_x_layer_dconv1_1, gen_x_layer_x]

  gen_x_pre = LL.get_output(gen_x_layer_x, deterministic=True)
  gen_x = gen_x_pre - meanx

  ''' define the sampling function '''
  samplefun = th.function(inputs=[meanx, y_1hot], outputs=gen_x) # to generate images conditioned on labels 

  weights_toload = np.load('logs/baseline_test_final/gen_x_params_epoch%d.npz'%epoch)
  weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
  LL.set_all_param_values(gen_x_layers, weights_list_toload)

  ''' load mean img '''
  meanimg = np.load('data/meanimg.npy')

  refy = np.zeros((args.batch_size,), dtype=np.int)
  for i in range(args.batch_size):
    refy[i] =  1
    refy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
  refy_1hot[np.arange(args.batch_size), refy] = 1

  samples = []
  for i in range(args.num_samples // args.batch_size):
    imgs = samplefun(meanimg, refy_1hot)
    imgs = imgs + meanimg
    imgs = np.transpose(np.reshape(imgs, (args.batch_size, 3, 32, 32)), (0, 2, 3, 1))
    samples.append(imgs)

  samples = np.concatenate(samples, 0) 
  np.save('sampled_imgs/baseline_DCGAN_samples_for_inception_%d.npy' % epoch, samples)
  scipy.misc.imsave("cifar_samples_dcgan_%d.png" % epoch, imgs[15])
  """
  ''' sample images by DCGAN '''
  imgs = [imgs[i] for i in range(100)]
  rows = []
  for i in range(10):
      rows.append(np.concatenate(imgs[i::10], 1))
  imgs = np.concatenate(rows, 0)
  scipy.misc.imsave("cifar_samples.png", imgs)
  """


