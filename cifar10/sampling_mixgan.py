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
from train_mixgan import MixSGAN


if __name__ == '__main__':
  ''' settings '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_dir', type=str, default='logs/mixgan_shared_z_final')
  parser.add_argument('--data_dir', type=str, default='data/cifar-10-python')
  parser.add_argument('--save_interval', type = int, default = 1)
  parser.add_argument('--num_epoch', type = int, default = 200)
  parser.add_argument('--start_epoch', type = int, default = 0)
  parser.add_argument('--z0dim', type = int, default = 50)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--seed_data', type=int, default=1)
  parser.add_argument('--advloss_weight', type=float, default=1.) # weight for adversarial loss
  parser.add_argument('--condloss_weight', type=float, default=1.) # weight for conditional loss
  parser.add_argument('--entloss_weight', type=float, default=10.) # weight for entropy loss
  parser.add_argument('--mix_entloss_weight', type=float, default=0.1) # weight for entropy loss
  parser.add_argument('--mincost', type=float, default=0.3) # weight for entropy loss
  parser.add_argument('--ng', type=int, default=5) # number of generators 
  parser.add_argument('--nd', type=int, default=5) # number of discriminators 
  parser.add_argument('--labloss_weight', type=float, default=1.) # weight for entropy loss
  parser.add_argument('--gen_lr', type=float, default=0.0001) # learning rate for generator
  parser.add_argument('--disc_lr', type=float, default=0.0001) # learning rate for discriminator
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--load_epoch', type=int, default=16)
  args = parser.parse_args()
  print(args)

  for i in range(152,153,1):
    gan = MixSGAN(args)
    gan.sampling(50000, i)
    print(i)

  """
  gan = MixSGAN(args)
  gan.sampling(50000, 105)
  """

  """
  ''' sample images by DCGAN '''
  imgs = [imgs[i] for i in range(100)]
  rows = []
  for i in range(10):
      rows.append(np.concatenate(imgs[i::10], 1))
  imgs = np.concatenate(rows, 0)
  scipy.misc.imsave("cifar_samples.png", imgs)
  """


