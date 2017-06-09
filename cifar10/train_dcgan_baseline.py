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
import time
import nn
import scipy
import scipy.misc
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.ifelse import ifelse

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/baseline_test_final')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--labloss_weight', type=float, default=1.)
parser.add_argument('--fealoss_weight', type=float, default=1.)
parser.add_argument('--advloss_weight', type=float, default=1.)
parser.add_argument('--zloss_weight', type=float, default=10.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10) #number of labeled examples per class
parser.add_argument('--resume_train', type=bool, default=False)
args = parser.parse_args()
print(args)

# make out_dir if it does not exist, copy current script to out_dir to ensure reproducible experiments
if not args.resume_train:
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print "Created folder {}".format(args.out_dir)
    else:
        print "folder {} already exists. please remove it first.".format(args.out_dir)
        exit(1)
    shutil.copyfile(sys.argv[0], args.out_dir + '/training_script.py')


# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('data/cifar-10-python/cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('data/cifar-10-python/cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    # X_train_flip = X_train[:,:,:,::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train,X_train_flip),axis=0)
    # Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return pixel_mean, dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

## specify generator, gen_pool5 = G(z, y_1hot)
#z = theano_rng.uniform(size=(args.batch_size, 100)) # uniform noise
#y_1hot = T.matrix()
#gen_pool5_layer_z = LL.InputLayer(shape=(args.batch_size, 100), input_var=z) # z, 100
#gen_pool5_layer_z_embed = nn.batch_norm(LL.DenseLayer(gen_pool5_layer_z, num_units=256, W=Normal(0.02), nonlinearity=T.nnet.relu), g=None) # 100 -> 256
#gen_pool5_layer_y = LL.InputLayer(shape=(args.batch_size, 10), input_var=y_1hot) # y, 10
#gen_pool5_layer_y_embed = nn.batch_norm(LL.DenseLayer(gen_pool5_layer_y, num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu), g=None) # 10 -> 512
#gen_pool5_layer_fc4 = LL.ConcatLayer([gen_pool5_layer_z_embed,gen_pool5_layer_y_embed],axis=1) #512+256 = 768
##gen_pool5_layer_fc4 = nn.batch_norm(LL.DenseLayer(gen_pool5_layer_fc5, num_units=512, nonlinearity=T.nnet.relu))#, g=None) 
#gen_pool5_layer_fc3 = nn.batch_norm(LL.DenseLayer(gen_pool5_layer_fc4, num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu), g=None) 
#gen_pool5_layer_pool5_flat = LL.DenseLayer(gen_pool5_layer_fc3, num_units=4*4*32, nonlinearity=T.nnet.relu) # NO batch normalization at output layer
##gen_pool5_layer_pool5_flat = nn.batch_norm(LL.DenseLayer(gen_pool5_layer_fc3, num_units=4*4*32, W=Normal(0.02), nonlinearity=T.nnet.relu), g=None) # no batch-norm at output layer
#gen_pool5_layer_pool5 = LL.ReshapeLayer(gen_pool5_layer_pool5_flat, (args.batch_size,32,4,4))
#gen_pool5_layers = [gen_pool5_layer_z, gen_pool5_layer_z_embed, gen_pool5_layer_y, gen_pool5_layer_y_embed, #gen_pool5_layer_fc5,
# gen_pool5_layer_fc4, gen_pool5_layer_fc3, gen_pool5_layer_pool5_flat, gen_pool5_layer_pool5]
#gen_pool5 = LL.get_output(gen_pool5_layer_pool5, deterministic=False)
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
#from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
#from lasagne.layers import batch_norm
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import DropoutLayer
import nn

enc_layers = [LL.InputLayer(shape=(None, 3, 32, 32))]
enc_layer_conv1 = dnn.Conv2DDNNLayer(enc_layers[-1], 64, (5,5), pad=0, stride=1, W=Normal(0.01), nonlinearity=nn.relu)
enc_layers.append(enc_layer_conv1)
print(enc_layer_conv1.output_shape)
enc_layer_pool1 = LL.MaxPool2DLayer(enc_layers[-1], pool_size=(2, 2))
enc_layers.append(enc_layer_pool1)
enc_layer_conv2 = dnn.Conv2DDNNLayer(enc_layers[-1], 128, (5,5), pad=0, stride=1, W=Normal(0.01), nonlinearity=nn.relu)
enc_layers.append(enc_layer_conv2)
print(enc_layer_conv2.output_shape)
enc_layer_pool2 = LL.MaxPool2DLayer(enc_layers[-1], pool_size=(2, 2))
print(enc_layer_pool2.output_shape)
enc_layers.append(enc_layer_pool2)
# enc_layers.append(LL.ReshapeLayer(enc_layers[-1],(128, 32*4*4)))
enc_layer_fc3 = LL.DenseLayer(enc_layers[-1], num_units=256, nonlinearity=T.nnet.relu)
enc_layers.append(enc_layer_fc3)
enc_layer_fc4 = LL.DenseLayer(enc_layers[-1], num_units=10, nonlinearity=T.nnet.softmax)
enc_layers.append(enc_layer_fc4)
net=enc_layers
 
weights_toload = np.load('pretrained/encoder.npz')
weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
lasagne.layers.set_all_param_values(enc_layers[-1], weights_list_toload)
# check batch norm
for l in LL.get_all_layers(enc_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        assert np.abs(np.mean(l.avg_batch_mean.get_value()) - 0)>1e-7

# input variables
y = T.ivector()
y_1hot = T.matrix()
x = T.tensor4()
meanx = T.tensor3()
# real_fc3 = LL.get_output(enc_layer_fc3, x, deterministic=True)

#y_pred, real_pool3 = LL.get_output([fc8, poo5], x, deterministic=False)
# real_pool3 = LL.get_output(poo5, x, deterministic=False)
#enc_error = T.mean(T.neq(T.argmax(y_pred,axis=1),y)) # classification error of the encoder, to make sure the encoder is working properly


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

gen_x_pre = LL.get_output(gen_x_layer_x, deterministic=False)
gen_x = gen_x_pre - meanx

# # specify discriminator for x, Note: no weight normalization
# disc_x_layers = [LL.InputLayer(shape=(None, 3, 32, 32))]
# disc_x_layers.append(LL.GaussianNoiseLayer(disc_x_layers[-1], sigma=0.05))
# disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 32, (3,3), pad=1, W=Normal(0.01), nonlinearity=nn.lrelu)))
# disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 32, (3,3), pad=1, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)))
# disc_x_layers.append(LL.DropoutLayer(disc_x_layers[-1], p=0.3))
# disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 64, (3,3), pad=1, W=Normal(0.01), nonlinearity=nn.lrelu)))
# disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)))
# disc_x_layers_shared = LL.DropoutLayer(disc_x_layers[-1], p=0.3)
# disc_x_layers.append(disc_x_layers_shared)

# disc_x_layer_z_recon = LL.DenseLayer(disc_x_layers_shared, num_units=50, nonlinearity=None)
# disc_x_layers.append(disc_x_layer_z_recon) # also need to recover z from x

# disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers_shared, 128, (3,3), pad=0, W=Normal(0.01), nonlinearity=nn.lrelu)))
# disc_x_layers.append(LL.GlobalPoolLayer(disc_x_layers[-1],))
# #disc_x_layers.append(nn.MinibatchLayer(disc_x_layers[-1], num_kernels=100))
# disc_x_layer_adv = LL.DenseLayer(disc_x_layers[-1], num_units=10, W=Normal(0.01), nonlinearity=None)
# disc_x_layers.append(disc_x_layer_adv)

# specify discriminative model
disc_x_layers = [LL.InputLayer(shape=(None, 3, 32, 32))]
disc_x_layers.append(LL.GaussianNoiseLayer(disc_x_layers[-1], sigma=0.2))
disc_x_layers.append(dnn.Conv2DDNNLayer(disc_x_layers[-1], 96, (3,3), pad=1, W=Normal(0.01), nonlinearity=nn.lrelu))
disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)))
disc_x_layers.append(LL.DropoutLayer(disc_x_layers[-1], p=0.5))
disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 192, (3,3), pad=1, W=Normal(0.01), nonlinearity=nn.lrelu)))
disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)))
disc_x_layers.append(LL.DropoutLayer(disc_x_layers[-1], p=0.5))
disc_x_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc_x_layers[-1], 192, (3,3), pad=0, W=Normal(0.01), nonlinearity=nn.lrelu)))
disc_x_layers_shared = LL.NINLayer(disc_x_layers[-1], num_units=192, W=Normal(0.01), nonlinearity=nn.lrelu)
disc_x_layers.append(disc_x_layers_shared)

disc_x_layer_z_recon = LL.DenseLayer(disc_x_layers_shared, num_units=50, nonlinearity=None)
disc_x_layers.append(disc_x_layer_z_recon) # also need to recover z from x

# disc_x_layers.append(nn.MinibatchLayer(disc_x_layers_shared, num_kernels=100))
disc_x_layers.append(LL.GlobalPoolLayer(disc_x_layers_shared))
disc_x_layer_adv = LL.DenseLayer(disc_x_layers[-1], num_units=10, W=Normal(0.01), nonlinearity=None)
disc_x_layers.append(disc_x_layer_adv)

#output_before_softmax_x = LL.get_output(disc_x_layer_adv, x, deterministic=False)
#output_before_softmax_gen = LL.get_output(disc_x_layer_adv, gen_x, deterministic=False)

# temp = LL.get_output(gen_x_layers[-1], deterministic=False, init=True)
# temp = LL.get_output(disc_x_layers[-1], x, deterministic=False, init=True)
# init_updates = [u for l in LL.get_all_layers(gen_x_layers)+LL.get_all_layers(disc_x_layers) for u in getattr(l,'init_updates',[])]

output_before_softmax_real = LL.get_output(disc_x_layer_adv, x, deterministic=False) 
output_before_softmax_gen, recon_z = LL.get_output([disc_x_layer_adv, disc_x_layer_z_recon], gen_x, deterministic=False) # discriminator's predicted probability that gen_x is real

l_lab = output_before_softmax_real[T.arange(args.batch_size),y]
l_unl = nn.log_sum_exp(output_before_softmax_real)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_class_x = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_real))) # loss for not correctly classifying the category of real images
loss_real_x = -T.mean(l_unl) + T.mean(T.nnet.softplus(l_unl)) # loss for classifying real as fake
loss_fake_x = T.mean(T.nnet.softplus(l_gen)) # loss for classifying fake as real
loss_disc_x_adv = 0.5*loss_real_x  + 0.5*loss_fake_x
loss_z_recon = T.mean((recon_z - z)**2)
loss_disc_x = args.labloss_weight * loss_class_x + args.advloss_weight * loss_disc_x_adv + args.zloss_weight * loss_z_recon

# loss for generator
y_recon = LL.get_output(enc_layer_fc4, gen_x, deterministic=True) # reconstructed pool3 activations
#loss_gen_x_adv = -loss_fake_x  # adversarial loss
loss_gen_x_adv = -T.mean(T.nnet.softplus(l_gen))
# loss_gen_x_fea = T.mean((recon_fc3 - real_fc3)**2) # feature loss, euclidean distance in feature space
loss_gen_x_fea = T.mean(T.nnet.categorical_crossentropy(y_recon, y_1hot)) # feature loss
loss_gen_x = args.advloss_weight * loss_gen_x_adv + args.fealoss_weight * loss_gen_x_fea + args.zloss_weight * loss_z_recon

# collect parameters
lr = T.scalar()
mincost = T.scalar()
disc_x_params = LL.get_all_params(disc_x_layers, trainable=True)
#disc_x_param_updates = nn.adam_updates(disc_x_params, loss_disc_x, lr=lr, mom1=0.5) # loss for discriminator = supervised_loss + unsupervised loss
disc_x_param_updates = nn.adam_conditional_updates(disc_x_params, loss_disc_x, mincost=mincost+args.zloss_weight*loss_z_recon+args.labloss_weight * loss_class_x, lr=lr, mom1=0.5) # if loss_disc_x < mincost, don't update the discriminator
disc_x_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_x_params] # initialized with 0
disc_x_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_x_params,disc_x_param_avg)] # online update of historical parameters
disc_x_avg_givens = [(p,a) for p,a in zip(disc_x_params,disc_x_param_avg)]
disc_x_bn_updates = [u for l in LL.get_all_layers(disc_x_layers[-1]) for u in getattr(l,'bn_updates',[])]
disc_x_bn_params = []
for l in LL.get_all_layers(disc_x_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        disc_x_bn_params.append(l.avg_batch_mean)
        disc_x_bn_params.append(l.avg_batch_var)

gen_x_params = LL.get_all_params(gen_x_layers[-1], trainable=True)
gen_x_param_updates = nn.adam_updates(gen_x_params, loss_gen_x, lr=lr, mom1=0.5)
gen_x_bn_updates = [u for l in LL.get_all_layers(gen_x_layers[-1]) for u in getattr(l,'bn_updates',[])]
gen_x_bn_params = []
for l in LL.get_all_layers(gen_x_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        gen_x_bn_params.append(l.avg_batch_mean)
        gen_x_bn_params.append(l.avg_batch_var)
w_gen_x_probe = gen_x_params[-2]
dw_gen_x_probe = T.grad(loss_gen_x, w_gen_x_probe)

# init_param = th.function(inputs=[x], outputs=None, updates=init_updates)
train_batch_disc_x = th.function(inputs=[meanx, x, y, y_1hot, lr, mincost], outputs=[loss_class_x, loss_real_x, loss_fake_x, gen_x, x], 
    updates=disc_x_param_updates+disc_x_avg_updates+disc_x_bn_updates) # train discriminator for x with one batch of labeled images, and generated things, return losses, update parameters using adam and update historical parameters online
train_batch_gen_x = th.function(inputs=[meanx, y_1hot, lr], outputs=[loss_z_recon, loss_gen_x_adv, loss_gen_x_fea, dw_gen_x_probe], updates=gen_x_param_updates+gen_x_bn_updates)
#samplefun = th.function(inputs=[],outputs=gen_x) # z -> x
reconfun = th.function(inputs=[meanx, y_1hot],outputs=gen_x) # z -> x
# load CIFAR data
if not os.path.exists("data/cifar-10-python/cifar-10-batches-py"):
    print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")

# Load the dataset
print("Loading data...")
meanimg, data = load_data()
trainx = data['X_train']
trainy = data['Y_train']
testx = data['X_test']
testy = data['Y_test']
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# Load the features
# feature_data = np.load('features/cifar/vgglighter/fc3.npy')
#assert feature_data.shape == (trainx.shape[0], 64, 8, 8)

# //////////// perform training //////////////
if args.resume_train:
    temp = np.load(args.out_dir + '/losses.npy')
    temp = temp.item()
    start_epoch = len(temp['loss_real_x'])/nr_batches_train
    # load trained weights at previous epoch
    weights_disc = np.load(args.out_dir + "/disc_x_params_epoch{}.npz".format(start_epoch-1))
    weights_disc_list = [weights_disc['arr_'.format(k)] for k in range(len(weights_disc.files))]
    LL.set_all_param_values(disc_x_params, weights_disc_list)
        
    weights_gen = np.load(args.out_dir + '/gen_x_params_epoch{}.npz'.format(start_epoch-1))
    weights_gen_list = [weights_gen['arr_'.format(k)] for k in range(len(weights_gen.files))]
    LL.set_all_param_values(gen_x_params, weights_gen_list)

#   LL.set_all_param_values(disc_x_layers, weights_disc_list, trainable = True)
#   LL.set_all_param_values(gen_layers, weights_gen_list, trainable = True)
else: 
    start_epoch = 0
 
gen_x_lr = 0.0001
disc_x_lr = 0.0001
mincost = 0.3 # if discriminator loss < mincost, don't update the discriminator
if args.resume_train:
    losses = np.load(args.out_dir + '/losses.npy')
    losses = losses.item() 
else:
    losses = {'loss_real_x': [], 'loss_fake_x': [], 'loss_gen_x_adv': [], 'loss_gen_x_fea': [], 'loss_z_recon': [], 'loss_class_x': [],
    'var_gen_x_w': [], 'var_gen_x_wgrad': [], 'var_gen_x': [], 'var_real_x': []}

print "training from epoch {}".format(start_epoch)
for epoch in range(start_epoch, 300):
    begin = time.time()
    # if epoch==0:
    #     init_param(trainx[:500])
    # construct randomly permuted minibatches
    inds = rng.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    # feature_data = feature_data[inds]
    #nr_batches_train = 1 # for test purpose
    for t in range(nr_batches_train):
        batchx = trainx[t*args.batch_size:(t+1)*args.batch_size]
        batchy = trainy[t*args.batch_size:(t+1)*args.batch_size]
        # batchfea = feature_data[t*args.batch_size:(t+1)*args.batch_size]
        batchy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
        batchy_1hot[np.arange(args.batch_size), batchy] = 1 # convert to one-hot label

        # train discriminator
        l_class_x, l_real_x, l_fake_x, g_x, r_x = train_batch_disc_x(meanimg, batchx, batchy, batchy_1hot, disc_x_lr, mincost)
        #print encoder_error
        #randomy = np.random.randint(10, size = (args.batch_size,))
        #randomy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
        #randomy_1hot[np.arange(args.batch_size), randomy] = 1

        # train generator, adjustable training speed
        l_adv = (l_fake_x+l_real_x)/2.
        if l_adv > 0.65: # discriminator is random guessing
            n_iter = 1
        elif l_adv > 0.5: # discriminator works reasonably well (60%)
            n_iter = 3
        elif l_adv > 0.3: # discriminator very strong (74%)
            n_iter = 5
        else:
            n_iter = 7
        #n_iter = 3

        for i in range(n_iter):
            #pass
            l_z_recon, l_gen_x_adv, l_gen_x_fea, dw_gen_x = train_batch_gen_x(meanimg, batchy_1hot, gen_x_lr)

        w_gen_x = w_gen_x_probe.get_value()
        var_gen_x_w = np.var(np.array(w_gen_x)) # variance of some generator weights
        var_gen_x_w_grad = np.var(np.array(dw_gen_x)) # variance of some generator weight gradients
        var_gen_x = np.var(np.array(g_x)) # variance of the generated representations
        var_real_x = np.var(np.array(r_x)) # variance of the "real" representations

        
        # store loss information
        losses['loss_real_x'].append(l_real_x)
        losses['loss_fake_x'].append(l_fake_x)
        losses['loss_class_x'].append(l_class_x)
        losses['loss_gen_x_adv'].append(l_gen_x_adv)
        losses['loss_gen_x_fea'].append(l_gen_x_fea)
        losses['loss_z_recon'].append(l_z_recon)
        losses['var_gen_x_w'].append(var_gen_x_w)
        losses['var_gen_x_wgrad'].append(var_gen_x_w_grad)
        losses['var_gen_x'].append(var_gen_x)
        losses['var_real_x'].append(var_real_x)
        print("Epoch %d, time = %ds, var gen x weights= %.4f, var gen x weights gradient = %.8f, var gen x = %.4f, var real x = %.4f" % 
         (epoch, time.time()-begin, var_gen_x_w, var_gen_x_w_grad, var_gen_x, var_real_x))
        print("loss_real_x = %.4f, loss_fake_x = %.4f, loss_class_x = %.4f, loss_gen_x_adv = %.4f,  loss_gen_x_fea = %.4f, loss_z_recon = %.4f" % (l_real_x, l_fake_x, l_class_x, l_gen_x_adv, l_gen_x_fea, l_z_recon))

    ## sample
    #imgs = samplefun()
    #imgs = np.reshape(imgs[:100,], (100, 28, 28))
    #imgs = [imgs[i, :, :] for i in range(100)]
    #rows = []
    #for i in range(10):
    #    rows.append(np.concatenate(imgs[i::10], 1))
    #imgs = np.concatenate(rows, 0)
    #scipy.misc.imsave(args.out_dir + "/mnist_sample_lgan_epoch{}.png".format(epoch), imgs)
    
    # recon
    reconx = reconfun(meanimg, batchy_1hot)
    reconx = reconx[:100,]
    print reconx.shape
    reconx = np.transpose(reconx, (0,2,3,1)) # (100, 32, 3, 3)
    reconx = [reconx[i, :, :, :] for i in range(100)]
    rows = []
    for i in range(10):
        rows.append(np.concatenate(reconx[i::10], 1))
    reconx = np.concatenate(rows, 0)
    scipy.misc.imsave(args.out_dir + "/cifar_recon_lgan_epoch{}.png".format(epoch), reconx)

    orix = batchx[:100,]
    orix = np.transpose(orix, (0,2,3,1)) # (100, 32, 3, 3)
    orix = [orix[i, :, :] for i in range(100)]
    rows = []
    for i in range(10):
        rows.append(np.concatenate(orix[i::10], 1))
    orix = np.concatenate(rows, 0)
    scipy.misc.imsave(args.out_dir + "/cifar_ori_lgan_epoch{}.png".format(epoch), orix)

    # save params
    if epoch%args.save_interval==0:
        np.savez(args.out_dir + "/disc_x_params_epoch{}.npz".format(epoch), *LL.get_all_param_values(disc_x_layers))
        np.savez(args.out_dir + '/gen_x_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(gen_x_layers))
        np.savez(args.out_dir + '/gen_x_bn_params_epoch{}.npz'.format(epoch), *[param.get_value() for param in gen_x_bn_params])
        np.save(args.out_dir + '/losses.npy',losses)

