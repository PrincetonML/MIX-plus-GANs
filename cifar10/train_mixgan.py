import sys
import os
import shutil
import argparse
import numpy as np
import theano as th
import theano.tensor as T
from theano import pp
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



class MixSGAN:
    def __init__(self, args):

        self.args = args

        rng = np.random.RandomState(self.args.seed) # fixed random seeds
        theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
        lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
        data_rng = np.random.RandomState(self.args.seed_data)

        ''' specify pre-trained generator E '''
        self.enc_layers = [LL.InputLayer(shape=(None, 3, 32, 32), input_var=None)]
        enc_layer_conv1 = dnn.Conv2DDNNLayer(self.enc_layers[-1], 64, (5,5), pad=0, stride=1, W=Normal(0.01), nonlinearity=nn.relu)
        self.enc_layers.append(enc_layer_conv1)
        enc_layer_pool1 = LL.MaxPool2DLayer(self.enc_layers[-1], pool_size=(2, 2))
        self.enc_layers.append(enc_layer_pool1)
        enc_layer_conv2 = dnn.Conv2DDNNLayer(self.enc_layers[-1], 128, (5,5), pad=0, stride=1, W=Normal(0.01), nonlinearity=nn.relu)
        self.enc_layers.append(enc_layer_conv2)
        enc_layer_pool2 = LL.MaxPool2DLayer(self.enc_layers[-1], pool_size=(2, 2))
        self.enc_layers.append(enc_layer_pool2)
        self.enc_layer_fc3 = LL.DenseLayer(self.enc_layers[-1], num_units=256, nonlinearity=T.nnet.relu)
        self.enc_layers.append(self.enc_layer_fc3)
        self.enc_layer_fc4 = LL.DenseLayer(self.enc_layers[-1], num_units=10, nonlinearity=T.nnet.softmax)
        self.enc_layers.append(self.enc_layer_fc4)


        ''' load pretrained weights for encoder '''
        weights_toload = np.load('pretrained/encoder.npz')
        weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
        LL.set_all_param_values(self.enc_layers[-1], weights_list_toload)


        ''' input tensor variables '''
        #self.G_weights
        #self.D_weights
        self.dummy_input = T.scalar()
        self.G_layers = []
        self.z = theano_rng.uniform(size=(self.args.batch_size, self.args.z0dim))
        self.x = T.tensor4()
        self.meanx = T.tensor3()
        self.Gen_x = T.tensor4() 
        self.D_layers = []
        self.D_layer_adv = [] 
        self.D_layer_z_recon = []
        self.gen_lr = T.scalar() # learning rate
        self.disc_lr = T.scalar() # learning rate
        self.y = T.ivector()
        self.y_1hot = T.matrix()
        self.Gen_x_list = []
        self.y_recon_list = []
        self.mincost = T.scalar()
        #self.enc_layer_fc3 = self.get_enc_layer_fc3()

        self.real_fc3 = LL.get_output(self.enc_layer_fc3, self.x, deterministic=True)

    def get_enc_layer_fc3(self):
        return self.enc_layer_fc3

    def get_enc_layer_fc4(self):
        return self.enc_layer_fc4

    def get_generator(self, meanx, z0, y_1hot):
        ''' specify generator G0, gen_x = G0(z0, h1) '''
        """
        #z0 = theano_rng.uniform(size=(self.args.batch_size, 16)) # uniform noise
        gen0_layers = [LL.InputLayer(shape=(self.args.batch_size, 50), input_var=z0)] # Input layer for z0
        gen0_layers.append(nn.batch_norm(LL.DenseLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[0], num_units=128, W=Normal(0.02), nonlinearity=nn.relu)),
                          num_units=128, W=Normal(0.02), nonlinearity=nn.relu))) # embedding, 50 -> 128
        gen0_layer_z_embed = gen0_layers[-1] 

        #gen0_layers.append(LL.InputLayer(shape=(self.args.batch_size, 256), input_var=real_fc3)) # Input layer for real_fc3 in independent training, gen_fc3 in joint training
        gen0_layers.append(LL.InputLayer(shape=(self.args.batch_size, 10), input_var=y_1hot)) # Input layer for real_fc3 in independent training, gen_fc3 in joint training
        gen0_layer_fc3 = gen0_layers[-1]

        gen0_layers.append(LL.ConcatLayer([gen0_layer_fc3,gen0_layer_z_embed], axis=1)) # concatenate noise and fc3 features
        gen0_layers.append(LL.ReshapeLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[-1], num_units=256*5*5, W=Normal(0.02), nonlinearity=T.nnet.relu)),
                         (self.args.batch_size,256,5,5))) # fc
        gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (self.args.batch_size,256,10,10), (5,5), stride=(2, 2), padding = 'half',
                         W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
        gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (self.args.batch_size,128,14,14), (5,5), stride=(1, 1), padding = 'valid',
                         W=Normal(0.02),  nonlinearity=nn.relu))) # deconv

        gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (self.args.batch_size,128,28,28), (5,5), stride=(2, 2), padding = 'half',
                         W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
        gen0_layers.append(nn.Deconv2DLayer(gen0_layers[-1], (self.args.batch_size,3,32,32), (5,5), stride=(1, 1), padding = 'valid',
                         W=Normal(0.02),  nonlinearity=T.nnet.sigmoid)) # deconv

        gen_x_pre = LL.get_output(gen0_layers[-1], deterministic=False)
        gen_x = gen_x_pre - meanx
        # gen_x_joint = LL.get_output(gen0_layers[-1], {gen0_layer_fc3: gen_fc3}, deterministic=False) - meanx

        return gen0_layers, gen_x 
        """
        gen_x_layer_z = LL.InputLayer(shape=(self.args.batch_size, self.args.z0dim), input_var=z0) # z, 20
        # gen_x_layer_z_embed = nn.batch_norm(LL.DenseLayer(gen_x_layer_z, num_units=128), g=None) # 20 -> 64

        gen_x_layer_y = LL.InputLayer(shape=(self.args.batch_size, 10), input_var=y_1hot) # conditioned on real fc3 activations
        gen_x_layer_y_z = LL.ConcatLayer([gen_x_layer_y,gen_x_layer_z],axis=1) #512+256 = 768
        gen_x_layer_pool2 = LL.ReshapeLayer(nn.batch_norm(LL.DenseLayer(gen_x_layer_y_z, num_units=256*5*5)), (self.args.batch_size,256,5,5))
        gen_x_layer_dconv2_1 = nn.batch_norm(nn.Deconv2DLayer(gen_x_layer_pool2, (self.args.batch_size,256,10,10), (5,5), stride=(2, 2), padding = 'half',
                         W=Normal(0.02),  nonlinearity=nn.relu))
        gen_x_layer_dconv2_2 = nn.batch_norm(nn.Deconv2DLayer(gen_x_layer_dconv2_1, (self.args.batch_size,128,14,14), (5,5), stride=(1, 1), padding = 'valid',
                         W=Normal(0.02),  nonlinearity=nn.relu))

        gen_x_layer_dconv1_1 = nn.batch_norm(nn.Deconv2DLayer(gen_x_layer_dconv2_2, (self.args.batch_size,128,28,28), (5,5), stride=(2, 2), padding = 'half',
                         W=Normal(0.02),  nonlinearity=nn.relu))
        gen_x_layer_x = nn.Deconv2DLayer(gen_x_layer_dconv1_1, (self.args.batch_size,3,32,32), (5,5), stride=(1, 1), padding = 'valid',
                         W=Normal(0.02),  nonlinearity=T.nnet.sigmoid)
        # gen_x_layer_x = dnn.Conv2DDNNLayer(gen_x_layer_dconv1_2, 3, (1,1), pad=0, stride=1, 
        #                  W=Normal(0.02), nonlinearity=T.nnet.sigmoid)

        gen_x_layers = [gen_x_layer_z, gen_x_layer_y, gen_x_layer_y_z, gen_x_layer_pool2, gen_x_layer_dconv2_1, 
            gen_x_layer_dconv2_2, gen_x_layer_dconv1_1, gen_x_layer_x]

        gen_x_pre = LL.get_output(gen_x_layer_x, deterministic=False)
        gen_x = gen_x_pre - meanx

        return gen_x_layers, gen_x 


    def get_discriminator(self):
        ''' specify discriminator D0 '''
        """
        disc0_layers = [LL.InputLayer(shape=(self.args.batch_size, 3, 32, 32))]
        disc0_layers.append(LL.GaussianNoiseLayer(disc0_layers[-1], sigma=0.05))
        disc0_layers.append(dnn.Conv2DDNNLayer(disc0_layers[-1], 96, (3,3), pad=1, W=Normal(0.02), nonlinearity=nn.lrelu))
        disc0_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc0_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.02), nonlinearity=nn.lrelu))) # 16x16
        disc0_layers.append(LL.DropoutLayer(disc0_layers[-1], p=0.1))
        disc0_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc0_layers[-1], 192, (3,3), pad=1, W=Normal(0.02), nonlinearity=nn.lrelu)))
        disc0_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc0_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.02), nonlinearity=nn.lrelu))) # 8x8
        disc0_layers.append(LL.DropoutLayer(disc0_layers[-1], p=0.1))
        disc0_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(disc0_layers[-1], 192, (3,3), pad=0, W=Normal(0.02), nonlinearity=nn.lrelu))) # 6x6
        disc0_layer_shared = LL.NINLayer(disc0_layers[-1], num_units=192, W=Normal(0.02), nonlinearity=nn.lrelu) # 6x6
        disc0_layers.append(disc0_layer_shared)

        disc0_layer_z_recon = LL.DenseLayer(disc0_layer_shared, num_units=50, W=Normal(0.02), nonlinearity=None)
        disc0_layers.append(disc0_layer_z_recon) # also need to recover z from x

        disc0_layers.append(LL.GlobalPoolLayer(disc0_layer_shared))
        disc0_layer_adv = LL.DenseLayer(disc0_layers[-1], num_units=10, W=Normal(0.02), nonlinearity=None)
        disc0_layers.append(disc0_layer_adv)

        return disc0_layers, disc0_layer_adv, disc0_layer_z_recon
        """
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

        disc_x_layer_z_recon = LL.DenseLayer(disc_x_layers_shared, num_units=self.args.z0dim, nonlinearity=None)
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
        return disc_x_layers, disc_x_layer_adv, disc_x_layer_z_recon


    def train(self):

        self.G_weights_layer = nn.softmax_weights(self.args.ng, LL.InputLayer(shape=(), input_var=self.dummy_input))
        self.D_weights_layer = nn.softmax_weights(self.args.ng, LL.InputLayer(shape=(), input_var=self.dummy_input))

        self.G_weights = LL.get_output(self.G_weights_layer, None, deterministic=True)
        self.D_weights = LL.get_output(self.D_weights_layer, None, deterministic=True)

        self.Disc_weights_entropy = T.sum((-1./self.args.nd) * T.log(self.D_weights + 0.000001), [0,1])
        self.Gen_weights_entropy = T.sum((-1./self.args.ng) * T.log(self.G_weights + 0.000001), [0,1]) 

        for i in range(self.args.ng):
            gen_layers_i, gen_x_i = self.get_generator(self.meanx, self.z, self.y_1hot)
            self.G_layers.append(gen_layers_i)
            self.Gen_x_list.append(gen_x_i)
        self.Gen_x = T.concatenate(self.Gen_x_list, axis=0)

        for i in range(self.args.nd):
            disc_layers_i, disc_layer_adv_i, disc_layer_z_recon_i = self.get_discriminator()
            self.D_layers.append(disc_layers_i)
            self.D_layer_adv.append(disc_layer_adv_i)
            self.D_layer_z_recon.append(disc_layer_z_recon_i)
            #T.set_subtensor(self.Gen_x[i*self.args.batch_size:(i+1)*self.args.batch_size], gen_x_i)

            #self.samplers.append(self.sampler(self.z[i], self.y))
        ''' forward pass '''
        loss_gen0_cond_list = []
        loss_disc0_class_list = []
        loss_disc0_adv_list = []
        loss_gen0_ent_list = []
        loss_gen0_adv_list = []
        #loss_disc_list
        
        for i in range(self.args.ng):
            self.y_recon_list.append(LL.get_output(self.enc_layer_fc4, self.Gen_x_list[i], deterministic=True)) # reconstructed pool3 activations

        for i in range(self.args.ng):
            #loss_gen0_cond = T.mean((recon_fc3_list[i] - self.real_fc3)**2) # feature loss, euclidean distance in feature space
            loss_gen0_cond = T.mean(T.nnet.categorical_crossentropy(self.y_recon_list[i], self.y))
            loss_disc0_class = 0
            loss_disc0_adv = 0
            loss_gen0_ent = 0
            loss_gen0_adv = 0
            for j in range(self.args.nd):
                output_before_softmax_real0 = LL.get_output(self.D_layer_adv[j], self.x, deterministic=False) 
                output_before_softmax_gen0, recon_z0 = LL.get_output([self.D_layer_adv[j], self.D_layer_z_recon[j]], self.Gen_x_list[i], deterministic=False) # discriminator's predicted probability that gen_x is real
                ''' loss for discriminator and Q '''
                l_lab0 = output_before_softmax_real0[T.arange(self.args.batch_size),self.y]
                l_unl0 = nn.log_sum_exp(output_before_softmax_real0)
                l_gen0 = nn.log_sum_exp(output_before_softmax_gen0)
                loss_disc0_class += T.dot(self.D_weights[0,j], -T.mean(l_lab0) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_real0)))) # loss for not correctly classifying the category of real images
                loss_real0 = -T.mean(l_unl0) + T.mean(T.nnet.softplus(l_unl0)) # loss for classifying real as fake
                loss_fake0 = T.mean(T.nnet.softplus(l_gen0)) # loss for classifying fake as real
                loss_disc0_adv += T.dot(self.D_weights[0,j], 0.5*loss_real0 + 0.5*loss_fake0)
                loss_gen0_ent += T.dot(self.D_weights[0,j], T.mean((recon_z0 - self.z)**2))
                #loss_gen0_ent = T.mean((recon_z0 - self.z)**2)
                ''' loss for generator '''
                loss_gen0_adv += T.dot(self.D_weights[0,j], -T.mean(T.nnet.softplus(l_gen0)))

            loss_gen0_cond_list.append(T.dot(self.G_weights[0,i], loss_gen0_cond))
            loss_disc0_class_list.append(T.dot(self.G_weights[0,i], loss_disc0_class))
            loss_disc0_adv_list.append(T.dot(self.G_weights[0,i], loss_disc0_adv))
            loss_gen0_ent_list.append(T.dot(self.G_weights[0,i], loss_gen0_ent))
            loss_gen0_adv_list.append(T.dot(self.G_weights[0,i], loss_gen0_adv))

        self.loss_gen0_cond = sum(loss_gen0_cond_list)
        self.loss_disc0_class = sum(loss_disc0_class_list)
        self.loss_disc0_adv = sum(loss_disc0_adv_list)
        self.loss_gen0_ent = sum(loss_gen0_ent_list)
        self.loss_gen0_adv = sum(loss_gen0_adv_list)

        self.loss_disc = self.args.labloss_weight * self.loss_disc0_class + self.args.advloss_weight * self.loss_disc0_adv + self.args.entloss_weight * self.loss_gen0_ent + self.args.mix_entloss_weight * self.Disc_weights_entropy
        self.loss_gen = self.args.advloss_weight * self.loss_gen0_adv + self.args.condloss_weight * self.loss_gen0_cond + self.args.entloss_weight * self.loss_gen0_ent + self.args.mix_entloss_weight * self.Gen_weights_entropy

        if self.args.load_epoch is not None:
            print("loading model")
            self.load_model(self.args.load_epoch)
            print("success")

        ''' collect parameter updates for discriminators '''
        Disc_params = LL.get_all_params(self.D_weights_layer, trainable=True)
        Disc_bn_updates = []
        Disc_bn_params = []

        self.threshold = self.mincost + self.args.labloss_weight * self.loss_disc0_class + self.args.entloss_weight * self.loss_gen0_ent + self.args.mix_entloss_weight * self.Disc_weights_entropy
        #threshold = mincost + self.args.labloss_weight * self.loss_disc0_class + self.args.entloss_weight * self.loss_gen0_ent

        for i in range(self.args.nd):
            Disc_params.extend(LL.get_all_params(self.D_layers[i], trainable=True))
            Disc_bn_updates.extend([u for l in LL.get_all_layers(self.D_layers[i][-1]) for u in getattr(l,'bn_updates',[])])
            for l in LL.get_all_layers(self.D_layers[i][-1]):
                if hasattr(l, 'avg_batch_mean'):
                    Disc_bn_params.append(l.avg_batch_mean)
                    Disc_bn_params.append(l.avg_batch_var)
        Disc_param_updates = nn.adam_conditional_updates(Disc_params, self.loss_disc, mincost=self.threshold, lr=self.disc_lr, mom1=0.5) # if loss_disc_x < mincost, don't update the discriminator
        Disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in Disc_params] # initialized with 0
        Disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(Disc_params, Disc_param_avg)] # online update of historical parameters

        """
        #Disc_param_updates = nn.adam_updates(Disc_params, self.loss_disc, lr=self.lr, mom1=0.5) 
        # collect parameters
        #Disc_params = LL.get_all_params(self.D_layers[-1], trainable=True)
        Disc_params = LL.get_all_params(self.D_layers, trainable=True)
        #Disc_param_updates = nn.adam_updates(Disc_params, loss_disc_x, lr=lr, mom1=0.5) # loss for discriminator = supervised_loss + unsupervised loss
        Disc_param_updates = nn.adam_conditional_updates(Disc_params, self.loss_disc, mincost=threshold, lr=self.disc_lr, mom1=0.5) # if loss_disc_x < mincost, don't update the discriminator
        Disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in Disc_params] # initialized with 0
        Disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(Disc_params,Disc_param_avg)] # online update of historical parameters
        #Disc_avg_givens = [(p,a) for p,a in zip(Disc_params,Disc_param_avg)]
        Disc_bn_updates = [u for l in LL.get_all_layers(self.D_layers[-1]) for u in getattr(l,'bn_updates',[])]
        Disc_bn_params = []
        for l in LL.get_all_layers(self.D_layers[-1]):
            if hasattr(l, 'avg_batch_mean'):
                Disc_bn_params.append(l.avg_batch_mean)
                Disc_bn_params.append(l.avg_batch_var)
        """


        ''' collect parameter updates for generators '''
        Gen_params = LL.get_all_params(self.G_weights_layer, trainable=True)
        Gen_params_updates = []
        Gen_bn_updates = []
        Gen_bn_params = []

        for i in range(self.args.ng):
            Gen_params.extend(LL.get_all_params(self.G_layers[i][-1], trainable=True))
            Gen_bn_updates.extend([u for l in LL.get_all_layers(self.G_layers[i][-1]) for u in getattr(l,'bn_updates',[])])
            for l in LL.get_all_layers(self.G_layers[i][-1]):
                if hasattr(l, 'avg_batch_mean'):
                    Gen_bn_params.append(l.avg_batch_mean)
                    Gen_bn_params.append(l.avg_batch_var)
        Gen_param_updates = nn.adam_updates(Gen_params, self.loss_gen, lr=self.gen_lr, mom1=0.5) 
        """
        #print(Gen_params)
        #train_batch_gen = th.function(inputs=[self.x, self.meanx, self.z, self.y_1hot, self.lr], outputs=[self.loss_gen], on_unused_input='warn')
        #theano.printing.debugprint(train_batch_gen) 
        Gen_param_updates = nn.adam_updates(Gen_params, self.loss_gen, lr=self.lr, mom1=0.5) 
        Gen_params = LL.get_all_params(self.G_layers[-1], trainable=True)
        Gen_param_updates = nn.adam_updates(Gen_params, self.loss_gen, lr=self.gen_lr, mom1=0.5)
        Gen_bn_updates = [u for l in LL.get_all_layers(self.G_layers[-1]) for u in getattr(l,'bn_updates',[])]
        Gen_bn_params = []
        for l in LL.get_all_layers(self.G_layers[-1]):
            if hasattr(l, 'avg_batch_mean'):
                Gen_bn_params.append(l.avg_batch_mean)
                Gen_bn_params.append(l.avg_batch_var)
         """

        ''' define training and testing functions '''
        #train_batch_disc = th.function(inputs=[x, meanx, y, lr], outputs=[loss_disc0_class, loss_disc0_adv, gen_x, x], 
        #    updates=disc0_param_updates+disc0_bn_updates) 
        #th.printing.debugprint(self.loss_disc)  
        train_batch_disc = th.function(inputs=[self.dummy_input, self.meanx, self.x, self.y, self.y_1hot, self.mincost, self.disc_lr], outputs=[self.loss_disc0_class, self.loss_disc0_adv], updates=Disc_param_updates+Disc_bn_updates+Disc_avg_updates) 
        #th.printing.pydotprint(train_batch_disc, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)  
        #train_batch_gen = th.function(inputs=[x, meanx, y_1hot, lr], outputs=[loss_gen0_adv, loss_gen0_cond, loss_gen0_ent], 
        #    updates=gen0_param_updates+gen0_bn_updates)
        #train_batch_gen = th.function(inputs=gen_inputs, outputs=gen_outputs, updates=gen0_param_updates+gen0_bn_updates)
        #train_batch_gen = th.function(inputs=[self.dummy_input, self.x, self.meanx, self.z, self.y_1hot, self.lr], outputs=[self.loss_gen0_adv, self.loss_gen0_cond, self.loss_gen0_ent], updates=Gen_param_updates+Gen_bn_updates)
        train_batch_gen = th.function(inputs=[self.dummy_input, self.meanx, self.y, self.y_1hot, self.gen_lr], outputs=[self.loss_gen0_adv, self.loss_gen0_cond, self.loss_gen0_ent], updates=Gen_param_updates+Gen_bn_updates)
        

        # samplefun = th.function(inputs=[meanx, y_1hot], outputs=gen_x_joint)   # sample function: generating images by stacking all generators
        reconfun = th.function(inputs=[self.meanx, self.y_1hot], outputs=self.Gen_x)       # reconstruction function: use the bottom generator 
                                                                # to generate images conditioned on real fc3 features
        mix_weights = th.function(inputs=[self.dummy_input], outputs=[self.D_weights, self.Disc_weights_entropy, self.G_weights, self.Gen_weights_entropy])

        ''' load data '''
        print("Loading data...")
        meanimg, data = load_cifar_data(self.args.data_dir)
        trainx = data['X_train']
        trainy = data['Y_train']
        nr_batches_train = int(trainx.shape[0]/self.args.batch_size)
        # testx = data['X_test']
        # testy = data['Y_test']
        # nr_batches_test = int(testx.shape[0]/self.args.batch_size)

        ''' perform training  ''' 
        #logs = {'loss_gen0_adv': [], 'loss_gen0_cond': [], 'loss_gen0_ent': [], 'loss_disc0_class': [], 'var_gen0': [], 'var_real0': []} # training logs
        logs = {'loss_gen0_adv': [], 'loss_gen0_cond': [], 'loss_gen0_ent': [], 'loss_disc0_class': []} # training logs
        for epoch in range(self.args.load_epoch+1, self.args.num_epoch):
            begin = time.time()

            ''' shuffling '''
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]

            for t in range(nr_batches_train):
            #for t in range(1):
                ''' construct minibatch '''
                #batchz = np.random.uniform(size=(self.args.batch_size, self.args.z0dim)).astype(np.float32)
                batchx = trainx[t*self.args.batch_size:(t+1)*self.args.batch_size]
                batchy = trainy[t*self.args.batch_size:(t+1)*self.args.batch_size]
                batchy_1hot = np.zeros((self.args.batch_size, 10), dtype=np.float32)
                batchy_1hot[np.arange(self.args.batch_size), batchy] = 1 # convert to one-hot label
                # randomy = np.random.randint(10, size = (self.args.batch_size,))
                # randomy_1hot = np.zeros((self.args.batch_size, 10),dtype=np.float32)
                # randomy_1hot[np.arange(self.args.batch_size), randomy] = 1

                ''' train discriminators '''
                l_disc0_class, l_disc0_adv = train_batch_disc(0.0, meanimg, batchx, batchy, batchy_1hot, self.args.mincost, self.args.disc_lr)

                ''' train generators '''
                #prob_gen0 = np.exp()
                if l_disc0_adv > 0.65:
                    n_iter = 1
                elif l_disc0_adv > 0.5:
                    n_iter = 3
                elif l_disc0_adv > 0.3:
                    n_iter = 5
                else:
                    n_iter = 7
                for i in range(n_iter):
                    #l_gen0_adv, l_gen0_cond, l_gen0_ent = train_batch_gen(0.0, batchx, meanimg, batchz, batchy_1hot, self.args.gen_lr)
                    l_gen0_adv, l_gen0_cond, l_gen0_ent = train_batch_gen(0.0, meanimg, batchy, batchy_1hot, self.args.gen_lr)

                d_mix_weights, d_entloss, g_mix_weights, g_entloss = mix_weights(0.0)


                ''' store log information '''
                # logs['loss_gen1_adv'].append(l_gen1_adv)
                # logs['loss_gen1_cond'].append(l_gen1_cond)
                # logs['loss_gen1_ent'].append(l_gen1_ent)
                # logs['loss_disc1_class'].append(l_disc1_class)
                # logs['var_gen1'].append(np.var(np.array(g1)))
                # logs['var_real1'].append(np.var(np.array(r1)))

                logs['loss_gen0_adv'].append(l_gen0_adv)
                logs['loss_gen0_cond'].append(l_gen0_cond)
                logs['loss_gen0_ent'].append(l_gen0_ent)
                logs['loss_disc0_class'].append(l_disc0_class)
                #logs['var_gen0'].append(np.var(np.array(g0)))
                #logs['var_real0'].append(np.var(np.array(r0)))
                
                print("---Epoch %d, time = %ds" % (epoch, time.time()-begin))
                print("D_weights=[%.6f, %.6f, %.6f, %.6f, %.6f] loss = %0.6f" % (d_mix_weights[0,0], d_mix_weights[0,1], d_mix_weights[0,2], d_mix_weights[0,3], d_mix_weights[0,4], d_entloss))
                print("G_weights=[%.6f, %.6f, %.6f, %.6f, %.6f] loss = %0.6f" % (g_mix_weights[0,0], g_mix_weights[0,1], g_mix_weights[0,2], g_mix_weights[0,3], g_mix_weights[0,4], g_entloss))
                #print("G_weights=[%.6f]" % (g_mix_weights[0,0]))
                print("loss_disc0_adv = %.4f, loss_gen0_adv = %.4f,  loss_gen0_cond = %.4f, loss_gen0_ent = %.4f, loss_disc0_class = %.4f" % (l_disc0_adv, l_gen0_adv, l_gen0_cond, l_gen0_ent, l_disc0_class))
            # ''' sample images by stacking all generators'''
            # imgs = samplefun(meanimg, refy_1hot)
            # imgs = np.transpose(np.reshape(imgs[:100,], (100, 3, 32, 32)), (0, 2, 3, 1))
            # imgs = [imgs[i] for i in range(100)]
            # rows = []
            # for i in range(10):
            #     rows.append(np.concatenate(imgs[i::10], 1))
            # imgs = np.concatenate(rows, 0)
            # scipy.misc.imsave(self.args.out_dir + "/mnist_sample_epoch{}.png".format(epoch), imgs)

            """
            ''' original images in the training set'''
            orix = np.transpose(np.reshape(batchx[:100,], (100, 3, 32, 32)), (0, 2, 3, 1))
            orix = [orix[i] for i in range(100)]
            rows = []
            for i in range(10):
                rows.append(np.concatenate(orix[i::10], 1))
            orix = np.concatenate(rows, 0)
            scipy.misc.imsave(self.args.out_dir + "/mnist_ori_epoch{}.png".format(epoch), orix)
            """

            if epoch%self.args.save_interval==0:
                # np.savez(self.args.out_dir + "/disc1_params_epoch{}.npz".format(epoch), *LL.get_all_param_values(disc1_layers[-1]))
                # np.savez(self.args.out_dir + '/gen1_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(gen1_layers[-1]))
                #np.savez(self.args.out_dir + "/disc0_params_epoch{}.npz".format(epoch), *LL.get_all_param_values(disc0_layers))
                #np.savez(self.args.out_dir + '/gen0_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(gen0_layers))
                np.savez(self.args.out_dir + '/Dweights_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(self.D_weights_layer))
                np.savez(self.args.out_dir + '/Gweights_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(self.G_weights_layer))
                for i in range(self.args.ng):
                    np.savez(self.args.out_dir + ("/disc%d_params_epoch%d.npz" % (i,epoch)), *LL.get_all_param_values(self.D_layers[i]))
                    np.savez(self.args.out_dir + ("/gen%d_params_epoch%d.npz" % (i,epoch)), *LL.get_all_param_values(self.G_layers[i]))
                np.save(self.args.out_dir + '/logs.npy',logs)

            ''' reconstruct images '''
            reconx = reconfun(meanimg, batchy_1hot) + meanimg
            width = np.round(np.sqrt(self.args.batch_size)).astype(int)
            for i in range(self.args.ng):
                reconx_i = np.transpose(np.reshape(reconx[i*self.args.batch_size:(i+1)*self.args.batch_size], (self.args.batch_size, 3, 32, 32)), (0, 2, 3, 1))
                reconx_i = [reconx_i[j] for j in range(self.args.batch_size)]
                rows = []
                for j in range(width):
                    rows.append(np.concatenate(reconx_i[j::width], 1))
                reconx_i = np.concatenate(rows, 0)
                scipy.misc.imsave(self.args.out_dir + ("/cifar_recon_%d_epoch%d.png"%(i,epoch)), reconx_i) 

            


    def load_model(self, epoch):
        weights_toload = np.load(self.args.out_dir + '/Dweights_params_epoch{}.npz'.format(epoch))
        weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
        LL.set_all_param_values(self.D_weights_layer, weights_list_toload)

        weights_toload = np.load(self.args.out_dir + '/Gweights_params_epoch{}.npz'.format(epoch))
        weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
        LL.set_all_param_values(self.G_weights_layer, weights_list_toload)

        for i in range(self.args.ng):
            weights_toload = np.load(self.args.out_dir + '/disc%d_params_epoch%d.npz' % (i,epoch))
            weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
            LL.set_all_param_values(self.D_layers[i], weights_list_toload)
            weights_toload = np.load(self.args.out_dir + '/gen%d_params_epoch%d.npz' % (i,epoch))
            weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
            LL.set_all_param_values(self.G_layers[i], weights_list_toload)

    def sampling(self, num_samples, epoch):

        sample_batch_size = 10
        self.args.batch_size = sample_batch_size

        rng = np.random.RandomState(self.args.seed) # fixed random seeds
        theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
        lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
        data_rng = np.random.RandomState(self.args.seed_data)
        self.G_weights_layer = nn.softmax_weights(self.args.ng, LL.InputLayer(shape=(), input_var=self.dummy_input))
        self.D_weights_layer = nn.softmax_weights(self.args.ng, LL.InputLayer(shape=(), input_var=self.dummy_input))

        self.G_weights = LL.get_output(self.G_weights_layer, None, deterministic=True)
        self.D_weights = LL.get_output(self.D_weights_layer, None, deterministic=True)

        Gen_x_list = []
        z = theano_rng.uniform(size=(sample_batch_size, 50))
        y_1hot = T.fmatrix()
        for i in range(self.args.ng):
            gen_layers_i, gen_x_i = self.get_generator(self.meanx, z, y_1hot)
            self.G_layers.append(gen_layers_i)
            Gen_x_list.append(gen_x_i)

        for i in range(self.args.nd):
            disc_layers_i, disc_layer_adv_i, disc_layer_z_recon_i = self.get_discriminator()
            self.D_layers.append(disc_layers_i)
            self.D_layer_adv.append(disc_layer_adv_i)
            self.D_layer_z_recon.append(disc_layer_z_recon_i)

        self.load_model(epoch)

        samplefun_list = []
        for i in range(self.args.ng):
            samplefun_list.append(th.function(inputs=[self.meanx, y_1hot], outputs=Gen_x_list[i]))

        mix_weights = th.function(inputs=[self.dummy_input], outputs=[self.G_weights])
        g_mix_weights = mix_weights(0.0)

        

        ''' load mean img '''
        meanimg = np.load('data/meanimg.npy')

        samples = []
        prob_list = g_mix_weights[0].tolist()[0]
        prob_list[-1] = 1 - sum(prob_list[:-1])

        refy = np.zeros((sample_batch_size,), dtype=np.int)
        for i in range(sample_batch_size):
            refy[i] =  i%10

        refy_1hot = np.zeros((sample_batch_size, 10),dtype=np.float32)
        refy_1hot[np.arange(sample_batch_size), refy] = 1
        for k in range(num_samples // sample_batch_size):
            Gen_indx = np.random.choice(5, 1, p=prob_list)
            #Gen_indx = [4]
            #z = np.uniform(size=(sample_batch_size, 16), dtype=np.float32)
            imgs = samplefun_list[Gen_indx[0]](meanimg, refy_1hot)
            imgs = imgs + meanimg
            imgs = np.transpose(np.reshape(imgs, (sample_batch_size, 3, 32, 32)), (0, 2, 3, 1))
            samples.append(imgs)

        samples = np.concatenate(samples, 0) 
        np.save('sampled_imgs/MixGAN_samples_for_inception_epoch%d.npy' % epoch, samples)
        scipy.misc.imsave("cifar_samples_mixgan_%d.png" % epoch, samples[20])

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
    parser.add_argument('--batch_size', type=int, default=49)
    parser.add_argument('--load_epoch', type=int, default=20)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir) # make out_dir if it does not exist, copy current script to out_dir
        print "Created folder {}".format(args.out_dir)
        shutil.copyfile(sys.argv[0], args.out_dir + '/training_script.py')
    elif args.load_epoch is None:
        print "folder {} already exists. please remove it first.".format(args.out_dir)
        exit(1)

    rng = np.random.RandomState(args.seed) # fixed random seeds
    theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
    lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
    data_rng = np.random.RandomState(args.seed_data)

    gan = MixSGAN(args)
    gan.train()

