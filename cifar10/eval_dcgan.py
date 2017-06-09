import sys
import os
import shutil
import argparse
import numpy as np
import time
import inception_score as inception

for epoch in range(0,106,5):
	#imgs = np.load('sampled_imgs/MixGAN1_single0_samples_for_inception_epoch%d.npy' % epoch) 
	#imgs = np.load('sampled_imgs/single1_single0_samples_for_inception_epoch199.npy') 
	imgs = np.load('sampled_imgs/baseline_DCGAN_samples_for_inception_%d.npy' % epoch) 
	imgs = (imgs + 1.) * 127.5
	score, std = inception.get_inception_score(list(imgs), splits=10)
	print("------------")
	print(score)
	print(std)

