import sys
import os
import shutil
import argparse
import numpy as np
import time
import inception_score as inception
for epoch in range(152,153,1):
	#imgs = np.load('sampled_imgs/MixGAN1_single0_samples_for_inception_epoch%d.npy' % epoch) 
	#imgs = np.load('sampled_imgs/single1_single0_samples_for_inception_epoch199.npy') 
	imgs = np.load('sampled_imgs/MixGAN_samples_for_inception_epoch%d.npy' % epoch) 
	imgs = imgs * 255
	score, std = inception.get_inception_score(list(imgs), splits=10)
	print("------------")
	print(score)
	print(std)
