#encoding:utf-8

import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
#mpl.use('Agg')# AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F

import xml.etree.ElementTree as ET
#import cv2
from itertools import product
from math import sqrt

from generator import *

class Discriminator(nn.Module):
	def __init__(self,z_dim=20,image_size=64):
		super(Discriminator,self).__init__()

		self.layer1 = nn.Sequential(
				nn.Conv2d(3,image_size,kernel_size=4,stride=2,padding=1),
				nn.LeakyReLU(0.1,inplace=True))
		self.layer2 = nn.Sequential(
				nn.Conv2d(image_size,image_size*2,kernel_size=4,stride=2,padding=1),
				nn.LeakyReLU(0.1,inplace=True))
		self.layer3 = nn.Sequential(
				nn.Conv2d(image_size*2,image_size*4,kernel_size=4,stride=2,padding=1),
				nn.LeakyReLU(0.1,inplace=True))
		self.layer4 = nn.Sequential(
				nn.Conv2d(image_size*4,image_size*8,kernel_size=4,stride=2,padding=1),
				nn.LeakyReLU(0.1,inplace=True))
		self.last = nn.Sequential(
				nn.Conv2d(image_size*8,1,kernel_size=4,stride=1))

	def forward(self,x):
		out = self.layer1(x);
		out = self.layer2(out);
		out = self.layer3(out);
		out = self.layer4(out);
		out = self.last(out);
		return out

#動作確認
G = Generator(z_dim=20,image_size=64)
D = Discriminator(z_dim=20,image_size=64)

input_z = torch.randn(1,20)
input_z = input_z.view(input_z.size(0),input_z.size(1),1,1)
fakeimg = G(input_z)

d_out = D(fakeimg)

print(nn.Sigmoid()(d_out))

