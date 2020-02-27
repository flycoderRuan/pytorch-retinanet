import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

import skimage.io
import skimage.transform
import skimage.color
import skimage

def main(args=None):
	retinanet = torch.load('/home/pytorch-retinanet/csv_retinanet_35.pt')
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	path = '/home/test_images_apart/image24624-blur-5-300.jpg'

	img = skimage.io.imread(path)

	if len(img.shape) == 2:
		img = skimage.color.gray2rgb(img)

	img =  img.astype(np.float32) / 255.0

	img_tensor = torch.tensor(img, dtype=torch.float32)
	img_tensor = img_tensor.unsqueeze(0)
	img_tensor = img_tensor.permute(0, 3, 1, 2)
	print(img_tensor.size())
	print(type(img_tensor))
	with torch.no_grad():
		scores, classification, transformed_anchors = retinanet(img_tensor.cuda().float())
		print(transformed_anchors)
	print(type(img))


if __name__ == '__main__':
    main()