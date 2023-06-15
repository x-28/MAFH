import torch
#import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.autograd import Function
#from torchvision import models
from torchvision import utils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'
import torch.nn as nn

from new_models.resnet import resnet18,resnet34
plt.switch_backend('agg')

model =resnet34(32, 4)


model.load_state_dict(torch.load('D:/lixue/cross-modal/20220812DCMH/MESDCH-master/pth/32-ASCHN_resnet34.pth'))


def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input

def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	plt.imshow(cam[..., -1::-1])#RGB-BRG转换
	plt.axis('off') # 关掉坐标轴为 off
	plt.show()
	cv2.imwrite("our_12_14.jpg", np.uint8(255 * cam))
	#cv2.imwrite("cam_18.png", np.uint8(255 * cam))

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)
    def __call__(self, x):
        outputs = []
        self.gradients = []
        # x = self.model.feature_layers(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        for name, module in self.model._modules.items():##resnet50没有.feature这个特征，直接删除用就可以。
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        x = self.model.global_avgpool(x)

        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):

		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)#torch.Size([1, 512])
		return target_activations, output

class GradCam():
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		# exit()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):

		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)
		

		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		# print(one_hot.shape)#(1, 1000)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		self.model.zero_grad()##features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
		#self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		target = features[-1]

		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]

		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):

			cam += w * target[i, :, :]
		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (192, 192))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

def get_args():
	parser = argparse.ArgumentParser()  #
	parser.add_argument('-use-cuda', action='store_true', default=True,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('-image', type=str, default='D:/lixue/cross-modal/20220812DCMH/MESDCH-master/hotmap/0009_2741003681.jpg',  #图片放这个文件夹里
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")
	return args

if __name__ == '__main__':

	args = get_args()
	# model = resnet18(pretrained=True)#这里相对vgg19而言我们处理的不一样，这里需要删除fc层，因为后面model用到的时候会用不到fc层，只查到fc层之前的所有层数。
	model = resnet34(32, 4)
	model.load_state_dict(torch.load('D:/lixue/cross-modal/20220812DCMH/MESDCH-master/pth/32-ASCHN_resnet34.pth'))
	grad_cam = GradCam(model,target_layer_names = ["layer4"], use_cuda=args.use_cuda)  ##这里改成"layer4"（3,2,1）也很简单，我把每层name和size都打印出来了，想看哪层自己直接嵌套就可以了。（最后你会在终端看得到name的）
	img = cv2.imread(args.image)
	img = np.float32(cv2.resize(img, (192, 192))) / 255
	input = preprocess_image(img)
	input.required_grad = True
	target_index =None
	mask = grad_cam(input, target_index)
	i = 0
	i=i+1 
	show_cam_on_image(img, mask,i)