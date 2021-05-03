import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
plt.style.use('ggplot')


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_MNIST(batch_size_train = 64, batch_size_test = 1000, batch_size_fit = 10000):
	"""
	Method loading MNIST dataset:
	Args:
	- batch_sizes of training, test and fit datasets

	Returns:
	- training, test, fit loaders
	"""

	transform = torchvision.transforms.Compose(
		[torchvision.transforms.ToTensor(),
		 torchvision.transforms.Normalize((0.1307,), (0.3081,))])

	train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
							 transform=transform)

	test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
							 transform=transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

	fit_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_fit, shuffle=True)

	return train_loader, test_loader, fit_loader

def load_CIFAR10(batch_size_train = 64, batch_size_test = 1000, batch_size_fit = 10000):
	"""
	Method loading CIFAR-10 dataset:
	Args:
	- batch_sizes of training, test and fit datasets

	Returns:
	- training, test, fit loaders
	"""

	transform = torchvision.transforms.Compose(
		[torchvision.transforms.ToTensor(),
		 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
							 transform=transform)

	test_dataset = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
							 transform=transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

	fit_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_fit, shuffle=True)

	return train_loader, test_loader, fit_loader



def imshow(example_data):
	"""
	Method to show a grid of images, representing a batch of data
	Args:
	- batch of example data to be plotted

	Plots:
	- A plot of the batch of data in a rectangular/square grid
	"""
	img = torchvision.utils.make_grid(example_data)
	plt.figure(figsize=(10,10))
	npimg = img.numpy()
	
	if len(npimg.shape) > 3:
		npimg = np.squeeze(npimg)
	elif len(npimg.shape) < 3:
		npimg = np.expand_dims(npimg, axis = 0)

	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


def optimize(model, optimizer, train, test, fit, epochs=20, scheduler=None, logger = None):
	"""
	Method to optimize a model of Parametric Kernel(s)
	Args:
	- model: model instance of Parametric Kernel(s)
	- optimizer: the optimizer used for the backward step
	- train, test, fit: train, test, fit loaders
	- epochs: number of iterations to be made for the optimization
	- scheduler (optional): the scheduler of the learning rate parameters
	- logger (optional): the logger method to keep track of the loss
	"""
	print(model)
	model = model.to(device)
	validation_accuracy = 0
	for i in range(1, epochs+1):
		
		#Training Loop
		training_loss = 0

		for idx, (batch_data, batch_labels) in enumerate(train, 0):

			batch_data = torch.flatten(batch_data, start_dim = 1).to(device)
			batch_labels = batch_labels.to(device)

			optimizer.zero_grad()
			model.fit(batch_data, batch_labels)
			loss = model.compute_loss(batch_labels)
			loss.backward()
			optimizer.step()

			training_loss += loss
			print('',end='\r')
			print("Epochs:[{}/{}] {}>{} train_loss: {} val_acc: {}".format(
				i,epochs,"-"*(int(20//(len(train)/(idx+1)))),"-"*(int(20 - 20//(len(train)/(idx+1)))),
				training_loss/(idx+1), validation_accuracy),end='')
		
		if scheduler is not None:
			scheduler.step()
		
		#Validation Loop
		fit_data, fit_labels = next(iter(fit))
		fit_data = torch.flatten(fit_data, start_dim = 1).to(device)
		fit_labels = fit_labels.to(device)

		model.fit(fit_data, fit_labels)

		with torch.no_grad():
			curr_correct, curr_total = 0,0

			for idx2, (test_data, test_labels) in enumerate(test,0):

				test_data = torch.flatten(test_data, start_dim = 1).to(device)
				test_labels = test_labels.to(device)

				correct, total = model.predict(fit_data, test_data, test_labels)

				curr_correct += correct
				curr_total += total

		validation_accuracy = curr_correct / curr_total

		if logger is not None:
			logger.log({
				"training_loss": training_loss/(idx+1),
				"validation_accuracy": validation_accuracy
				})


def show_heatmap(img, weights, dims):
	img = cv2.resize(img, dims)
	heatmap = cv2.resize(weights, dims)

	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)

	result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

	fig, ax = plt.subplots(figsize=(5,5))
	ax.imshow(result)
	plt.show()


def base_predict(model, test, fit):
	"""
	Method to test the effectiveness of a baseline Parametric Kernel
	Args:
	- model: model instance of Parametric Kernel(s)
	- test, fit: train, test, fit loaders
	"""
	assert type(model).__name__ == 'ParametricChain', "The method is intended to work only with the ParametricChain class"
	print(model)
	model = model.to(device)
	model.kernel.W = nn.Parameter(torch.ones_like(model.kernel.W)) #set weights to 1, as for a standard non-parametric kernel

	with torch.no_grad():
		fit_data, fit_labels = next(iter(fit))
		fit_data = torch.flatten(fit_data, start_dim = 1).to(device)
		fit_labels = fit_labels.to(device)

		model.fit(fit_data, fit_labels)


		curr_correct, curr_total = 0,0

		for idx2, (test_data, test_labels) in enumerate(test,0):

			test_data = torch.flatten(test_data, start_dim = 1).to(device)
			test_labels = test_labels.to(device)

			correct, total = model.predict(fit_data, test_data, test_labels)

			curr_correct += correct
			curr_total += total

	validation_accuracy = curr_correct / curr_total

	print("Baseline Non-parametric Kernel accuracy on test: {}%".format(validation_accuracy*100))


