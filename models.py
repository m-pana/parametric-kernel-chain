import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Kernel(nn.Module):
	def forward(self, X_i, X_j=None):
		if X_j is not None:
			return self.compute_kernel(X_i, X_j)
		else:
			return self.compute_kernel(X_i, X_i)

class ParametricRBF(Kernel):
	"""
	Implements RBF learnable kernel  
	"""
	def __init__(self, feature_in, feature_out, variance=1, lengthscale=1):
		super(ParametricRBF, self).__init__()
		self.variance = variance
		self.lengthscale = lengthscale
		self.W = torch.nn.parameter.Parameter(torch.randn(feature_out, feature_in)/feature_in)
		#nn.init.xavier_normal_(self.W) # XAVIER GLOROT

	def compute_kernel(self, X_i, X_j):
		assert X_i.shape[1] == self.W.shape[1] and X_j.shape[1] == self.W.shape[1], "Mismatch in input dimensionality and W matrix"
		#op1 = X_i @ self.W.T
		#op2 = X_j @ self.W.T
		op1 = torch.mul(X_i, self.W)
		op2 = torch.mul(X_j, self.W)
		dist = torch.cdist(op1, op2)**2
		return self.variance * torch.exp(- dist/(self.lengthscale **2))

class ParametricExponential(Kernel):
	"""
	Implements Exponential learnable kernel  
	"""
	def __init__(self, feature_in, feature_out, variance=1):
		super(ParametricExponential, self).__init__()
		self.variance = variance
		self.W = torch.nn.parameter.Parameter(torch.randn(feature_out, feature_in))
		nn.init.xavier_normal_(self.W) # XAVIER GLOROT

	def compute_kernel(self, X_i, X_j):
		assert X_i.shape[1] == self.W.shape[1] and X_j.shape[1] == self.W.shape[1], "Mismatch in input dimensionality and W matrix"
		
		op1 = torch.mul(X_i, self.W)
		op2 = torch.mul(X_j, self.W)
		dist = torch.cdist(op1, op2)
		return torch.exp(- dist/(2*self.variance **2))

class ParametricAnova(Kernel):
	"""
	Implements Anova learnable kernel  
	"""
	def __init__(self, feature_in, feature_out, degree = 2, n = 5, stdev=1):
		super(ParametricAnova, self).__init__()
		self.degree = degree
		self.n = n
		self.stdev = stdev
		self.W = torch.nn.parameter.Parameter(torch.randn(feature_out, feature_in))
		nn.init.xavier_normal_(self.W) # XAVIER GLOROT

	def compute_kernel(self, X_i, X_j):
		assert X_i.shape[1] == self.W.shape[1] and X_j.shape[1] == self.W.shape[1], "Mismatch in input dimensionality and W matrix"
		
		op1 = torch.mul(X_i, self.W)
		op2 = torch.mul(X_j, self.W)
		kernel = torch.zeros((X_i.shape[0], X_j.shape[0])).to(device)
		for k in range(1, self.n+1):
			dist = torch.cdist(torch.pow(op1,k), torch.pow(op2,k))**2
			kernel = torch.add(kernel, torch.pow(torch.exp(-self.stdev*(dist)),self.degree))
		
		return kernel

class ParametricInverseMultiquadratic(Kernel):
	"""
	Implements Parametric Inverse Multiquadratic learnable kernel
	Alternative to RBF: infinite future space 
	"""
	def __init__(self, feature_in, feature_out, c = 1):
		super(ParametricInverseMultiquadratic, self).__init__()
		self.c = c
		self.W = torch.nn.parameter.Parameter(torch.randn(feature_out, feature_in))
		nn.init.xavier_normal_(self.W) # XAVIER GLOROT

	def compute_kernel(self, X_i, X_j):
		assert X_i.shape[1] == self.W.shape[1] and X_j.shape[1] == self.W.shape[1], "Mismatch in input dimensionality and W matrix"
		
		op1 = torch.mul(X_i, self.W)
		op2 = torch.mul(X_j, self.W)
		dist = torch.cdist(op1, op2)**2
		
		return 1 / torch.sqrt(dist + self.c**2)


class ParametricChain(nn.Module):
	def __init__(self, kernel, lambda_reg=1):
		super(ParametricChain, self).__init__()
		self.kernel = kernel
		self.lambda_reg = lambda_reg
		#self.W = self.kernel.W
		

	def loss_fn(self, target, output):
		return torch.mean((target - output)**2)

	def fit(self, X, labels):
		self.kern = self.kernel(X)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	def compute_loss(self, labels):
		#output = K.T @ self.alpha
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)
		output = self.kern.T @ self.alpha
		loss = self.loss_fn(output, one_hot_y) + self.lambda_reg * torch.trace(self.alpha.T @ self.kern @ self.alpha)
		return loss
	
	def predict(self, batch_train,batch_test, test_labels):
		kern_test = self.kernel(batch_train, batch_test)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total


class ParametricCompositionalChain(nn.Module):
	def __init__(self, kernel, nb_kernels = 3, lambda_reg=1):
		super(ParametricCompositionalChain, self).__init__()
		self.nb_kernels = nb_kernels
		self.kernels = [kernel]*nb_kernels
		self.lambda_reg = lambda_reg
		self.W_comp = torch.nn.ParameterList(
			[torch.nn.parameter.Parameter(torch.randn(1, 1)) for i in range(self.nb_kernels)])
		#self.W = self.kernel.W
		

	def loss_fn(self, target, output):
		return torch.mean((target - output)**2)

	def fit(self, X, labels):
		self.kern = torch.sum(torch.stack([
			self.W_comp[i] * self.kernels[i](X) for i in range(self.nb_kernels)
			]), dim=0)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	def compute_loss(self, labels):
		#output = K.T @ self.alpha
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)
		output = self.kern.T @ self.alpha
		loss = self.loss_fn(output, one_hot_y) + self.lambda_reg * torch.trace(self.alpha.T @ self.kern @ self.alpha)
		return loss
	
	def predict(self, batch_train, batch_test, test_labels):
		kern_test = torch.sum(torch.stack([
			self.W_comp[i] * self.kernels[i](batch_train, batch_test) for i in range(self.nb_kernels)
			]), dim=0)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total

class ActivatedParametricCompositionalChain(nn.Module):
	def __init__(self, kernel, nb_kernels = 3, activation_fn = nn.ReLU(), lambda_reg=1):
		super(ActivatedParametricCompositionalChain, self).__init__()
		self.nb_kernels = nb_kernels
		self.kernels = [kernel]*nb_kernels
		self.lambda_reg = lambda_reg
		self.W_comp = torch.nn.ParameterList(
			[torch.nn.parameter.Parameter(torch.randn(1, 1)) for i in range(self.nb_kernels)])
		self.activation_fn = activation_fn
		#self.W = self.kernel.W
		

	def loss_fn(self, target, output):
		return torch.mean((target - output)**2)

	def fit(self, X, labels):
		self.kern = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernels[i](X)) for i in range(self.nb_kernels)
			]), dim=0)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	def compute_loss(self, labels):
		#output = K.T @ self.alpha
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)
		output = self.kern.T @ self.alpha
		loss = self.loss_fn(output, one_hot_y) + self.lambda_reg * torch.trace(self.alpha.T @ self.kern @ self.alpha)
		return loss
	
	def predict(self, batch_train, batch_test, test_labels):
		kern_test = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernels[i](batch_train, batch_test))
			for i in range(self.nb_kernels)
			]), dim=0)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total

class SkipConnParametricCompositionalChain(nn.Module):
	def __init__(self, kernel, nb_kernels = 3, activation_fn = nn.ReLU(),lambda_reg=1):
		super(SkipConnParametricCompositionalChain, self).__init__()
		self.nb_kernels = nb_kernels
		self.kernels = [kernel]*nb_kernels
		self.lambda_reg = lambda_reg
		self.W_comp = torch.nn.ParameterList(
			[torch.nn.parameter.Parameter(torch.randn(1, 1)) for i in range(self.nb_kernels)])
		self.activation_fn = activation_fn
		#self.W = self.kernel.W
		

	def loss_fn(self, target, output):
		return torch.mean((target - output)**2)

	def fit(self, X, labels):
		self.kern = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernels[i](X)) + self.W_comp[i] * self.kernels[i](X)
			for i in range(self.nb_kernels)
			]), dim=0)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	def compute_loss(self, labels):
		#output = K.T @ self.alpha
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)
		output = self.kern.T @ self.alpha
		loss = self.loss_fn(output, one_hot_y) + self.lambda_reg * torch.trace(self.alpha.T @ self.kern @ self.alpha)
		return loss
	
	def predict(self, batch_train, batch_test, test_labels):
		kern_test = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernels[i](batch_train, batch_test))
			for i in range(self.nb_kernels)
			]), dim=0)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total
