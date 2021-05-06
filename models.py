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
		if self.W.shape[0] != 1:
			op1 = torch.flatten(torch.mul(torch.unsqueeze(X_i, dim=1), self.W), start_dim = 1)
			op2 = torch.flatten(torch.mul(torch.unsqueeze(X_j, dim=1), self.W), start_dim = 1)
		else:
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


class Chain(nn.Module):
	"""
	Base class of Parametric Chain defining loss methods
	"""

	def __init__(self, kernel, loss ='mse', lambda_reg = 1):
		super(Chain, self).__init__()
		self.kernel = kernel
		self.loss = loss
		self.lambda_reg = lambda_reg	


	def MSE_loss(self, labels):
		#output = K.T @ self.alpha
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)
		output = self.kern.T @ self.alpha
		loss = torch.mean((output - one_hot_y)**2) + self.lambda_reg * torch.trace(self.alpha.T @ self.kern @ self.alpha)
		return loss

	def center_kernel(self, K):
		N = K.shape[0]#K is ofdimensions [N,N] where N is the number of samples
		H = torch.eye(N, device=K.device, dtype=torch.float) - torch.ones(size=(N,N), device=K.device, dtype=torch.float)/N
		return H @ K @ H

	def compute_KMA(self, labels):
		#Y has the one-hot vectors
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)
		Y_cent = self.center_kernel( one_hot_y @ one_hot_y.T ) # Y @ Y.T is a matrix with ones in (i,j) position if samples i,j are of the same class
		K_cent = self.center_kernel(self.kern)

		Y_cent_fro = torch.linalg.norm( Y_cent,'fro')
		A = K_cent.flatten(start_dim=0) @ Y_cent.flatten(start_dim=0) #shape: [1,]

		K_fro = torch.sqrt(torch.sum(K_cent**2)+1e-5) #Problems without 1e-5 if happens K having only zeros and the derivative of sqrt... is 1/sqrt()... (due to Frobernius)
		B = K_fro * Y_cent_fro #shape: [1,]
		target_dependencies = A /(B + 1e-5)

		return - target_dependencies

	def compute_loss(self, labels):
		if self.loss == 'mse':
			return self.MSE_loss(labels)

		elif self.loss =='kma':
			return self.compute_KMA(labels)

		else:
			raise ValueError("Invalid loss: need to specify it")


class ParametricChain(Chain):
	"""
	Parametric Chain Model
	Args:
	- kernel: a Parametric Kernel 
	- lambda_reg (optional): entity of the regularization term. If not passed it is set to 1

	Publicly exposed Methods:
	- fit: to fit the model on some data, labels
	- predict: to predict on a batch of data

	"""
	def __init__(self, kernel, loss ='mse', lambda_reg=1):
		super(ParametricChain, self).__init__(kernel, loss, lambda_reg)
		#self.kernel = kernel
		#self.lambda_reg = lambda_reg
		#self.W = self.kernel.W

	
	def fit(self, X, labels):
		"""
		Fit method
		Args:
		- X: a tensor, appropriately flattened, having sizes: (Batch Size, Features, 1)
		- labels: a tensor of labels, having sizes: (Batch Size, 1)

		"""
		self.kern = self.kernel(X)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	
	def predict(self, batch_train,batch_test, test_labels):
		"""
		Predict Method
		Args:
		- batch_train: a Tensor representing a batch of training data of sizes (Batch Size Train, Features, 1)
		- batch_test: a Tensor representing a batch of test data to be predicted of sizes (Batch Size Test, Features, 1)
		- test_labels: a Tensor representing a ground truth of test data of sizes (Batch Size Test, 1)
		"""
		kern_test = self.kernel(batch_train, batch_test)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total


class ParametricCompositionalChain(Chain):
	"""
	Parametric Compositional Chain Model (i.e. Linear Composition of Kernels)
	Args:
	- kernel: a Parametric Kernel 
	- nb_kernels (optional): number of kernels to be combined. If not passed it is set to 1
	- lambda_reg (optional): entity of the regularization term. If not passed it is set to 1

	Publicly exposed Methods:
	- fit: to fit the model on some data, labels
	- predict: to predict on a batch of data

	"""
	def __init__(self, kernel, loss='mse', lambda_reg=1):
		super(ParametricCompositionalChain, self).__init__(kernel, loss, lambda_reg)
		self.nb_kernels = len(self.kernel)
		#self.lambda_reg = lambda_reg
		self.W_comp = nn.Parameter(torch.randn(self.nb_kernels, ))
		#torch.nn.ParameterList(
			#[torch.nn.parameter.Parameter(torch.randn(1, 1)) for i in range(self.nb_kernels)])
		#self.W = self.kernel.W
		

	def fit(self, X, labels):
		"""
		Fit method
		Args:
		- X: a tensor, appropriately flattened, having sizes: (Batch Size, Features, 1)
		- labels: a tensor of labels, having sizes: (Batch Size, 1)

		"""
		#for p in self.W_comp:
		#	p.data.clamp_(0) #projection to ensure positive semi-definiteness

		#W_soft = F.softmax(self.W_comp)

		self.kern = torch.sum(torch.stack([
			self.W_comp[i] * self.kernel[i](X) for i in range(self.nb_kernels)
			]), dim=0)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	
	def predict(self, batch_train, batch_test, test_labels):
		"""
		Predict Method
		Args:
		- batch_train: a Tensor representing a batch of training data of sizes (Batch Size Train, Features, 1)
		- batch_test: a Tensor representing a batch of test data to be predicted of sizes (Batch Size Test, Features, 1)
		- test_labels: a Tensor representing a ground truth of test data of sizes (Batch Size Test, 1)
		"""
		kern_test = torch.sum(torch.stack([
			self.W_comp[i] * self.kernel[i](batch_train, batch_test) for i in range(self.nb_kernels)
			]), dim=0)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total


class ActivatedParametricCompositionalChain(Chain):
	"""
	Activated Parametric Compositional Chain Model (i.e. Linear Composition of Kernels with Activations)
	Args:
	- kernel: a Parametric Kernel 
	- nb_kernels (optional): number of kernels to be combined. If not passed it is set to 1
	- activation_fn (optional): activation function to be used. If not passed it is set to a ReLU
	- lambda_reg (optional): entity of the regularization term. If not passed it is set to 1

	Publicly exposed Methods:
	- fit: to fit the model on some data, labels
	- predict: to predict on a batch of data

	"""
	def __init__(self, kernel, loss='mse', lambda_reg=1, activation_fn = nn.ReLU()):
		super(ActivatedParametricCompositionalChain, self).__init__(kernel, loss, lambda_reg)
		self.nb_kernels = len(self.kernel)
		#self.lambda_reg = lambda_reg
		self.W_comp = nn.Parameter(torch.randn(self.nb_kernels, ))
		#self.W_comp = torch.nn.ParameterList(
			#[torch.nn.parameter.Parameter(torch.randn(1, 1)) for i in range(self.nb_kernels)])
		self.activation_fn = activation_fn
		#self.W = self.kernel.W
		

	def fit(self, X, labels):
		"""
		Fit method
		Args:
		- X: a tensor, appropriately flattened, having sizes: (Batch Size, Features, 1)
		- labels: a tensor of labels, having sizes: (Batch Size, 1)

		"""
		self.kern = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernel[i](X)) for i in range(self.nb_kernels)
			]), dim=0)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	
	def predict(self, batch_train, batch_test, test_labels):
		"""
		Predict Method
		Args:
		- batch_train: a Tensor representing a batch of training data of sizes (Batch Size Train, Features, 1)
		- batch_test: a Tensor representing a batch of test data to be predicted of sizes (Batch Size Test, Features, 1)
		- test_labels: a Tensor representing a ground truth of test data of sizes (Batch Size Test, 1)
		"""
		kern_test = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernel[i](batch_train, batch_test))
			for i in range(self.nb_kernels)
			]), dim=0)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total


class SkipConnParametricCompositionalChain(Chain):
	"""
	Activated Parametric Compositional Chain Model with Skip Connections
	(i.e. Linear Composition of Kernels with Activations and Skip Connections)
	Args:
	- kernel: a Parametric Kernel 
	- nb_kernels (optional): number of kernels to be combined. If not passed it is set to 1
	- activation_fn (optional): activation function to be used. If not passed it is set to a ReLU
	- lambda_reg (optional): entity of the regularization term. If not passed it is set to 1

	Publicly exposed Methods:
	- fit: to fit the model on some data, labels
	- predict: to predict on a batch of data

	"""
	def __init__(self, kernel, loss='mse' ,lambda_reg=1, activation_fn = nn.ReLU()):
		super(SkipConnParametricCompositionalChain, self).__init__(kernel, loss, lambda_reg)
		self.nb_kernels = len(self.kernel)
		#self.lambda_reg = lambda_reg
		self.W_comp = nn.Parameter(torch.randn(self.nb_kernels, ))
		#self.W_comp = torch.nn.ParameterList(
		#	[torch.nn.parameter.Parameter(torch.randn(1, 1)) for i in range(self.nb_kernels)])
		self.activation_fn = activation_fn
		#self.W = self.kernel.W
		

	def fit(self, X, labels):
		"""
		Fit method
		Args:
		- X: a tensor, appropriately flattened, having sizes: (Batch Size, Features, 1)
		- labels: a tensor of labels, having sizes: (Batch Size, 1)

		"""
		self.kern = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernel[i](X)) + self.W_comp[i] * self.kernels[i](X)
			for i in range(self.nb_kernels)
			]), dim=0)
		K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg
		L = torch.cholesky(K, upper=False)
		one_hot_y = F.one_hot(labels, num_classes = 10).type(torch.FloatTensor).to(device)

		#A, _ = torch.solve(kern, L)
		#V, _ = torch.solve(one_hot_y, L)
		#alpha = A.T @ V
		self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)

	
	def predict(self, batch_train, batch_test, test_labels):
		"""
		Predict Method
		Args:
		- batch_train: a Tensor representing a batch of training data of sizes (Batch Size Train, Features, 1)
		- batch_test: a Tensor representing a batch of test data to be predicted of sizes (Batch Size Test, Features, 1)
		- test_labels: a Tensor representing a ground truth of test data of sizes (Batch Size Test, 1)
		"""
		kern_test = torch.sum(torch.stack([
			self.activation_fn(self.W_comp[i] * self.kernel[i](batch_train, batch_test))
			for i in range(self.nb_kernels)
			]), dim=0)
		output = kern_test.T @ self.alpha
		pred_labels = torch.argmax(output, dim = 1)
		#val_acc = torch.sum(pred_labels == test_labels) * 100/ len(pred_labels)
		#return val_acc
		corrects =  torch.sum(pred_labels == test_labels)
		total = len(pred_labels)
		return corrects, total
