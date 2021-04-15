import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Kernel(nn.Module):
    """
    Implements a generic Kernel class.
    """
    def forward(self, X_i, X_j, W):
        return self.compute_kernel(X_i, X_j, W)


class ParametricRBF(Kernel):
    def __init__(self, variance, lengthscale):
        """
        A parametrized RBF kernel.
        :param variance: Variance of the kernel (will be a learnable parameter)
        :param lengthscale: Lengthscale of the kernel (will be a learnable parameter)
        """
        super(ParametricRBF, self).__init__()
        self.variance = nn.parameter.Parameter(torch.Tensor([variance]).type(torch.FloatTensor))
        self.lengthscale = nn.parameter.Parameter(torch.Tensor([lengthscale]).type(torch.FloatTensor))

    def compute_kernel(self, X_i, X_j, W):
        pdb.set_trace()
        dist = torch.cdist(X_i @ W, X_j @ W) ** 2
        # return self.variance * torch.exp(- dist/self.lengthscale **2)
        return torch.exp(- dist)


class ParametricChain(nn.Module):
    def __init__(self, kernel, lambda_reg, dim):
        """
        Models a chain of parametric kernels.
        :param kernel: instance of Kernel class # TODO: we should have a sequential with several of these
        :param lambda_reg: Regularization hyperparameter
        :param dim: dimension of the weight parameters
        """
        super(ParametricChain, self).__init__()
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.W = nn.parameter.Parameter(torch.unsqueeze(torch.randn(size=dim), 1))

    def loss_fn(self, target, output):
        return torch.mean((target - output) ** 2)

    def compute_loss(self, X, labels):
        self.kern = self.kernel(X, X, self.W)
        K = self.kern + torch.eye(self.kern.size()[0]).to(device) * self.lambda_reg

        L = torch.cholesky(K, upper=False)
        one_hot_y = F.one_hot(labels, num_classes=10).type(torch.FloatTensor)
        # A, _ = torch.solve(kern, L)
        # V, _ = torch.solve(one_hot_y, L)
        # alpha = A.T @ V
        self.alpha = torch.cholesky_solve(one_hot_y, L, upper=False)
        # output = K.T @ self.alpha
        output = self.kern.T @ self.alpha
        loss = self.loss_fn(output, one_hot_y) + self.lambda_reg * torch.trace(self.alpha.T @ self.kern @ self.alpha)
        return loss

    def predict(self, batch_train, batch_test, test_labels):
        kern_test = self.kernel(batch_train, batch_test, self.W)
        output = kern_test.T @ self.alpha # let's put a K here? (we kinda need to rethink this in general imho)
        pred_labels = torch.argmax(output, dim=1)
        return torch.sum(pred_labels == test_labels) * 100 / len(pred_labels)