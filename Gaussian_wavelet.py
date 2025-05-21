import torch
import torch.nn as nn
class GaussianWavelet(nn.Module):
    def forward(self,x):
        return x * torch.exp((-x**2)/2)
        
class Sigmpoid(nn.Module):
    def forward(self,x):
        a = 1 + torch.exp(-x)
        return 1/a
class GaussSigmoid(nn.Module):
    def forward(self, x):
        sigmoid_part = 1 / (1 + torch.exp(-x))
        wavelet_part = x * torch.exp(-x**2 / 2)
        return sigmoid_part + wavelet_part

        
    