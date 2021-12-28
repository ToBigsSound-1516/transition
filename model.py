import torch
from const import *

class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding):
        super().__init__()
        self.transconv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride, padding)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)

class Model(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv6 = GeneraterBlock(5, 5, (1, 2, 1), (1, 2, 1))
        self.conv5 = ConvolutionBlock(5, 16, (1, 1, 12), (1, 1, 12), (0, 0, 0))
        self.conv4 = ConvolutionBlock(16, 32, (1, 4, 1), (1, 4, 1), (0, 0, 0))
        self.conv3 = ConvolutionBlock(32, 64, (1, 1, 3), (1, 1, 1), (0, 0, 0))
        self.conv2 = ConvolutionBlock(64, 128, (1, 1, 4), (1, 1, 4), (0, 0, 0))
        self.conv1 = ConvolutionBlock(128, 256, (1, 4, 1), (1, 4, 1), (0, 0, 0))
        self.linear = torch.nn.Linear(256*4, latent_dim)

        self.transconv0 = GeneraterBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = torch.nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

    def forward(self, x):
        """ENCODER"""
        x = x.unsqueeze(dim=2)
        c6 = self.conv6(x)
        c6 = c6.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
        c5 = self.conv5(c6)
        c4 = self.conv4(c5)
        c3 = self.conv3(c4)
        c2 = self.conv2(c3)
        c1 = self.conv1(c2)
        c0 = c1.view(-1, 256*4)
        x = self.linear(c0)
        
        """DECODER"""
        x = x.view(-1, self.latent_dim, 1, 1, 1)
        x = self.transconv0(x) + c1
        x = self.transconv1(x) + c2
        x = self.transconv2(x) + c3
        x = self.transconv3(x) + c4
        x = [transconv(x)+c5 for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1) + c6
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x
