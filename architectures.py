# -*- coding: utf-8 -*-

import torch


class SimpleCNN(torch.nn.Module):
    def __init__(
        self,
        n_in_channels: int = 4,
        n_hidden_layers: int = 5,
        n_kernels: int = 64,
        kernel_size: int = 3,
    ):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()  # Call the constructor of the parent class (torch.nn.Module)

        cnn = []  # list of layers
        for i in range(n_hidden_layers):
            cnn.append(
                torch.nn.Conv2d(
                    in_channels=n_in_channels,  # number of input channels
                    out_channels=n_kernels,  # number of output channels
                    kernel_size=kernel_size,  # size of the convolution kernel
                    padding=int(
                        kernel_size / 2
                    ),  # padding to keep the spatial dimensions constant
                )
            )
            cnn.append(torch.nn.ReLU())  # ReLU activation
            n_in_channels = n_kernels  # number of input channels for the next layer is the number of output channels of the previous layer
        self.hidden_layers = torch.nn.Sequential(*cnn)  # apply all layers in sequence

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,  # number of input channels
            out_channels=3,  # number of output channels
            kernel_size=kernel_size,  # size of the convolution kernel
            padding=int(
                kernel_size / 2
            ),  # padding to keep the spatial dimensions constant
        )

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(
            x
        )  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(
            cnn_out
        )  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        pred = torch.clamp(pred, 0, 255)
        return pred  # return predictions (N, 3, X, Y)
