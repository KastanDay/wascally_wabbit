import torch
import torch.nn as nn

from math import floor

class Encoder(nn.Module):  # noqa
    def __init__(self, input_channel_dim, z_dim, y_dim, x_dim, latent_dim, channels, kernel_size, stride):
        """
        Convolutional layers to encode climate data to latent space
        """
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.padding = (0, 0, 0)
        self.number_of_conv = len(channels) - 1

        # Reduce dimension up to second last layer of Encoder
        layers = list()
        for num in range(self.number_of_conv):
            layers.append(
                nn.Conv3d(in_channels=channels[num], out_channels=channels[num + 1], kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding),
            )
            layers.append(nn.BatchNorm3d(num_features=channels[num + 1]))
            layers.append(nn.LeakyReLU())

        self.convolution_layer = nn.Sequential(*layers)

        def get_feature_dim(dim_list, kernel_size, padding, stride, dilation=1):
            output_dim = list()
            for idx, dim in enumerate(dim_list):
                output_dim.append(
                    floor((dim + 2 * padding[idx] - dilation * (kernel_size[idx] - 1) - 1) / stride[idx] + 1)
                )
            return output_dim

        # calculate modified dim by Conv3d and MaxPool3d
        self.feature_dim_list = [[z_dim, y_dim, x_dim]]
        for num in range(self.number_of_conv):
            self.feature_dim_list.append(
                get_feature_dim(self.feature_dim_list[-1], kernel_size=self.kernel_size, padding=self.padding,
                                stride=self.stride)
            )
        self.feature_dim = self.feature_dim_list[-1][0] * self.feature_dim_list[-1][1] * self.feature_dim_list[-1][2]

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(64, self.latent_dim)

        # Latent space variance
        self.encode_log_var = nn.Linear(64, self.latent_dim)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass through the Encoder
        """
        # Get results of encoder network
        hidden = self.convolution_layer(x)
        hidden = hidden.view(-1, self.feature_dim)
        hidden = self.fully_connected_layer(hidden)

        # latent space
        mu = self.encode_mu(hidden)
        log_var = self.encode_log_var(hidden)

        # Re-parameterize
        z = self.reparameterize(mu, log_var)

        return z, mu, log_var
