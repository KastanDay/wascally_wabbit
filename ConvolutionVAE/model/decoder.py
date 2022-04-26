import torch
import torch.nn as nn
import torch.nn.functional as f


class Decoder(nn.Module):  # noqa
    def __init__(self, output_channel_dim, latent_dim, cnn_feature_dim, cnn_feature_dim_list,
                 channels, kernel_size, stride):
        """
        Transpose convolutional layers to decode latent space vectors to original data
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (0, 0, 0)
        self.cnn_feature_dim = cnn_feature_dim
        self.cnn_feature_dim_list = cnn_feature_dim_list

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            nn.Linear(512, self.cnn_feature_dim)
        )

        self.dec_conv_layers = list()
        self.dec_batch_norm_layers = list()
        for num in reversed(range(1, len(channels))):
            self.dec_conv_layers.append(
                nn.ConvTranspose3d(in_channels=channels[num], out_channels=channels[num - 1],
                                   kernel_size=self.kernel_size, stride=self.stride),
            )
            self.dec_batch_norm_layers.append(nn.BatchNorm3d(num_features=channels[num - 1]))

    def forward(self, latent_vectors):
        hidden = self.fully_connected_layer(latent_vectors)
        hidden = hidden.view(-1, *self.cnn_feature_dim_list[-1]).unsqueeze(dim=0)
        cnt = len(self.cnn_feature_dim_list) - 2
        for conv_layer, bn_layer in zip(self.dec_conv_layers[:-1], self.dec_batch_norm_layers[:-1]):
            hidden = conv_layer(hidden, output_size=self.cnn_feature_dim_list[cnt])
            hidden = bn_layer(hidden)
            hidden = f.leaky_relu(hidden)
            cnt -= 1
        hidden = self.dec_conv_layers[-1](hidden, output_size=self.cnn_feature_dim_list[cnt])

        return hidden
