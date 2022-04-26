import os
import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from itertools import product
from ConvolutionVAE.model.encoder import Encoder
from ConvolutionVAE.model.decoder import Decoder

from datetime import datetime


if __name__ == '__main__':
    # set environments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # set X variables
    X_variables = list()
    data_dir = '/home/jcurtis2/hackathon_data/'
    with open(f'{data_dir}x_aerosols.txt', 'r') as file:
        lines = file.read().splitlines()
        for name in lines:
            X_variables.append(name)
        file.close()

    with open(f'{data_dir}x_gases.txt', 'r') as file:
        lines = file.read().splitlines()
        for name in lines:
            X_variables.append(name)
        file.close()

    # removed list
    removed_list = [
        'TOT_NUM_CONC', 'TOT_MASS_CONC', 'pmc_SO4', 'pmc_NO3', 'pmc_Cl', 'pmc_NH4', 'pmc_ARO1', 'pmc_ARO2', 'pmc_API1',
        'pmc_API2', 'pmc_LIM1', 'pmc_OC', 'h2so4', 'hno3', 'hcl', 'nh3', 'no', 'no3', 'n2o5', 'hono', 'hno4', 'o1d', 'ho2',
        'h2o2', 'co', 'so2', 'ch4', 'c2h6', 'ch3o2', 'hcho', 'ch3oh', 'ANOL', 'ch3ooh', 'ald2', 'hcooh', 'pan', 'aro1',
        'alk1', 'ole1', 'api1', 'api2', 'par', 'mgly', 'eth', 'OLET', 'OLEI', 'cres', 'to2', 'onit', 'ro2', 'ano2', 'xo2',
        'xpar', 'isop', 'isoprd', 'isopp', 'isopn', 'isopo2', 'api', 'dms', 'msa', 'dmso2', 'ch3sch2oo', 'ch3so3',
        'ch3so2oo', 'SULFHOX'
    ]  # 66

    important_features_idx = [1, 36, 6, 26, 2, 11, 0, 41, 19, 3, 13, 5, 8, 25, 47, 48]

    important_features_name = list()
    for idx in important_features_idx:
        important_features_name.append(removed_list[idx])

    # get new index
    important_features_new_index = list()
    for name in important_features_name:
        important_features_new_index.append(X_variables.index(name))

    print(important_features_name)

    for idx in important_features_new_index:
        print(X_variables[idx])

    # load data
    print("Loading data ...", flush=True)
    train_dataset = torch.load('all_normalized_x_train.pth').to(device)
    test_dataset = torch.load('all_normalized_x_test.pth').to(device)
    X_variables_mean = np.array(torch.load('all_x_train_mean_np.pth'))
    X_variables_std = np.array(torch.load('all_x_train_std_np.pth'))

    # set new X_variables & dataset
    train_dataset = train_dataset[:, important_features_new_index, :, :, :]
    test_dataset = test_dataset[:, important_features_new_index, :, :, :]
    X_variables_mean = X_variables_mean[important_features_new_index]
    X_variables_std = X_variables_std[important_features_new_index]

    # get dimension
    time_dim = train_dataset.shape[0]
    z_dim, y_dim, x_dim = train_dataset.shape[2:]

    # set parameters
    latent_dim = 8
    input_channel_dim = train_dataset.shape[1]
    channels = (len(important_features_new_index), 2, 4)
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)

    # partition parameters
    size_of_partition = 13
    overlap_size = 2

    # z idx -> 4 blocks
    z_start_idx = list(np.arange(start=0, stop=z_dim - size_of_partition, step=size_of_partition - 2)) + [
        z_dim - size_of_partition]
    z_stop_idx = list(np.arange(start=size_of_partition, stop=z_dim - 1, step=size_of_partition - 2)) + [z_dim]

    # y idx -> 15 blocks
    y_start_idx = list(np.arange(start=0, stop=y_dim - size_of_partition, step=size_of_partition - 2)) + [
        y_dim - size_of_partition]
    y_stop_idx = list(np.arange(start=size_of_partition, stop=y_dim - 1, step=size_of_partition - 2)) + [y_dim]

    # x idx -> 16 blocks
    x_start_idx = list(np.arange(start=0, stop=x_dim - size_of_partition, step=size_of_partition - 2)) + [
        x_dim - size_of_partition]
    x_stop_idx = list(np.arange(start=size_of_partition, stop=x_dim - 1, step=size_of_partition - 2)) + [x_dim]

    # set encoder
    encoder = Encoder(input_channel_dim=input_channel_dim, z_dim=size_of_partition, y_dim=size_of_partition,
                      x_dim=size_of_partition, latent_dim=latent_dim, channels=channels, kernel_size=kernel_size,
                      stride=stride).to(device)

    # set decoder
    decoder = Decoder(output_channel_dim=input_channel_dim, latent_dim=latent_dim, cnn_feature_dim=encoder.feature_dim,
                      cnn_feature_dim_list=encoder.feature_dim_list, channels=channels, kernel_size=kernel_size,
                      stride=stride).to(device)

    # set train parameters
    learning_rate = 1e-3
    num_epochs = 100
    criterion = nn.MSELoss(reduction='mean').to(device)
    mae_loss = nn.L1Loss()
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)

    # save directory
    parent_directory = '/home/seonghwan/2022HackathonAerosols/'
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H:%M:%S")
    save_path = os.path.join(parent_directory, 'cnn_vae_%s' % dt_string)
    os.mkdir(save_path)

    # train
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        # make plot save directory
        epoch_file_train_path = os.path.join(save_path, f'train_epoch_{epoch + 1}')
        os.mkdir(epoch_file_train_path)
        print("===== EPOCH %d Training =====" % (epoch + 1), flush=True)
        # pbar = tqdm(range(len(train_dataset)), total=len(train_dataset), leave=True)
        # with pbar as t:
        for data_idx in range(len(train_dataset)):
            # get data at each time
            train_data = train_dataset[data_idx].unsqueeze(dim=0)

            # data partitioning (x, y, z)
            for partition_idx in product(range(len(z_start_idx)), range(len(y_start_idx)), range(len(x_start_idx))):
                # get partition idx
                z_partition_start_idx = z_start_idx[partition_idx[0]]
                z_partition_end_idx = z_stop_idx[partition_idx[0]]
                y_partition_start_idx = y_start_idx[partition_idx[1]]
                y_partition_end_idx = y_stop_idx[partition_idx[1]]
                x_partition_start_idx = x_start_idx[partition_idx[2]]
                x_partition_end_idx = x_stop_idx[partition_idx[2]]

                # get partitioned data
                partitioned_data = train_data[0, :, z_partition_start_idx: z_partition_end_idx,
                                   y_partition_start_idx: y_partition_end_idx,
                                   x_partition_start_idx: x_partition_end_idx]

                # encode
                z_samples, mus, log_vars = encoder(partitioned_data.unsqueeze(0))

                # decode
                reconstructed_data = decoder(z_samples)

                kl_divergence = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
                # kl_divergence *= 1e-5
                # print("KL divergence loss is %.4f" % kl_divergence, flush=True)
                prediction_loss = criterion(input=reconstructed_data.flatten(), target=partitioned_data.flatten())
                loss = kl_divergence + prediction_loss

                # backpropagation
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                loss.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()

                # print error
                print('train_time_%d_(%d_%d_%d)_partitioned_train_prediction_MSE_loss=%.4f' %
                      (data_idx, partition_idx[0], partition_idx[1], partition_idx[2], float(prediction_loss)),
                      flush=True)

            # save model
            torch.save(encoder.state_dict(),
                       os.path.join(epoch_file_train_path, 'trained_encoder_time_%d.pth' % data_idx))
            torch.save(decoder.state_dict(),
                       os.path.join(epoch_file_train_path, 'trained_decoder_time_%d.pth' % data_idx))

        # Test
        print("===== EPOCH %d Test =====" % (epoch + 1), flush=True)
        epoch_file_test_path = os.path.join(save_path, f'test_epoch_{epoch + 1}')
        os.mkdir(epoch_file_test_path)  # make plot save directory
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            for data_idx in range(len(test_dataset)):
                # store values
                mse_list = list()
                mae_list = list()
                relative_mae_list = list()

                # load data (at certain time)
                test_data = test_dataset[data_idx].unsqueeze(dim=0)

                # data partitioning (x, y, z)
                for partition_idx in product(range(len(z_start_idx)), range(len(y_start_idx)),
                                             range(len(x_start_idx))):
                    # get partition idx
                    z_partition_start_idx = z_start_idx[partition_idx[0]]
                    z_partition_end_idx = z_stop_idx[partition_idx[0]]
                    y_partition_start_idx = y_start_idx[partition_idx[1]]
                    y_partition_end_idx = y_stop_idx[partition_idx[1]]
                    x_partition_start_idx = x_start_idx[partition_idx[2]]
                    x_partition_end_idx = x_stop_idx[partition_idx[2]]

                    # get partitioned data
                    partitioned_data = test_data[0, :, z_partition_start_idx: z_partition_end_idx,
                                       y_partition_start_idx: y_partition_end_idx,
                                       x_partition_start_idx: x_partition_end_idx]

                    # encode
                    z_samples, mus, log_vars = encoder(partitioned_data.unsqueeze(0))

                    # decode
                    reconstructed_data = decoder(z_samples)

                    test_prediction_loss = criterion(input=reconstructed_data.flatten(),
                                                     target=partitioned_data.flatten())

                    # print error
                    print('test_time_%d_(%d_%d_%d)_partitioned_train_prediction_MSE_loss=%.4f' %
                          (data_idx, partition_idx[0], partition_idx[1], partition_idx[2],
                           float(test_prediction_loss)), flush=True)
