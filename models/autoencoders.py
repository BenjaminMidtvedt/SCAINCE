import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from . import common

import numpy as np
import pytorch_lightning as pl


class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        # convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_size[0],
                out_channels=self.channel_mult * 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 8, self.channel_mult * 16, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, flat_fts, base_width, base_height):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 3
        self.fc_output_dim = flat_fts // 2
        self.base_width = base_width
        self.base_height = base_height

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True),
        )

        self.deflatten = nn.Sequential(
            nn.Unflatten(1, (self.channel_mult * 8, self.base_width, self.base_height))
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel_mult * 8, self.channel_mult * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.channel_mult * 4, self.channel_mult * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.channel_mult * 2, self.channel_mult * 1, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.channel_mult * 1, self.output_channels, 4, 2, 1, bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.deflatten(x)
        x = self.deconv(x)
        return x


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_shape, latent_dim=64):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = CNN_Encoder(latent_dim, input_shape)

        _tensor = torch.rand(1, *input_shape)
        _conv_out = self.encoder.conv(_tensor)
        print(_conv_out.shape)
        self.decoder = CNN_Decoder(
            latent_dim, self.encoder.flat_fts, _conv_out.shape[2], _conv_out.shape[3]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = F.l1_loss(x_hat, batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
