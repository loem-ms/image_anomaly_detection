import torch
from torch import nn
import torch.nn.functional as F

import torchvision.models as models


class ResNetVAE(nn.Module):
    def __init__(self, encoded_dim, ffn_dim1, ffn_dim2, dropout=0.):
        super(ResNetVAE, self).__init__()
        
        self.dropout = dropout
        
        resnet = models.resnet18(pretrained=True)
        encoder_blocks = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*encoder_blocks)
        self.ffn1 = nn.Linear(resnet.fc.in_features, ffn_dim1)
        self.batch_norm1 = nn.BatchNorm1d(ffn_dim1, momentum=0.01)
        self.ffn2 = nn.Linear(ffn_dim1, ffn_dim2)
        self.batch_norm2 = nn.BatchNorm1d(ffn_dim2, momentum=0.01)
        self.ffnout = nn.Linear(ffn_dim2, encoded_dim)
        
        self.ffn3 = nn.Linear(encoded_dim, ffn_dim2)
        self.batch_norm3 = nn.BatchNorm1d(ffn_dim2)
        self.ffn4 = nn.Linear(ffn_dim2, 128 * 7 * 7)
        self.batch_norm4 = nn.BatchNorm1d(128 * 7 * 7)
        self.activation = nn.ReLU(True)
        
        self.convtranspose3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(True),
        )
        self.convtranspose2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(0, 0)),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(True),
        )
        self.convtranspose1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=3,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(0, 0)),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
    
        x = self.batch_norm1(self.ffn1(x))
        x = self.activation(x)
        x = self.batch_norm2(self.ffn2(x))
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ffnout(x)
        
        return x
        
    
    def decode(self, x):
        x = self.batch_norm3(self.ffn3(x))
        x = self.activation(x)
        x = self.batch_norm4(self.ffn4(x))
        x = self.activation(x)
        x = x.view(-1, 128, 7, 7)
        x = self.convtranspose3(x)
        x = self.convtranspose2(x)
        x = self.convtranspose1(x)
        x = F.interpolate(x, size=(900, 900), mode='nearest')
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x