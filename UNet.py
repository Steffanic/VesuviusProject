import torch 
import numpy as np

class UNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(1, 64, 3, padding="same")
        self.batchnorm1_1 = torch.nn.BatchNorm2d(64)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding="same")
        self.batchnorm1_2 = torch.nn.BatchNorm2d(64)

        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, padding="same")
        self.batchnorm2_1 = torch.nn.BatchNorm2d(128)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding="same")
        self.batchnorm2_2 = torch.nn.BatchNorm2d(128)


        self.pool2 = torch.nn.MaxPool2d(2)

        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, padding="same")
        self.batchnorm3_1 = torch.nn.BatchNorm2d(256)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding="same")
        self.batchnorm3_2 = torch.nn.BatchNorm2d(256)


        # now the upsampling path 
        self.upsample1 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        # concatenation happens here, that's why we need 256 channels
        self.conv4_1 = torch.nn.Conv2d(256, 128, 3, padding="same")
        self.batchnorm4_1 = torch.nn.BatchNorm2d(128)
        self.conv4_2 = torch.nn.Conv2d(128, 128, 3, padding="same")
        self.batchnorm4_2 = torch.nn.BatchNorm2d(128)


        self.upsample2 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5_1 = torch.nn.Conv2d(128, 64, 3, padding="same")
        self.batchnorm5_1 = torch.nn.BatchNorm2d(64)
        self.conv5_2 = torch.nn.Conv2d(64, 64, 3, padding="same")
        self.batchnorm5_2 = torch.nn.BatchNorm2d(64)

        self.conv6 = torch.nn.Conv2d(64, 1, 1, padding="same")

        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        # first block
        x = self.conv1_1(x)
        x = self.batchnorm1_1(x)
        x = self.activation(x)
        x = self.conv1_2(x)
        x = self.batchnorm1_2(x)
        x = self.activation(x)
        x1 = x
        x = self.pool1(x)

        # second block
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.activation(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = self.activation(x)
        x2 = x
        x = self.pool2(x)

        # third block
        x = self.conv3_1(x)
        x = self.batchnorm3_1(x)
        x = self.activation(x)
        x = self.conv3_2(x)
        x = self.batchnorm3_2(x)
        x = self.activation(x)
        x3 = x

        # now the upsampling path
        x = self.upsample1(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv4_1(x)
        x = self.batchnorm4_1(x)
        x = self.activation(x)
        x = self.conv4_2(x)
        x = self.batchnorm4_2(x)
        x = self.activation(x)

        x = self.upsample2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv5_1(x)
        x = self.batchnorm5_1(x)
        x = self.activation(x)

        x = self.conv6(x)
        x = torch.sigmoid(x)
        return x
    
