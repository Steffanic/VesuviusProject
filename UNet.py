import torch 
import numpy as np

class UNet(torch.nn.Module):
    def __init__(self, asEmbedding=False, num_conv_per_block=2, num_pools = 2, num_input_channels=65) -> None:
        super().__init__()
        for block_no in range(num_pools):
            for conv_no in range(num_conv_per_block):
                N_input_channels_down = num_input_channels if block_no==0 and conv_no==0 else 32*block_no if conv_no==0 else 32*(block_no+1)
                N_input_channels_up = 
                vars()[f"conv{block_no}_{conv_no}_down"] = torch.nn.Conv2d(N_input_channels_down, 32*(block_no+1), 3, padding="same")
                vars()[f"batchnorm{block_no}_{conv_no}_down"] = torch.nn.BatchNorm2d(32*(block_no+1))
                vars()[f"conv{block_no}_{conv_no}_up"] = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
            vars()[f"pool{block_no}"] = torch.nn.MaxPool2d(2)


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
    
