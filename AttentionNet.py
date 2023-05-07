from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from FragmentWithInkCropDataset import FragmentWithInkCropDataset

class AttentionNet(torch.nn.Module):
    '''
    Processes a sequence of images and returns a sequence of masks
    Each input image is embedded into a vector by a CNN 
    
    '''
    def __init__(self) -> None:
        super().__init__()

        # CNN to embed the input images
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Transformer  
        self.transformer = torch.nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            custom_encoder=None,
            custom_decoder=None
        )

        # CNN to decode the output of the transformer
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x_src, x_tgt, seq_length = 32):
        '''
        x_src: a sequence of images of shape (batch_size, sequence_length, num_channels, height, width)
        x_tgt: a sequence of inklabels of shape (batch_size, 1, 1,height, width)
        '''
        # reshape the input images to (batch_size * sequence_length, num_channels, height, width)
        x_src = x_src.flatten(0, 1)
        # reshape the target inklabels to (batch_size, 1, height, width)
        x_tgt = x_tgt.flatten(1, 2)
        # embed the input images
        x_src = self.cnn(x_src)
        # embed the target inklabels
        x_tgt = self.cnn(x_tgt)

        # reshape the input images back into (batch_size, sequence_length, d_model)
        x_src = x_src.reshape(-1, seq_length, 256)
        # reshape the target inklabels back into (batch_size, 1, d_model)
        x_tgt = x_tgt.reshape(-1, 1, 256)

        # compute the attention masks
        attention_mask = torch.ones(seq_length, seq_length)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))

        # compute the output of the transformer
        output = self.transformer(x_src, x_tgt, tgt_mask=attention_mask)
        # reshape the output of the transformer into (batch_size, 1, height, width)
        output = output.reshape(-1, 1, 128, 128)
        # decode the output of the transformer
        output = self.decoder(output)
        return output
    


if __name__ == '__main__':
    # test the AttentionNet
    net = AttentionNet()
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = FragmentWithInkCropDataset("train/1", crop_size=128, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for images_batch, mask, target in dataloader:
        print(images_batch.shape)
        print(mask.shape)
        print(target.shape)
        # remove all size 1 dimensions from the batch

        print(images_batch.shape)
        print(mask.shape)
        output = net(images_batch, target)
        print(output.shape)
        plt.imshow(output[0, 0, :, :].detach().numpy())
        break