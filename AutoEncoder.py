import torch
from torch.utils.data import DataLoader
from FragmentWithInkCropDataset import FragmentWithInkCropDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np



class AutoEncoder(torch.nn.Module):
    def __init__(self, asEmbedding=False) -> None:
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(65, 8, 5)
        self.batchnorm1_1 = torch.nn.BatchNorm2d(8)
        self.conv1_2 = torch.nn.Conv2d(8, 16, 5)
        self.batchnorm1_2 = torch.nn.BatchNorm2d(16)
        self.conv1_3 = torch.nn.Conv2d(16, 32, 5)
        self.batchnorm1_3 = torch.nn.BatchNorm2d(32)

        self.transconv2_1 = torch.nn.ConvTranspose2d(32, 16, 5)
        self.batchnorm2_1 = torch.nn.BatchNorm2d(16)
        self.transconv2_2 = torch.nn.ConvTranspose2d(16, 8, 5)
        self.batchnorm2_2 = torch.nn.BatchNorm2d(8)
        self.transconv2_3 = torch.nn.ConvTranspose2d(8, 1, 5)
        self.batchnorm2_3 = torch.nn.BatchNorm2d(1)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.batchnorm1_1(x)
        x = self.activation(x)
        x = self.conv1_2(x)
        x = self.batchnorm1_2(x)
        x = self.activation(x)
        x = self.conv1_3(x)
        x = self.batchnorm1_3(x)
        x = self.activation(x)

        x1 = torch.flatten(x, start_dim=1)

        x = self.transconv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.activation(x)
        x = self.transconv2_2(x)
        x = self.batchnorm2_2(x)
        x = self.activation(x)
        x = self.transconv2_3(x)
        x = self.batchnorm2_3(x)
        x = self.activation(x)
        x2 = x

        return x1,x2

if __name__=="__main__":
    # make an autoencoder and a fragmentwithinkcropdataset 
    # and train the autoencoder on the dataset
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    model = AutoEncoder()
    dataset = FragmentWithInkCropDataset("train/1", crop_size=128, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()


    for epoch_no in range(100):
        for (i, data) in enumerate(dataloader):
            images_batch, mask, target = data
            print(f"{images_batch.shape=}")
            print(f"{mask.shape=}")
            print(f"{target.shape=}")

            images_batch = images_batch.reshape(images_batch.shape[0], -1, 128, 128)

            print(f"{images_batch.shape=}")

            latent_vecs, output = model(images_batch)
            print(f"{latent_vecs.shape=}")
            print(f"{output.shape=}")

            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(output, target)
            KLloss = torch.nn.functional.kl_div(latent_vecs, torch.zeros_like(latent_vecs))
            loss = loss+KLloss
            loss.backward()

            optimizer.step()
            print('Epoch: ', epoch_no, 'Batch: ', i, 'Loss: ', loss.item())
            # get one chanel and make a batc of images 
            written_img = images_batch[:, 0, :, :].unsqueeze(1)
            writer.add_scalar('Loss/train', loss.item(), epoch_no * len(dataloader) + i)
            writer.add_image('input', written_img, epoch_no * len(dataloader) + i, dataformats='NCHW')
            writer.add_images('target', target.detach().numpy(), epoch_no * len(dataloader) + i)
            writer.add_images('output', output, epoch_no * len(dataloader) + i)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch_no * len(dataloader) + i)
            writer.add_graph(model, images_batch)


