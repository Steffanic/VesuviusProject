from matplotlib import pyplot as plt
from UNet import UNet
from torch.utils.data import DataLoader
from FragmentWithInkCropDataset import FragmentWithInkCropDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms


def train():
    # define the transforms
    # to tensor
    # then normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Load the dataset
    dataset = FragmentWithInkCropDataset('train/1', crop_size=256 , transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load the model
    model = UNet()

    # Load the loss function
    loss_function = torch.nn.L1Loss()

    # Load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Load the summary writer
    writer = SummaryWriter()

    # Train the model
    for epoch in range(100):
        for i, data in enumerate(dataloader):
            images_batch, mask, target = data
            # check if sum(mask) == 0, if so, skip this batch
            if torch.sum(mask) == 0:
                print(f"Skipping batch {i} because sum(mask) == 0")
                continue
            if torch.sum(target) == 0:
                print(f"Skipping batch {i} because sum(target) == 0")
                continue
            else:
                print(f"sum(mask): {torch.sum(mask)}")
                print(f"sum(target): {torch.sum(target)}")
            print(f"images_batch.shape: {images_batch.shape}")
            print(f"mask.shape: {mask.shape}")
            print(f"target.shape: {target.shape}")
            images_batch = images_batch.reshape(2, -1, 256, 256)
            print(f"images_batch.shape: {images_batch.shape}")
            optimizer.zero_grad()
            output = model(images_batch)
            # stack copies of target to match output shape
            targets = torch.stack([target[0]] * output.shape[0])
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()
            print('Epoch: ', epoch, 'Batch: ', i, 'Loss: ', loss.item())
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
            [writer.add_images('images', images_batch[0][j], epoch * len(dataloader) + i, dataformats="HW") for j in range(len(images_batch))]
            writer.add_images('target', targets.detach().numpy(), epoch * len(dataloader) + i)
            writer.add_images('output', output, epoch * len(dataloader) + i)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)
        torch.save(model.state_dict(), 'model.pth')
        torch.save(optimizer.state_dict(), 'optimizer.pth')


if __name__ == '__main__':
    train()
