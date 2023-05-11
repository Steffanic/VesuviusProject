import time
import torchvision
import torch
from torch.utils.data import DataLoader
from FragmentWithInkCropDataset import FragmentWithInkCropDataset
from torch.utils.tensorboard import SummaryWriter


class DeepConverter(torch.nn.Module):
    '''
    Converts the 65 layers into just 3 channels for the deeplabv3
    '''
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(65, 3, 1)
        self.batchnorm1 = torch.nn.BatchNorm2d(3)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        return x

if __name__ =="__main__":
    converter = DeepConverter()
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=2, aux_loss=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print(model)
    # make a tensor and scale between 0 and 1
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = FragmentWithInkCropDataset("train/1", crop_size=1024, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    writer = SummaryWriter(log_dir=f"runs\{time.time()}")

    for i, data in enumerate(dataloader):
        optimizer.zero_grad()

        images_batch, mask, target = data
        print(f"{images_batch.shape=}")
        print(f"{mask.shape=}")
        print(f"{target.shape=}")
        images_batch = images_batch.reshape(images_batch.shape[0], -1, 1024, 1024)
        print(f"{images_batch.shape=}")
        converted_input = converter(images_batch)
        print(f"{converted_input.shape=}")
        output = model(converted_input)['out']
        # i want to convert the first dim to a 1d and make its value 1 class 2 is predicted and 0 if not 
        output= torch.argmax(output, dim=1, keepdim=True)

        # compute the loss
        loss = loss_fn(output, target)

        loss = torch.autograd.Variable(loss, requires_grad=True)

        print(f"{loss=}")
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss, i)
        writer.add_images("Input", images_batch[:,0,:,:].reshape(-1,1,1024,1024), i, dataformats='NCHW')
        writer.add_images("Output", output, i, dataformats='NCHW')
        writer.add_images("Target", target, i, dataformats='NCHW')

        

    