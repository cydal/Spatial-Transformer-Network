
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn.functional as F


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label



def get_train_transform(MEAN, STD, PAD=4):

    train_transform = A.Compose([
                                A.Normalize(mean=(MEAN), 
                                            std=STD),
                                ToTensorV2(),
    ])
    return(train_transform)


def get_test_transform(MEAN, STD):

    test_transform = A.Compose([
                                A.Normalize(mean=MEAN, 
                                            std=STD),
                                ToTensorV2(),
    ])
    return(test_transform)


def get_train_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      trainloader: DataLoader Object
  """
  if transform:
    trainset = Cifar10SearchDataset(transform=transform)
  else:
    trainset = Cifar10SearchDataset(root="~/data/cifar10", train=True, 
                                    download=True)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)
  return(trainloader)


def get_test_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      testloader: DataLoader Object
  """
  if transform:
    testset = Cifar10SearchDataset(transform=transform, train=False)
  else:
    testset = Cifar10SearchDataset(train=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, 
                                         shuffle=False, num_workers=2)

  return(testloader)





def train(model, train_loader, epoch, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure STN the performances on MNIST.
#


def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))



def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn(model, test_loader, device):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')