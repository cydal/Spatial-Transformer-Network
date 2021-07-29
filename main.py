from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from utils import *
from models.model import *


mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784] 


if __name__ == "__main__":


    train_transform = get_train_transform(mean, std)
    test_transform = get_test_transform(mean, std)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset
    train_loader = get_train_loader(transform=train_transform)

    # Test dataset
    test_loader = get_test_loader(transform=test_transform)


    model = Net().to(device)

    print(summary(model, input_size=(3, 32, 32)))

    optimizer = optim.SGD(model.parameters(), lr=0.01)


    for epoch in range(1, 50 + 1):
        train(model, train_loader, epoch, optimizer, device)
        test(model, test_loader, device)

    # Visualize the STN transformation on some input batch
    visualize_stn(model, test_loader, device)
    

    plt.ioff()
    plt.show()