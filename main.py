from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

from utils import *
from models.model import *


mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784] 


train_transform = get_train_transform(mean, std)
test_transform = get_test_transform(mean, std)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset
train_loader = get_train_loader(transform=train_transform)

# Test dataset
test_loader = get_test_loader(transform=test_transform)



optimizer = optim.SGD(model.parameters(), lr=0.01)



for epoch in range(1, 1 + 1):
    train(epoch)
    test()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()