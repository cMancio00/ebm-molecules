import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(2025)

print("CUDA available:", torch.cuda.is_available())

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = datasets.MNIST('./datasets', train=True, download=True,
                    transform=transform)
test_dataset = datasets.MNIST('./datasets', train=False,
                    transform=transform)
train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)

image, labels = next(iter(train_loader))

plt.imshow(image[0].squeeze(), cmap='gray')
plt.title(f'Label: {labels[0].item()}')
plt.axis('off')
plt.show()
plt.savefig('./mnist_image.png')
plt.close()
