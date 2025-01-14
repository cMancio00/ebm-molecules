import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import argparse

print("CUDA available:", torch.cuda.is_available())

def prepare_data(train_batch_size, test_batch_size, validation_split):
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
    validation_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=train_batch_size, shuffle=False)
    
    test_dataset = datasets.MNIST('./datasets', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-split', type=float, default=0.15, metavar='V',
                        help='input size to use in validation (default: 0.15)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)


    train_loader, validation_loader, test_loader = prepare_data(args.train_batch_size, args.test_batch_size, args.validation_split)

    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Validation dataset size: {len(validation_loader.dataset)}')
    print(f'Test dataset size: {len(test_loader.dataset)}')

if __name__ == '__main__':
    main()


