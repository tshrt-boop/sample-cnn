import torch
import torchvision


def get_dataset(path):

    trans = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(
        root=path, train=True, download=True, transform=trans
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=trans,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0
    )

    return trainloader, testloader
