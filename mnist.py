from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load(dname:str, train=True):
    """Params:
    - dname : dataset name ( 'digits' | 'fashion' )
    - train : training or test dataset
    Return:
    - MNIST set dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (.5, .5, .5),
            (.5, .5, .5)
        )
    ])

    if dname == 'fashion':
        dataset = datasets.FashionMNIST('MNIST_fashion', download=True, train=train, transform=transform)
    else:
        dataset = datasets.MNIST('MNIST_data', download=True, train=train, transform=transform)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

def random_img(dataset:str):
    """Params:
    - dataset : 'digits' or 'fashion'
    Return:
    - image as 28x28 tensor
    - image class label as 1x1 tensor
    """
    images, labels = next(iter(load(dataset, train=False)))
    return images[0], labels[0]

fashion_classes = [
    'T-shirt/top', 
    'Trouser', 
    'Pullover', 
    'Dress', 
    'Coat', 
    'Sandal', 
    'Shirt', 
    'Sneaker', 
    'Bag', 
    'Ankle Boot'
]