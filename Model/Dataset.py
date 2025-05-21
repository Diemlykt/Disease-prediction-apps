import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


class Hparams:
    def __init__(self, train_batch_size=64, test_batch_size=64, learning_rate=0.001, num_epochs=10, val_split=0.15, test_split=0.15, model_path='/content/saved_model', dataset_path='/content/Data'):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.test_split = test_split
        self.model_path = model_path
        self.dataset_path = dataset_path

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
            mean = torch.mean(x)
            std = torch.std(x)
        return x, mean, std, y
    
    def __len__(self):
        return len(self.subset)
    
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor(),
    ])

    return transform

def get_sample_weights(dataset, train_dataset):
    
    # Code taken from:
    #     https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
    y_train_indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in y_train_indices]
    
    class_sample_counts = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    
    weights = 1. / class_sample_counts
    sample_weights = np.array([weights[t] for t in y_train])
    sample_weights = torch.from_numpy(sample_weights)
    
    return sample_weights

def get_data_loaders(hparams):
    # Loading the dataset
    dataset = datasets.ImageFolder(hparams.dataset_path,
                                   transform=transforms.Compose([transforms.Grayscale()]))
    
    # Splitting dataset into train, validation and test partitions.
    proportions = [(1 - hparams.val_split - hparams.test_split), hparams.val_split, hparams.test_split]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    
    print(f'train size: {lengths[0]}, val size: {lengths[1]}, test size: {lengths[2]}')
    
    data_transforms = {
        'train': get_transforms(),
        'test': get_transforms()
    }
    
    # Using WeightedRandomSampler to overcome unbalance problem
    sample_weights = get_sample_weights(dataset, train_dataset)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))
    
    train_dataset = CustomDataset(train_dataset, transform=data_transforms['train'])
    val_dataset = CustomDataset(val_dataset, transform=data_transforms['test'])
    test_dataset = CustomDataset(test_dataset, transform=data_transforms['test'])
    
    # Creating loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.train_batch_size, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.train_batch_size, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.test_batch_size)

    return train_loader, val_loader, test_loader

def get_input_customer(img_path):
    # Load the image (make sure it's RGB)
    image = Image.open(img_path).convert("RGB")

    # Get your transform
    transform = get_transforms()

    # Apply the transform
    input_tensor = transform(image)

    # Add a batch dimension (since model expects a batch)
    input_tensor = input_tensor.unsqueeze(0)  # shape: (1, C, H, W)

