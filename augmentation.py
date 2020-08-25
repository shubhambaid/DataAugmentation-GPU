import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import kornia

# I am creating a dummy data here
# You can import your own data and work :)

class SuperDummy(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_index = self.build_index(self.data_root)

    def build_index(self, data_root):
        return range(10)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        sample = self.data_index[idx]

        image = torch.rand(3, 240, 320)
        label = torch.rand(1, 240, 320)
        return dict(images=image, labels=label)


# The beautiful feature of Kornia enhance which allows us put the 
# operations directly inside nn.sequential <3

transform = nn.Sequential(
    kornia.enhance.AdjustBrightness(0.5),
    kornia.enhance.AdjustGamma(gamma=2.),
    kornia.enhance.AdjustContrast(0.7),
)

# NOTE: you can put 'cpu' as well.
# But what's the point since you are referring this
# repo for running augmentation on GPU.

device = torch.device('cuda')
print(f"I a eating up your {device}")

# creating a data loader 
dataset = SuperDummy()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# to get some samples and run augmentation on those
for i_batch, sample_batched in enumerate(dataloader):
    images = sample_batched['images'].to(device)
    labels = sample_batched['labels'].to(device)

    # perform the transforms :)
    images = transform(images)
    labels = transform(labels)

    print(f"Iteration: {i_batch} Image shape: {images.shape}")
