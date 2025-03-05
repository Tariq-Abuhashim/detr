import torch
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection

def compute_mean_std(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std

# Assuming your dataset is set up for torchvision datasets
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor()
])


# The transforms argument in the CocoDetection dataset expects a callable that receives both an image and its annotations (target) 
# and returns the transformed image and target. However, the provided transform (transforms.ToTensor()) only expects an image.
# we should define a custom transform that can handle both image and its target (annotations). 
# This custom transform should be passed to the CocoDetection dataset.
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, image, target):
        return self.transform(image), target


custom_transform = CustomTransform(transform)

# Replace datasets.ImageFolder with your dataset if it's different
dataset = CocoDetection(root='man_ir/train/', annFile='man_ir/annotations/instances_train.json', transforms=custom_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

mean, std = compute_mean_std(dataloader)
print("Mean:", mean)
print("Standard Deviation:", std)

#Mean: tensor([0.4647, 0.4583, 0.4607])
#Standard Deviation: tensor([0.1642, 0.1628, 0.1632])