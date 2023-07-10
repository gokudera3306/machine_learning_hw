from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
import os


# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.

    transforms.RandomResizedCrop((128, 128)),
    # transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-45, +45)),
    transforms.ColorJitter(),

    transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),

    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label

        return im, label


"""# Model"""
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = models.resnet50()
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 11),
        )

    def forward(self, x):
        return self.cnn(x)
