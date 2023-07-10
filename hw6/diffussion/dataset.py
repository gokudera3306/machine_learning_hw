from torchvision import transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size
    ):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]
        #################################
        ## TODO: Data Augmentation ##
        #################################
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
