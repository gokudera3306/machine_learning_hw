import numpy as np
import pandas as pd

from nwd import NuclearWassersteinDiscrepancy
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import Classifier
from transforms import source_transform, target_transform

target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

model = Classifier()
model.load_state_dict(torch.load(f'ckpt_resnet50/uda_60.bin'))

model = model.cuda()

model.eval()

result = []
for i, (test_data, _) in (enumerate(test_dataloader)):
    test_data = test_data.repeat(1, 3, 1, 1).cuda()

    class_logits = model(test_data)

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('DALN_submission.csv',index=False)

