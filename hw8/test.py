import PIL
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
import pandas as pd
from matplotlib import pyplot as plt

from dataset import CustomTensorDataset

"""# Inference
Model is loaded and generates its anomaly score predictions.

## Initialize
- dataloader
- model
- prediction file
"""


def tensor_to_image(tensor):
    n = (tensor.cpu() + 1) * 127.5
    n = n.permute(1, 2, 0)
    n = np.array(n, dtype=np.uint8)
    return PIL.Image.fromarray(n)


model_type = 'n_res'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}

test = np.load('data/testingset.npy', allow_pickle=True)
print(test.shape)

eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
# checkpoint_path = f'best_model_{model_type}.pt'
checkpoint_path = f'resnet34_mae_64_78748.pt'
model = torch.load(checkpoint_path)

# model.is_test = True

model.eval()

# prediction file
out_file = 'prediction.csv'

anomality = list()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        img = data.float().cuda()

        # img_show = tensor_to_image(img[0])
        # imgplot = plt.imshow(img_show)
        # plt.show()

        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)

        mask = torch.bernoulli(torch.ones_like(img) * 0.95)
        masked_img = img * mask

        output = model(masked_img)
        if model_type in ['vae']:
            output = output[0]
        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)

        # img_show = tensor_to_image(output.view(-1, 3, 64, 64).squeeze())
        # imgplot = plt.imshow(img_show)
        # plt.show()

anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')

