import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from model import FoodDataset, test_tfm, Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
batch_size = 256

tta_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomResizedCrop((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-45, +45)),
    transforms.ToTensor(),
])

# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("./test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

tta_set = FoodDataset("./test", tfm=tta_tfm)
tta_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""# Testing and generate prediction CSV"""

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"./ckpt/83340.ckpt"))
model_best.eval()

model_second = Classifier().to(device)
model_second.load_state_dict(torch.load(f"./ckpt/83228.ckpt"))
model_second.eval()

model_third = Classifier().to(device)
model_third.load_state_dict(torch.load(f"./ckpt/83211.ckpt"))
model_third.eval()

model_fourth = Classifier().to(device)
model_fourth.load_state_dict(torch.load(f"./ckpt/82543.ckpt"))
model_fourth.eval()

model_fifth = Classifier().to(device)
model_fifth.load_state_dict(torch.load(f"./ckpt/82481.ckpt"))
model_fifth.eval()


prediction = []
temp_pred = np.zeros((3000, 11))
with torch.no_grad():
    temp_temp_pred = np.empty((0, 11))
    for data, _ in tqdm(test_loader):
        test_pred_best = model_best(data.to(device))
        test_pred_second = model_second(data.to(device))
        test_pred_third = model_third(data.to(device))
        test_pred_fourth = model_fourth(data.to(device))
        test_pred_fifth = model_fifth(data.to(device))
        p = test_pred_best.cpu().data.numpy() + test_pred_second.cpu().data.numpy() + test_pred_third.cpu().data.numpy() + test_pred_fourth.cpu().data.numpy() + test_pred_fifth.cpu().data.numpy()
        temp_temp_pred = np.concatenate((temp_temp_pred, p), axis=0)
    temp_pred = np.add(temp_pred, temp_temp_pred)

    temp_pred = np.exp(temp_pred) / np.sum(np.exp(temp_pred), axis=1, keepdims=True)

    tta_pred = np.zeros((3000, 11))
    for j in range(10):
        temp_temp_pred = np.empty((0, 11))
        for data, _ in tqdm(tta_loader):
            test_pred_best = model_best(data.to(device))
            test_pred_second = model_second(data.to(device))
            test_pred_third = model_third(data.to(device))
            test_pred_fourth = model_fourth(data.to(device))
            test_pred_fifth = model_fifth(data.to(device))
            p = test_pred_best.cpu().data.numpy() + test_pred_second.cpu().data.numpy() + test_pred_third.cpu().data.numpy() + test_pred_fourth.cpu().data.numpy() + test_pred_fifth.cpu().data.numpy()
            temp_temp_pred = np.concatenate((temp_temp_pred, p), axis=0)
        tta_pred = np.add(tta_pred, temp_temp_pred)


    tta_pred = np.exp(tta_pred) / np.sum(np.exp(tta_pred), axis=1, keepdims=True)

    temp_pred = temp_pred * 0.8 + tta_pred * 0.2
    test_label = np.argmax(temp_pred, axis=1)
    prediction += test_label.squeeze().tolist()

# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)

