import numpy as np
import torch
import os
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random

from config import cfg, save_path
from dataset import FoodDataset, train_tfm, test_tfm

myseed = cfg['seed']  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

os.makedirs(save_path, exist_ok=True)

valid_set = FoodDataset(os.path.join(cfg['dataset_root'], "validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

teacher_ckpt_path = os.path.join(cfg['dataset_root'], "resnet18_teacher.ckpt")
device = "cuda" if torch.cuda.is_available() else "cpu"

for i in range(20):
    teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)
    teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu'))
    teacher_model.to(device)  # MEDIUM BASELINE
    teacher_model.eval()  # MEDIUM BASELINE

    ratio = 0.05 * i  # specify the pruning ratio
    for name, module in teacher_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):  # if the nn.module is torch.nn.Conv2d
            prune.l1_unstructured(module, name='weight', amount=ratio)

    valid_accs = []
    valid_lens = []
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(imgs) # MEDIUM BASELINE

        acc = (teacher_logits.argmax(dim=-1) == labels).float().sum()

        batch_len = len(imgs)
        valid_accs.append(acc)
        valid_lens.append(batch_len)

    valid_acc = sum(valid_accs) / sum(valid_lens)
    print(f"prune ratio: {ratio}, acc: {valid_acc}")
