from nwd import NuclearWassersteinDiscrepancy
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import Classifier
from transforms import source_transform, target_transform

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True, drop_last=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True, drop_last=True)

model = Classifier()
model.load_state_dict(torch.load(f'ckpt/best.bin'))

discrepancy = NuclearWassersteinDiscrepancy(model.backbone.fc)

model = model.cuda()
discrepancy = discrepancy.cuda()

class_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in tqdm(range(0, 110)):
    total_loss = 0
    total_cls_loss = 0
    total_d_loss = 0
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.repeat(1, 3, 1, 1).cuda()
        target_data = target_data.repeat(1, 3, 1, 1).cuda()
        source_label = source_label.cuda()

        merged_input = torch.cat((source_data, target_data), dim=0)
        merged_output = model(merged_input)
        source_output, _ = merged_output.chunk(2, dim=0)

        cls_loss = class_criterion(source_output, source_label)
        total_cls_loss += cls_loss

        discrepancy_loss = -discrepancy(model.features[0])
        total_d_loss += discrepancy_loss

        transfer_loss = discrepancy_loss * 1 # multiply the lambda to trade off the loss term
        loss = cls_loss + transfer_loss
        total_loss += loss

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    current_loss = total_loss / i
    current_cls_loss = total_cls_loss / i
    current_d_loss = total_d_loss / i
    print(f"loss avg:{current_loss}, cls loss avg:{current_cls_loss}, d loss:{current_d_loss}")
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'ckpt/uda_{epoch}.bin')

torch.save(model.state_dict(), f'ckpt/uda_last.bin')



