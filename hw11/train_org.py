import torch
from torch import optim, nn
from tqdm import tqdm

from model import Classifier
from train_valid_loader import get_train_valid_loader

# source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
# source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
source_dataloader, valid_source_dataloader = get_train_valid_loader('real_or_drawing/train_data', 32, 1000)

model = Classifier().cuda()

class_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

current_best = 100.0

for e in tqdm(range(0, 500)):
    model.train()
    total_train_loss = 0
    for i, (source_data, source_label) in enumerate(source_dataloader):
        source_data = source_data.repeat(1, 3, 1, 1).cuda()
        source_label = source_label.cuda()

        output = model(source_data)

        loss = class_criterion(output, source_label)
        total_train_loss += loss

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    current_train_loss = total_train_loss / i
    print(f"train loss avg:{current_train_loss}")

    model.eval()
    total_val_loss = 0
    for i, (valid_data, valid_label) in enumerate(valid_source_dataloader):
        valid_data = valid_data.repeat(1, 3, 1, 1).cuda()
        valid_label = valid_label.cuda()

        output = model(valid_data)

        loss = class_criterion(output, valid_label)
        total_val_loss += loss

    current_val_loss = total_val_loss / i
    print(f"val loss avg:{current_val_loss}")

    if current_val_loss < current_best:
        print("save best")
        torch.save(model.state_dict(), f'ckpt/best.bin')
        current_best = current_val_loss

torch.save(model.state_dict(), f'ckpt/last.bin')
