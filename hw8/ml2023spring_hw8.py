
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from qqdm import qqdm, format_str
from dataset import CustomTensorDataset
from model import fcn_autoencoder, conv_autoencoder, VAE, loss_vae, resnet_autoencoder, custom_classifier, \
    resnet_noise_autoencoder

"""# Loading data"""

train = np.load('data/trainingset.npy', allow_pickle=True)

print(train.shape)

"""## Random seed
Set the random seed to a certain value for reproducibility.
"""

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(48763)

"""# Training

## Configuration
"""

# Training hyperparameters
num_epochs = 250
batch_size = 128 # Hint: batch size may be lower
learning_rate = 1e-4

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'n_res'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE(), 'resnet': resnet_autoencoder(), 'n_res': resnet_noise_autoencoder()}
model = model_classes[model_type].cuda()
# classifier = custom_classifier(64 * 64 * 3).cuda()

# Loss and optimizer
criterion = nn.MSELoss()
# classifier_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""## Training loop"""

best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    tot_loss = list()
    # tot_normal_loss = list()
    # tot_noise_loss = list()
    # tot_cls_loss = list()
    for data in train_dataloader:

        # ===================loading=====================
        img = data.float().cuda()
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)

        mask = torch.bernoulli(torch.ones_like(img) * 0.5)
        masked_img = img * mask

        # noise = torch.randn_like(img)
        # ===================forward=====================
        output = model(masked_img)
        # noise_output = model(torch.add(img, noise))

        if model_type in ['vae']:
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
            # noise_loss = loss_vae(noise_output[0], noise, noise_output[1], noise_output[2], criterion)
        else:
            loss = criterion(output, img)
            # cls_loss = classifier_loss(output[1], output[2])

        # normal = output.view(batch_size, 1, -1)
        # abnormal = noise_output.view(batch_size, 1, -1)

        # labels = torch.zeros(0, 2).cuda()
        # final_output = torch.zeros(0, 2, 64 * 64 * 3).cuda()
        #
        # for i in range(0, batch_size):
        #     normal_one_record = normal[i].view(1, 1, -1)
        #     abnormal_one_record = abnormal[i].view(1, 1, -1)
        #
        #     random_num = random.randint(0, 1)
        #     if random_num == 0:
        #         temp_output = torch.cat((abnormal_one_record, normal_one_record), 1)
        #         temp_label = torch.tensor([[0, 1]]).cuda()
        #     else:
        #         temp_output = torch.cat((normal_one_record, abnormal_one_record), 1)
        #         temp_label = torch.tensor([[1, 0]]).cuda()
        #
        #     labels = torch.cat((labels, temp_label), 0)
        #     final_output = torch.cat((final_output, temp_output), 0)
        #
        # final_output = classifier(final_output.view(batch_size, -1))
        #
        # cls_loss = classifier_loss(final_output, labels)

        # cls_loss = 10000 * cls_loss

        # tot_normal_loss.append(loss.cpu().detach().numpy())
        # tot_noise_loss.append(noise_loss.cpu().detach().numpy())
        # tot_cls_loss.append(cls_loss.cpu().detach().numpy())

        # loss = loss + cls_loss

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)

    # mean_normal_loss = np.mean(tot_normal_loss)
    # mean_noise_loss = np.mean(tot_noise_loss)
    # mean_cls_loss = np.mean(tot_cls_loss)

    # print(f"loss: {mean_normal_loss}, cls_loss: {mean_cls_loss}")

    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
    })
    # ===================save_last========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))
