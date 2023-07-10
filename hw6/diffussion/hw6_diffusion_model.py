import torch

from model import Unet, GaussianDiffusion
from trainer import Trainer

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)

if torch.cuda.is_available():
  torch.cuda.manual_seed(4096)


"""# Training Hyper-parameters"""

path = '../faces'
IMG_SIZE = 64             # Size of images, do not change this if you do not know why you need to change
batch_size = 64
train_num_steps = 100000        # total training steps
lr = 1e-3
grad_steps = 1            # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = 0.995           # exponential moving average decay

channels = 32             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4)        # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)

timesteps = 1000            # Number of steps (adding noise)
beta_schedule = 'linear'

model = Unet(
    dim = channels,
    dim_mults = dim_mults
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    timesteps = timesteps,
    beta_schedule = beta_schedule
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = batch_size,
    train_lr = lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = grad_steps,
    ema_decay = ema_decay,
    save_and_sample_every = 1000
)

trainer.train()
