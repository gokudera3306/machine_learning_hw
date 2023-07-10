import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, track
from model import FeatureExtractor, LabelPredictor, DomainClassifier
import pandas as pd

from transforms import target_transform

target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()

feature_extractor.load_state_dict(torch.load(f'extractor_model_1900.bin'))
label_predictor.load_state_dict(torch.load(f'predictor_model_1900.bin'))

result = []
label_predictor.eval()
feature_extractor.eval()
with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn()) as progress:
    test_tqdm = progress.add_task(description="inference progress", total=len(test_dataloader))
    for i, (test_data, _) in (enumerate(test_dataloader)):
        test_data = test_data.cuda()

        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)
        progress.advance(test_tqdm)

result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('DaNN_submission.csv',index=False)
