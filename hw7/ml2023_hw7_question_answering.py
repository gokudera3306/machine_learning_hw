import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from accelerate import Accelerator

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_scheduler
)

from dataset import QA_Dataset
from utils import read_data, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"


# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(36)

model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")


"""## Read Data

- Training set: 26918 QA pairs
- Dev set: 2863  QA pairs
- Test set: 3524  QA pairs

- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
"""


train_questions, train_paragraphs = read_data("./data/hw7_train.json")
dev_questions, dev_paragraphs = read_data("./data/hw7_dev.json")

"""## Tokenize Data"""

# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                      add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions],
                                    add_special_tokens=False)

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)

# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model


train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)


num_epoch = 30
validation = True
logging_step = 1000
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
train_batch_size = 4
model_save_dir = "saved_model"


#### TODO: gradient_accumulation (optional)####
# Note: train_batch_size * gradient_accumulation_steps = effective batch size
# If CUDA out of memory, you can make train_batch_size lower and gradient_accumulation_steps upper
# Doc: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
gradient_accumulation_steps = 4



# dataloader
# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)

# Change "fp16_training" to True to support automatic mixed
# precision training (fp16)	
fp16_training = True
if fp16_training:
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
else:
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_loader) / gradient_accumulation_steps) * num_epoch
)

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
model, optimizer, train_loader, dev_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, dev_loader, lr_scheduler)

model.train()

print("Start Training ...")
best_acc = 0.0

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0

    for data in tqdm(train_loader):
        with accelerator.accumulate(model):
            # Load all data into GPU
            data = [i.to(device) for i in data]

            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3],
                           end_positions=data[4])
            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()

            train_loss += output.loss

            accelerator.backward(output.loss)

            step += 1
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            ##### TODO: Apply linear learning rate decay #####

            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device),
                               token_type_ids=data[1].squeeze(dim=0).to(device),
                               attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                dev_acc += evaluate(tokenizer, data, output, dev_paragraphs_tokenized, dev_paragraphs) == dev_questions[i]["answer_text"]

            current_acc = dev_acc / len(dev_loader)
            print(f"Validation | Epoch {epoch + 1} | acc = {current_acc:.3f}")

            if current_acc > best_acc and epoch > 10:
                print("Saving best Model ...")
                model.save_pretrained(f"{model_save_dir}_{epoch}")
                best_acc = current_acc

        model.train()

# Save a model and its configuration file to the directory 「saved_model」 
# i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
print("Saving Model ...")
model.save_pretrained(model_save_dir)
