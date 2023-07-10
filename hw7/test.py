import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

from dataset import QA_Dataset
from utils import read_data, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"


model = AutoModelForQuestionAnswering.from_pretrained("./saved_model_20/", local_files_only=True).to(device)
model_1 = AutoModelForQuestionAnswering.from_pretrained("./c/", local_files_only=True).to(device)
model_2 = AutoModelForQuestionAnswering.from_pretrained("./b/", local_files_only=True).to(device)
# model_3 = AutoModelForQuestionAnswering.from_pretrained("./saved_model_26/", local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

test_questions, test_paragraphs = read_data("./data/hw7_test.json")
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                     add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# Change "fp16_training" to True to support automatic mixed
# precision training (fp16)
fp16_training = True
if fp16_training:
    accelerator = Accelerator(mixed_precision="fp16")
else:
    accelerator = Accelerator()

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
# model, test_loader = accelerator.prepare(model, test_loader)
model, model_1, model_2, test_loader = accelerator.prepare(model, model_1, model_2, test_loader)

print("Evaluating Test Set ...")

result = []

model.eval()
model_1.eval()
model_2.eval()
# model_3.eval()

with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        output_1 = model_1(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
        output_2 = model_2(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
        # output_3 = model_3(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
        #                    attention_mask=data[2].squeeze(dim=0).to(device))

        output.start_logits = torch.add(output.start_logits, output_1.start_logits)
        output.end_logits = torch.add(output.end_logits, output_1.end_logits)

        output.start_logits = torch.add(output.start_logits, output_2.start_logits)
        output.end_logits = torch.add(output.end_logits, output_2.end_logits)

        # output.start_logits = torch.add(nn.functional.normalize(output.start_logits), nn.functional.normalize(output_1.start_logits))
        # output.end_logits = torch.add(nn.functional.normalize(output.end_logits), nn.functional.normalize(output_1.end_logits))
        #
        # output.start_logits = torch.add(output.start_logits, nn.functional.normalize(output_2.start_logits))
        # output.end_logits = torch.add(output.end_logits, nn.functional.normalize(output_2.end_logits))

        # output.start_logits = torch.add(output.start_logits, output_3.start_logits)
        # output.end_logits = torch.add(output.end_logits, output_3.end_logits)

        result.append(evaluate(tokenizer, data, output, test_paragraphs_tokenized, test_paragraphs))

result_file = "result.csv"
with open(result_file, 'w') as f:
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")

