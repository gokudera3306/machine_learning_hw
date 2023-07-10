import json

import torch


def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


def evaluate(tokenizer, data, output, tokenized_paragraph, original_paragraph):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    # if '洛杉磯加利福尼亞大學的對抗是' in original_paragraph[data[3]]:
    #     print("Hi")

    for k in range(num_of_windows):

        para_start = data[4][k]
        para_end = data[5][k]

        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k][para_start:para_end], dim=0)
        start_index = start_index + para_start

        end_prob, end_index = torch.max(output.end_logits[k][start_index:para_end], dim=0)
        end_index = end_index + start_index

        # print(start_index)
        # print(end_index)

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])

            if '[UNK]' in answer:
                raw_start = tokenized_paragraph[data[3]].token_to_chars(start_index - data[4][k] + data[6][k])[0]
                raw_end = tokenized_paragraph[data[3]].token_to_chars(end_index - data[4][k] + data[6][k])[1]
                answer = original_paragraph[data[3]][raw_start:raw_end]

    # if answer == '' or answer == '[SEP]':
    #     print("no answer")
    #     for k in range(num_of_windows):
    #         para_start = data[4][k]
    #         para_end = data[5][k]
    #
    #         # Obtain answer by choosing the most probable start position / end position
    #         start_prob, start_index = torch.max(output.start_logits[k][para_start:para_end], dim=0)
    #         end_prob, end_index = torch.max(output.end_logits[k][para_start:para_end], dim=0)
    #
    #         start_index = start_index + para_start
    #         end_index = end_index + para_start
    #
    #         # Probability of answer is calculated as sum of start_prob and end_prob
    #         prob = start_prob + end_prob
    #         print(tokenizer.decode(data[0][0][k][start_index: end_index + 1]))
    #         print(prob)

    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ', '')


