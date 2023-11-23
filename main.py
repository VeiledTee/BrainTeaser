import numpy as np

from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

# initialize the model

# model_path = "Phind/Phind-CodeLlama-34B-v2"
# model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name: str = 'roberta-large'
# model_name: str = 'Phind/Phind-CodeLlama-34B-v2'


def answer_MCQ(prompt: str, options: list[str]) -> str:
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_context: str = '\n'.join(options)
    QA_input = {
        'question': prompt,
        'context': f"Choose one of the following: {QA_context}"
    }
    res = nlp(QA_input)
    return res


if __name__ == '__main__':
    train_data = np.load('data/WP-train.npy', allow_pickle=True)
    for i, d in enumerate(train_data):
        if i < 3:
            test_prompt: str = f"Answer the following brainteaser. Only choose one of the answers listed:" \
                               f"Question: {train_data[i]['question']}" \
                               # f"Answers: {data[i]['choice_list']}"
            print(train_data[i]['question'])
            print('\n'.join(train_data[i]['choice_list']))
            valid_data = np.load('data/WP_eval_data_for_practice.npy', allow_pickle=True)
            # print(valid_data[i]['question'])
            # print('\n'.join(valid_data[i]['choice_list']))
            answer: str = answer_MCQ(prompt=test_prompt, options=train_data[i]['choice_list'])
            print(f"Q: {train_data[i]['question']}\nOptions: {train_data[i]['choice_list']}\nCorrect: {train_data[i]['answer']}\nA: {answer['answer'].strip()}\n=====")
        else:
            break
