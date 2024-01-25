import numpy as np

from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

# initialize the model

# model_path = "Phind/Phind-CodeLlama-34B-v2"
# model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name: str = "roberta-large"
# model_name: str = 'Phind/Phind-CodeLlama-34B-v2'


def answer_MCQ(prompt: str, options: list[str]) -> str:
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    QA_context: str = "\n".join(options)
    QA_input = {
        "question": prompt,
        "context": f"Choose one of the following: {QA_context}",
    }
    res = nlp(QA_input)
    return res


if __name__ == "__main__":
    data = np.load("data/WP-train.npy", allow_pickle=True)
    for i, d in enumerate(data):
        if i < 3:
            test_prompt: str = (
                f"Answer the following brainteaser. Only choose one of the answers listed:"
                f"Question: {data[i]['question']}"
            )  # f"Answers: {data[i]['choice_list']}"
            # print(data[0]['question'])
            # print('\n'.join(data[0]['choice_list']))
            # data = np.load('data/WP_eval_data_for_practice.npy', allow_pickle=True)
            # print(data[0])
            # print(generate_one_completion())
            answer: str = answer_MCQ(prompt=test_prompt, options=data[i]["choice_list"])
            print(
                f"Q: {data[i]['question']}\nOptions: {data[i]['choice_list']}\nCorrect: {data[i]['answer']}\nA: {answer['answer'].strip()}\n====="
            )
        else:
            break
