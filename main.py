import numpy as np

from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

# initialize the model

# model_path = "Phind/Phind-CodeLlama-34B-v2"
# model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# HumanEval helper

def generate_one_completion(prompt: str):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=384, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion


# run `evaluate_functional_correctness samples.jsonl` in your HumanEval code sandbox

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = 'roberta-large'


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
    data = np.load('data/WP-train.npy', allow_pickle=True)
    for i, d in enumerate(data):
        if i < 3:
            test_prompt: str = f"Answer the following brainteaser. Only choose one of the answers listed:" \
                               f"Question: {data[i]['question']}" \
                               # f"Answers: {data[i]['choice_list']}"
            # print(data[0]['question'])
            # print('\n'.join(data[0]['choice_list']))
            # data = np.load('data/WP_eval_data_for_practice.npy', allow_pickle=True)
            # print(data[0])
            # print(generate_one_completion())
            answer: str = answer_MCQ(prompt=test_prompt, options=data[i]['choice_list'])
            print(f"Q: {data[i]['question']}\nOptions: {data[i]['choice_list']}\nCorrect: {data[i]['answer']}\nA: {answer['answer'].strip()}\n=====")
        else:
            break
