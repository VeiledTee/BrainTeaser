from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertModel
import torch

from POSTagging import SimplePOSTagger
from dataset_tomfoolery import load_dataset
from sensembert import BrainTeaserWSD


def calculate_probabilities(main_phrase, endings, model, tokenizer):
    main_prompt = main_phrase + " "
    completions = [main_prompt + ending for ending in endings]

    # Add a new pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    input_ids = tokenizer(completions, return_tensors="pt", padding=True, truncation=True).input_ids

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[:, -1, :]
    probabilities = logits.softmax(dim=-1)

    return probabilities


if __name__ == '__main__':
    dataset = load_dataset('data/SP_new_test.npy')
    save_file = 'data/answer_sen.txt'
    # dataset = load_dataset('data/WP_new_test.npy')
    # save_file = 'data/answer_word.txt'

    wsd_comparison = BrainTeaserWSD()
    pos_comparison = SimplePOSTagger()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_model = BertModel.from_pretrained('bert-large-cased')

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    with open(save_file, 'a') as file:
        for i, record in enumerate(dataset):
            print(f"{i} / {len(dataset)}")
            question_text = record['question']
            answer_texts = record['choice_list']

            probabilities = calculate_probabilities(question_text, answer_texts, model, tokenizer)
            print("Probabilities:")
            for i, ending in enumerate(answer_texts):
                # Find the index of the maximum probability
                max_prob_index = torch.argmax(probabilities[i]).item()

                print(f"{ending}: {probabilities[i][max_prob_index].item():.4f}")

            # Find the index of the ending with the maximum probability across all endings
            max_prob_across_endings_index = torch.argmax(probabilities.sum(dim=0)).item()
            max_prob_across_endings = answer_texts[max_prob_across_endings_index]

            print(f"\nSelected Ending with Maximum Probability Across All Endings: {max_prob_across_endings}")

            # file.write(str(max_comparison_index) + '\n')
