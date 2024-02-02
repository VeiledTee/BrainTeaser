import numpy as np


def load_dataset(file_name: str) -> list[dict[str, str]]:
    data = np.load(file_name, allow_pickle=True)
    loaded_data = [
        {"question": record["question"], "choice_list": record["choice_list"]}
        for record in data
    ]
    return loaded_data


if __name__ == "__main__":
    sentence_train = load_dataset("data/SP-train.npy")
    sentence_eval = load_dataset("data/SP_eval_data_for_practice.npy")
    sentence_test = load_dataset("data/SP_new_test.npy")
    word_train = load_dataset("data/WP-train.npy")
    word_eval = load_dataset("data/WP_eval_data_for_practice.npy")
    word_test = load_dataset("data/WP_new_test.npy")

    print(sentence_train[0])
    print(sentence_eval[0])
    print(len(sentence_test))

    print(word_train[0])
    print(word_eval[0])
    print(word_test[0])
