from src.util import write_fasttext_data, load_fasttext_data, pad_fasttext_data


def main() -> None:
    write_fasttext_data('../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3',
                        '../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3')


if __name__ == '__main__':
    # main()
    import time
    t = time.time()
    x_train, x_test, y_train, y_test = \
        load_fasttext_data('../../data/bloom_1_1B/dev_v2.1_short_prompts_train_fasttext.npy',
                           '../../data/bloom_1_1B/dev_v2.1_short_prompts_test_fasttext.npy',
                           '../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3',
                           '../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3')
    print(f'Loading data took {time.time() - t}s')
    t = time.time()
    x_train, x_test = pad_fasttext_data(x_train, x_test)
    print(f'Padding data took {time.time() - t}s')
