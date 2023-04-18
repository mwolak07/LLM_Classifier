from typing import Dict
import random
import json
from llm_classifier.util import cd_to_executing_file
from llm_classifier.dataset import MSMarcoDataset


def get_5_questions(dataset: MSMarcoDataset) -> Dict[str, str]:
    """
    Gets 5 questions from the dataset. This is the longest prompt, and 4 other random prompts that are not that prompt.
    """
    output = {}
    # Getting the longest prompt in the dataset.
    max_prompt = max([dataset.prompt(i, short=False) for i in range(len(dataset))], key=len)
    output['max_question'] = max_prompt
    # Getting 4 random elements in the dataset that have answers.
    indexes = []
    while len(indexes) <= 4:
        index = random.randint(0, len(dataset) - 1)
        element = dataset[index]
        if len(element.answers) > 0:
            indexes.append(index)
    # Getting the prompts for those 4 elements.
    random_prompts = [dataset.prompt(i) for i in indexes]
    output['random_questions'] = random_prompts
    return output


def write_questions() -> None:
    """
    Writes a set of sample questions to test_questions.json. Takes the longest prompt from each dataset, followed by 4
    other random prompts from each dataset. The longest prompt overall get put under 'max_question', and the 8 combined
    random samples go under 'random_questions'.
    """
    # Assume we are running from the /test directory.
    train_file = '../../data/MS_MARCO/train_v2.1.json'
    test_file = '../../data/MS_MARCO/dev_v2.1.json'
    output_file = 'mock_question_data.json'
    output = {}

    # Getting the prompts from the training set.
    train_dataset = MSMarcoDataset(train_file)
    train_output = get_5_questions(train_dataset)

    # Getting the prompts from the testing set.
    test_dataset = MSMarcoDataset(test_file)
    test_output = get_5_questions(test_dataset)

    # Constructing the JSON output.
    max_prompt = max([train_output['max_question'], test_output['max_question']], key=len)
    random_prompts = train_output['random_questions'] + test_output['random_questions']
    output['max_question'] = max_prompt
    output['random_questions'] = random_prompts

    # Writing the output list to the output file.
    with open(output_file, 'w+') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    cd_to_executing_file(__file__)
    write_questions()
