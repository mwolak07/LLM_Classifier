from typing import Dict, Any, List
import random
import json
from src.util import cd_to_executing_file
from src.dataset import MSMarcoDataset, MSMarcoItem, LLMClassifierDataset


def get_dataset_elements(dataset: MSMarcoDataset) -> Dict[str: Any]:
    """
    Gets target elements from the dataset. These are:
    - The element with the longest prompt, and its length.
    - A random element with no answer.
    - A random element with multiple answers.
    - 3 random other elements.

    Args:
        dataset: The dataset we are getting the 5 elements for.

    Returns:
        A dict, containing a string identifier for the group of elements, and the target elements in that group.
    """
    output = {}
    selected_indexes = []

    # Getting the element with the longest prompt.
    prompt_lengths = [len(LLMClassifierDataset.prompt(element.passages, element.query)) for element in dataset]
    max_prompt_len = max(prompt_lengths)
    max_prompt_index = prompt_lengths.index(max_prompt_len)
    output['max'] = (dataset[max_prompt_index], max_prompt_index)
    selected_indexes.append(max_prompt_index)

    # Getting a random element with no answer.
    no_answer_elements = [element for element in dataset if len(element.answer) == 0]
    no_answer_index = random.choice(range(len(no_answer_elements)))
    output['no_answer'] = dataset[no_answer_index]
    selected_indexes.append(no_answer_index)

    # Getting a random element with more than one answer.
    many_answer_elements = [element for element in dataset if len(element.answer) > 1]
    many_answer_index = random.choice(range(len(many_answer_elements)))
    output['no_answer'] = dataset[many_answer_index]
    selected_indexes.append(many_answer_index)

    # Getting 3 random elements in the dataset that have answers, and have not already been picked.
    remaining_elements = [dataset[i] for i in range(len(dataset)) if i not in selected_indexes]
    output['random'] = random.sample(remaining_elements, 3)

    return output


def get_mock_ms_marco_dataset(max_element: MSMarcoItem,
                              no_answer_elements: List[MSMarcoItem],
                              many_answer_elements: List[MSMarcoItem],
                              random_elements: List[MSMarcoItem]) -> Dict[str: Any]:
    """
    Creates a MS MARCO dataset dict using the target elements from the real sets. They are manipulated as follows:
    -
    - Half of the random elements will have their answers be wellFormedAnswers, with their answer as 'garbage'.
    - The other half of the random elements will have their answers be answers, with no wellFormedAnswers.
    -
    """


def write_mock_ms_marco_data() -> None:
    """
    Writes a mock MS MARCO dataset for testing.  Takes the element with the longest prompt from each dataset, followed
    by 4 other random elements with answers from each dataset. Then, it constructs a sample MS_MARCO dataset with
    these elements. Half will have their answers go in answers, half in well-formed answers. The ones with well-formed
    answers will have "garbage" as the answer. Next, there will be one element with an empty answer, and one element
    with two answers.
    """
    # Assume we are running from the /test directory.
    train_file = '../../data/MS_MARCO/train_v2.1.json'
    test_file = '../../data/MS_MARCO/dev_v2.1.json'
    output_file = 'mock_question_data.json'
    output = {}

    # Getting the elements from the training set.
    train_dataset = MSMarcoDataset(train_file)
    train_elements = get_dataset_elements(train_dataset)

    # Getting the elements from the testing set.
    test_dataset = MSMarcoDataset(test_file)
    test_elements = get_dataset_elements(test_dataset)

    # Getting the max element by looking at both sets.
    train_max_greater = train_elements['max'][1] > test_elements['max'][1]
    max_element = train_elements['max'] if train_max_greater else test_elements['max'][1]

    # Building our
    max_element = max([train_output['max'], test_output['max']], key=len)
    random_prompts = train_output['random_questions'] + test_output['random_questions']
    output['max_question'] = max_prompt
    output['random_questions'] = random_prompts

    # Writing the output list to the output file.
    with open(output_file, 'w+') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    cd_to_executing_file(__file__)
    write_questions()
