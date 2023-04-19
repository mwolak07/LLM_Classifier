from typing import Dict, Any, List, Tuple, Union
import random
import json
from src.util import cd_to_executing_file
from src.dataset import MSMarcoDataset, MSMarcoItem, LLMClassifierDataset


def get_mock_data_elements(dataset: MSMarcoDataset) -> Dict[str, Any]:
    """
    Gets target elements from the dataset. These are:
    - The longest long prompt
    - The longest short prompt
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

    # Getting the element with the longest long prompt.
    max_long_prompt = max([LLMClassifierDataset.prompt(element.passages, element.query) for element in dataset],
                          key=len)
    output['long_max'] = max_long_prompt

    # Getting the element with the longest short prompt.
    max_short_prompt = max([LLMClassifierDataset.prompt(element.chosen_passages, element.query) for element in dataset],
                           key=len)
    output['short_max'] = max_short_prompt

    # Getting a random element with no answer.
    no_answer_elements = [element for element in dataset if len(element.answers) == 0]
    no_answer_index = random.choice(range(len(no_answer_elements)))
    output['no_answer'] = no_answer_elements[no_answer_index]
    selected_indexes.append(no_answer_index)

    # Getting a random element with more than one answer.
    many_answer_elements = [element for element in dataset if len(element.answers) > 1]
    many_answer_index = random.choice(range(len(many_answer_elements)))
    output['many_answer'] = many_answer_elements[many_answer_index]
    selected_indexes.append(many_answer_index)

    # Getting 3 random elements in the dataset that have answers, and have not already been picked.
    remaining_elements = [dataset[i] for i in range(len(dataset))
                          if i not in selected_indexes and len(dataset[i].answers) >= 1]
    output['random'] = random.sample(remaining_elements, 3)

    return output


def get_mock_ms_marco_dataset(no_answer_elements: List[MSMarcoItem],
                              many_answer_elements: List[MSMarcoItem],
                              random_elements: List[MSMarcoItem]) -> Dict[str, Any]:
    """
    Creates a MS MARCO dataset dict using the target elements from the real sets.
    Elements are put into the dataset as follows:
    - {'query': {'<index>': element.query}}
    - {'query_type': {'<index>': element.query_type}}
    - {'query_id': {'<index>': <index>}}
    - {'passages': {'<index>': [{'is_selected': 1 or 0,
                                 'passage_text': element.passages or element.chosen_passages,
                                 'url': 'no_url'}]}}
    - {'answers': {'<index>': element.answers}}
    - {'wellFormedAnswers': {'<index>': <empty wellFormedAnswer>}}
    Empty answers are indicated as follows:
    - answers: 'No Answer Present.'
    - wellFormedAnswers: '[]'
    The following additional modifications are made:
    - Half of the random elements will have their answers be wellFormedAnswers, with their answer as 'garbage'.
    - The other half of the random elements will have their answers be answers, with no wellFormedAnswers.

    Args:
        no_answer_elements: Two elements with no answers.
        many_answer_elements: Two elements with multiple answers.
        random_elements: Six other random elements.

    Returns:
        A dict containing the provided data formatted the same way as
    """
    # Initializing our output
    output = {'query': {}, 'query_type': {}, 'query_id': {}, 'passages': {}, 'answers': {}, 'wellFormedAnswers': {}}

    # Inserting the no answer elements.
    output = _add_element_to_output(0, no_answer_elements[0], output, False)
    output = _add_element_to_output(1, no_answer_elements[1], output, True)

    # Inserting one many answer element with wellFormedAnswers, the other without.
    output = _add_element_to_output(2, many_answer_elements[0], output, True)
    output = _add_element_to_output(3, many_answer_elements[1], output, False)

    # Inserting the first half of the random elements without wellFormedAnswers.
    split_point = len(random_elements) // 2
    i = 4
    for element in random_elements[:split_point]:
        output = _add_element_to_output(i, element, output, False)
        i += 1

    # Inserting the second half of the random elements with wellFormedAnswers.
    for element in random_elements[:split_point]:
        output = _add_element_to_output(i, element, output, True)
        i += 1

    return output


def _add_element_to_output(index: int, element: MSMarcoItem, output: Dict[str, Any], well_formed: bool) \
        -> Dict[str, Any]:
    """
    Adds the MSMarcoItem to the output dict. If well_formed is True, the answers go in wellFormedAnswers with "garbage"
    in answers for each answer.

    Args:
        index: The index of the element in the output.
        element: The MSMarcoItem to add to the output.
        output: The output MS MARCO dataset dict.

    Returns:
        The output dict, with the element added.
    """
    output['query'][str(index)] = element.query
    output['query_type'][str(index)] = element.query_type.value
    output['query_id'][str(index)] = index
    output['passages'][str(index)] = _get_ms_marco_passages(element)
    answers, well_formed_answers = _get_ms_marco_answers(element, well_formed)
    output['answers'][str(index)] = answers
    output['wellFormedAnswers'][str(index)] = well_formed_answers
    return output


def _get_ms_marco_answers(element: MSMarcoItem, well_formed: bool) -> Tuple[List[str], Union[str, List[str]]]:
    """
    Gets the answers in the MS marco format from the element's answers.
    - Converts empty to the correct value for answer or wellFormedAnswer.
    - Sets answer to ["garbage"] and wellFormedAnswer to the answer if well_formed is True.
    - Sets answer to the answer and wellFormedAnswer to '[]' if well_formed is False.

    Args:
        element: The MSMarcoItem to get the answers for.
        well_formed: Are the answers for the element supposed to be wellFormedAnswers?

    Returns:
        answers, wellFormedAnswers in the MS Marco format.
    """
    if len(element.answers) == 0:
        answers = MSMarcoDataset.answer_empty
        well_formed_answers = MSMarcoDataset.well_formed_answer_empty
    elif well_formed:
        answers = ['garbage'] * len(element.answers)
        well_formed_answers = element.answers
    else:
        answers = element.answers
        well_formed_answers = MSMarcoDataset.well_formed_answer_empty
    return answers, well_formed_answers


def _get_ms_marco_passages(element: MSMarcoItem) -> List[Dict[str, Any]]:
    """
    Gets the passages in the MS MARCO format from the MSMarcoItem.

    Args:
        element: The MSMarcoItem to get the passages list from.

    Returns:
        A list of MS MARCO formatted passage objects.
    """
    output = []
    # Getting the distinct classes of passages.
    chosen_passages = element.chosen_passages
    other_passages = [passage for passage in element.passages if passage not in chosen_passages]
    # Converting the chosen passages.
    for passage in chosen_passages:
        output.append({'is_selected': 1, 'passage_text': passage, 'url': 'not_a_url'})
    # Converting the other passages.
    for passage in other_passages:
        output.append({'is_selected': 0, 'passage_text': passage, 'url': 'not_a_url'})
    return output


def write_mock_ms_marco_data() -> None:
    """
    Writes a mock MS MARCO dataset for testing. Takes the elements as defined in get_dataset_elements and converts them
    to an MS MARCO dataset format as defined in get_mock_ms_marco_dataset.
    """
    # Assume we are running from the /test directory.
    train_file = '../../data/MS_MARCO/train_v2.1.json'
    test_file = '../../data/MS_MARCO/dev_v2.1.json'
    mock_dataset_file = 'mock_data_ms_marco.json'
    max_prompt_file = 'mock_data_max_prompts.json'

    # Getting the elements from the testing set.
    print('Reading the testing set...')
    test_dataset = MSMarcoDataset(test_file)
    test_elements = get_mock_data_elements(test_dataset)
    print('Done')

    # Getting the elements from the training set.
    print('Reading the training set...')
    train_dataset = MSMarcoDataset(train_file)
    train_elements = get_mock_data_elements(train_dataset)
    print('Done')

    print('Generating the mock sets...')

    # Getting the mock dataset elements.
    no_answer_elements = [train_elements['no_answer'], test_elements['no_answer']]
    many_answer_elements = [train_elements['many_answer'], test_elements['many_answer']]
    random_elements = train_elements['random'] + test_elements['random']

    # Building our MS MARCO dataset dict.
    mock_dataset = get_mock_ms_marco_dataset(no_answer_elements, many_answer_elements, random_elements)

    # Writing the mock dataset dict to the mock dataset file.
    with open(mock_dataset_file, 'w+') as f:
        json.dump(mock_dataset, f, indent=4)
    print('Done')

    # Writing our max prompts to the max prompts file.
    max_data = {
        'max_long_prompt': max(test_elements['long_max'], train_elements['long_max'], key=len),
        'max_short_prompt': max(test_elements['short_max'], train_elements['short_max'], key=len)
    }
    with open(max_prompt_file, 'w+') as f:
        json.dump(max_data, f, indent=4)


if __name__ == '__main__':
    cd_to_executing_file(__file__)
    write_mock_ms_marco_data()
