import os
from src.datasets import MSMarcoDataset, InferenceLLM, InferenceGPTNeoX, LLMClassifierDataset


def generate_dataset(train: bool, short_prompts: bool, llm: InferenceLLM, db_file: str) -> None:
    """
    Generates the dataset, with the specified parameters. This will result in a database file being written to
    database_folder

    Args:
        train: If True, uses ms_marco_train. If False, uses ms_marco_test.
        short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                       will be excluded. This will remove the "no answer" instruction from the prompt as well.
        llm: The llm we will be using to do inference on our prompts.
        db_file: The file we will be writing our database to.
    """
    # Storing up the file paths for MS MARCO train/test, and the database folder.
    ms_marco_train = '../../data/MS_MARCO/train_v2.1.json'
    ms_marco_test = '../../data/MS_MARCO/dev_v2.1.json'
    db_folder = '../../data/llm_classifier'
    # Deciding which ms_marco dataset to use based on train vs. test.
    ms_marco_path = ms_marco_train if train else ms_marco_test
    # Making the database file path.
    db_path = os.path.join(db_folder, db_file)
    # Setting up the MSMarcoDataset and LLMClassifierDataset objects.
    dataset = LLMClassifierDataset(db_path=db_path)
    ms_marco = MSMarcoDataset(ms_marco_path)
    # Creating the database.
    dataset.create_database(ms_marco_dataset=ms_marco, llm=llm, short_prompts=short_prompts)


def generate_datasets() -> None:
    """
    Generates 4 datasets:
    - Training data with full-length prompts.
    - Testing data with full-length prompts.
    - Training data with short prompts.
    - Testing data with short prompts.
    """
    llm = InferenceGPTNeoX()
    generate_dataset(train=True, short_prompts=False, llm=llm, db_file='train_full_prompts.sqlite3')
    generate_dataset(train=False, short_prompts=False, llm=llm, db_file='test_full_prompts.sqlite3')
    generate_dataset(train=True, short_prompts=True, llm=llm, db_file='train_short_prompts.sqlite3')
    generate_dataset(train=False, short_prompts=True, llm=llm, db_file='test_short_prompts.sqlite3')


if __name__ == '__main__':
    generate_datasets()
