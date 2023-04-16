import os
from src.util import cd_to_executing_file
from src.dataset import MSMarcoDataset, InferenceLLM, LLMClassifierDataset


def generate_dataset(train: bool, short_prompts: bool, llm: InferenceLLM, db_path: str) -> None:
    """
    Generates the dataset, with the specified parameters. This will result in a database file being written to
    database_folder

    Args:
        train: If True, uses ms_marco_train. If False, uses ms_marco_test.
        short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                       will be excluded. This will remove the "no answer" instruction from the prompt as well.
        llm: The llm we will be using to do inference on our prompts.
        db_path: The path we will be writing our database to.
    """
    # Assume we are in the /src directory.
    ms_marco_train = '../../data/MS_MARCO/train_v2.1.json'
    ms_marco_test = '../../data/MS_MARCO/dev_v2.1.json'
    # Deciding which ms_marco dataset to use based on train vs. test.
    ms_marco_path = ms_marco_train if train else ms_marco_test
    # Setting up the MSMarcoDataset and LLMClassifierDataset objects.
    dataset = LLMClassifierDataset(db_path=db_path)
    ms_marco = MSMarcoDataset(ms_marco_path)
    # Creating the database.
    dataset.create_database(ms_marco_dataset=ms_marco, llm=llm, short_prompts=short_prompts)


def generate_datasets_for_llm(llm: InferenceLLM, db_folder: str) -> None:
    """
    Generates 4 datasets:
    - Training data with full-length prompts.
    - Testing data with full-length prompts.
    - Training data with short prompts.
    - Testing data with short prompts.

    Args:
        llm: The LLM we will use to answer the prompts.
        db_folder: The folder we will be writing our databases to.
    """
    generate_dataset(train=True, short_prompts=False, llm=llm,
                     db_path=os.path.join(db_folder, 'train_full_prompts.sqlite3'))
    generate_dataset(train=False, short_prompts=False, llm=llm,
                     db_path=os.path.join(db_folder, 'test_full_prompts.sqlite3'))
    generate_dataset(train=True, short_prompts=True, llm=llm,
                     db_path=os.path.join(db_folder, 'train_short_prompts.sqlite3'))
    generate_dataset(train=False, short_prompts=True, llm=llm,
                     db_path=os.path.join(db_folder, 'test_short_prompts.sqlite3'))
    

def generate_datasets():
    """
    Generates a dataset for each inference llm:

    """
    generate_datasets_for_llm(InferenceLLM('facebook/opt-1.3b'), '../../data/opt_1_3B')
    generate_datasets_for_llm(InferenceLLM('bigscience/bloom-1b1'), '../../data/bloom_1_1B')
    generate_datasets_for_llm(InferenceLLM('EleutherAI/gpt-neo-1.3B'), '../../data/gpt_neo_1_3B')


if __name__ == '__main__':
    cd_to_executing_file(__file__)
    generate_datasets()
