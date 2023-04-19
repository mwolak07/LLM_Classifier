import os
from src.dataset import MSMarcoDataset, InferenceLLM, LLMClassifierDataset
from src.util import cd_to_executing_file


def generate_dataset(ms_marco_path: str, short_prompts: bool, llm: InferenceLLM, batch_size: int, db_path: str) -> None:
    """
    Generates the dataset, with the specified parameters. This will result in a database file being written to
    database_folder

    Args:
        ms_marco_path: The path to the MSMarcoDataset to generate from.
        short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                       will be excluded. This will remove the "no answer" instruction from the prompt as well.
        llm: The llm we will be using to do inference on our prompts.
        batch_size: The batch size to use with the LLM when running inference.
        db_path: The path we will be writing our database to.
    """
    # Setting up the MSMarcoDataset and LLMClassifierDataset objects.
    dataset = LLMClassifierDataset(db_path=db_path)
    ms_marco = MSMarcoDataset(ms_marco_path)
    # Creating the database.
    dataset.create_database(ms_marco_dataset=ms_marco, llm=llm, short_prompts=short_prompts, batch_size=batch_size)


def generate_datasets_for_llm(llm: InferenceLLM, db_folder: str, batch_size: int = 1,
                              ms_marco_test: str = '../../data/MS_MARCO/dev_v2.1.json',
                              ms_marco_train: str = '../../data/MS_MARCO/train_v2.1.json') -> None:
    """
    Generates 2 datasets:
    - Testing data with short prompts.
    - Training data with short prompts.

    Args:
        llm: The LLM we will use to answer the prompts.
        db_folder: The folder we will be writing our databases to.
        batch_size: The batch size to use with the LLM when running inference.
        ms_marco_test: The path to the MS MARCO test set.
        ms_marco_train: The path to the MS MARCO train set.
    """
    cd_to_executing_file(__file__)
    print('Generating testing set...')
    generate_dataset(ms_marco_test, short_prompts=True, llm=llm, batch_size=batch_size,
                     db_path=os.path.join(db_folder, 'test_short_prompts.sqlite3'))
    print('Done')
    print('Generating training set...')
    generate_dataset(ms_marco_train, short_prompts=True, llm=llm, batch_size=batch_size,
                     db_path=os.path.join(db_folder, 'train_short_prompts.sqlite3'))
    print('Done')
    

def generate_datasets():
    """
    Generates a dataset for each inference llm.
    """
    # generate_datasets_for_llm(InferenceLLM('facebook/opt-1.3b'), '../../data/opt_1_3B')
    generate_datasets_for_llm(InferenceLLM('bigscience/bloom-1b1'), '../../data/bloom_1_1B', batch_size=64)


if __name__ == '__main__':
    generate_datasets()
