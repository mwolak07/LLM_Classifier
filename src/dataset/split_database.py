import random
from src.dataset import LLMClassifierDatabase


def split_database(db_path: str) -> None:
    """
    Splits the database at the given db_path into a test_ and train_ database, with 10% randomly inserted into test_
    and 90% randomly inserted into train_.

    Args:
        db_path: The path the the database to be split.
    """
    train_ratio = 0.9
    extension = '.sqlite3'
    # Getting the rows from the database at the db_path and shuffling them randomly.
    print(f'Reading the database...')
    database = LLMClassifierDatabase(db_path)
    rows = database.tolist()
    random.shuffle(rows)
    print(f'Splitting the rows...')
    # Splitting into train and test according to train:test ratio.
    split_point = int(round(train_ratio * len(rows)))
    train_rows = rows[:split_point]
    test_rows = rows[split_point:]
    # Computing the names for the new databases.
    train_db_path = db_path.replace(extension, '') + '_train' + extension
    test_db_path = db_path.replace(extension, '') + '_test' + extension
    print(f'Writing the new databases...')
    # Creating the new databases and inserting the rows.
    train_db = LLMClassifierDatabase(train_db_path)
    test_db = LLMClassifierDatabase(test_db_path)
    train_db.add_rows(train_rows)
    test_db.add_rows(test_rows)


def main() -> None:
    split_database('../../data/bloom_1_1B/dev_v2.1_short_prompts.sqlite3')


if __name__ == '__main__':
    main()
