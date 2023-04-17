from __future__ import annotations
from typing import Any, Optional, List, Tuple, Dict, Type
from collections.abc import Sequence
from dataclasses import dataclass
from sqlite3 import connect, Connection, Cursor
import json
from src.dataset import MSMarcoDataset


@dataclass
class LLMClassifierRow:
    """
    Represents one item from the dataset for the GPT LLM classification problem.
    """
    query: str
    passages: List[str]
    chosen_passages: List[str]
    prompt: str
    human_answer: str
    llm_answer: str
    has_answer: bool


class LLMClassifierDatabase(Sequence):
    """
    Responsible for providing an easy-to-use interface for creating and accessing the database for storing the data for
    the GPT LLM classification problem.
    Uses sqlalchemy for storing and loading the relevant info to/from an sqlite3 database.
    """
    _table: str = 'llm_classifier_data'
    _connection: Connection
    _cursor: Cursor

    def __init__(self, db_path: str):
        """
        Initializes a new GPTClassifierDataset and connects to the database. The database will be created at the
        specified location if it is not already there.

        Args:
            db_path: The location on disk or the network of the database.
        """

        self._connection = connect(db_path)
        self._cursor = self._connection.cursor()

    def _create_table(self):
        """
        Creates self._table in the database if it does not already exist.
        """
        # Setting up oir statement.
        statement = f'CREATE TABLE "{self._table}" (' \
                    f'  id int primary_key,' \
                    f''
        # Checking if the table exists
        if not self._table_exists():


    def __del__(self):
        """
        Deletes this GPTClassifierDataset object. Makes sure we commit any pending changes and disconnect from the
        database.
        """
        self._connection.commit()
        self._connection.close()

    def add_ms_marco_dataset(self, dataset: MSMarcoDataset, short_prompts: bool = True) -> None:
        """
        Adds the elements from the given MS Marco dataset to the database. This will populate all of the columns except
        for 'llm_answer', which it will leave blank. The id will be the index in the dataset.

        Assume:
            This should run first, only when the table is empty.

        Args:
            dataset: The MS Marco dataset object we will be getting the MS Marco data from.
            short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                           will be excluded. This will remove the "no answer" instruction from the prompt as well.

        Raises:
            RuntimeError: When the table is not already empty.
        """
        # Should only run when the table is empty.
        if len(self) != 0:
            raise RuntimeError('Table is not empty!')
        # Unpacking the rows for each element in the dataset into a large list of rows. A flattening step is needed.
        values = [row for i in range(len(dataset)) for row in self._ms_marco_item_to_rows(i, dataset, short_prompts)]
        statement = insert(self._table).values(values)
        self._session.execute(statement)
        self._session.commit()

    @staticmethod
    def _ms_marco_item_to_rows(index: int, dataset: MSMarcoDataset, short_prompts: bool = True) \
            -> List[Dict[str, Any]]:
        """
        Converts the MS Marco element at the given index in the dataset to an dict, which represents the value for each
        column in the llm_classifier_data table. Creates multiple rows for multiple answers. Omits llm_answer.

        Args:
            index: The index of the element in the MS Marco Dataset.
            dataset: The MS Marco dataset object we will be getting the MS Marco data from.
            short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                           will be excluded. This will remove the "no answer" instruction from the prompt as well.

        Returns:
            A dict representing the row in the llm_classifier_data table.
        """
        output = []
        # Grabbing the item.
        item = dataset[index]
        # Generating the prompt. This is different for short_prompts.
        if short_prompts:
            prompt = dataset.prompt(index, short_prompts) if len(item.answers) != 0 else None
        else:
            prompt = dataset.prompt(index, short_prompts)
        # Going through each answer. If there is no answer, we get back an empty list.
        for answer in item.answers:
            output.append({'id': index,
                           'query': dataset[index].query,
                           'passages': dataset[index].passages,
                           'chosen_passages': dataset[index].chosen_passages,
                           'prompt': prompt,
                           'human_answer': answer,
                           'llm_answer': None,
                           'has_answer': True})
        return output

    def add_llm_answers(self, answers: List[str]) -> None:
        """
        Takes a list of answers from the LLM, in the order of row 'id', and inserts them.

        Assume:
            This should run only after add_ms_marco_dataset, when the table is not empty.

        Args:
            answers: A list of answers the LLM gave to the prompts, in order of row 'id'.

        Raises:
            RuntimeError: When the table is empty.
        """
        if len(self):
            raise RuntimeError('Table is empty!')
        values = [{'llm_answer': answer} for answer in answers]
        statement = update(self._table).values(values)
        self._session.execute(statement)
        self._session.commit()

    def prompts(self) -> List[str]:
        """
        Returns a list of all of the prompts, in order by increasing row 'id'.

        Returns:
            A list of prompts, ordered by increasing row 'id'.
        """
        statement = select(self._table.prompt).order_by(self._table.id)
        prompts = self._session.execute(statement).all()
        return prompts

    def clear(self) -> None:
        """
        Clears the llm_classifier_data table. Asks the user for confirmation before proceeding.
        """
        choice = input('WARNING: You are deleting the database. Are you SURE this is your intention? (y/n): ')
        if choice != 'y':
            return
        self._session.query(self._table).delete()

    def __getitem__(self, index: int) -> LLMClassifierRow:
        """
        Returns the row at 'id' index.

        Args:
            index: The index of the row to retrieve. This corresponds to the row 'id'.

        Returns:
            A GPTClassifierItem, which represents one row in our database.
        """
        statement = select(self._table).where(self._table.id == index)
        item = self._session.execute(statement).first()
        return item

    def __len__(self) -> int:
        """
        Returns the number of rows in the llm_classifier_data table.

        Returns:
            The number of rows in the table.
        """
        statement = select(func.count()).select_from(self._table)
        rows = self._session.execute(statement).scalar()
        return rows


class _JsonList(TypeDecorator):
    """
    Allows us to store python lists as JSON objects in our sqlalchemy database.
    """
    impl = StringType  # Specifies which class we inherit behavior from.
    cache_ok = True  # Specifies that this type is OK to cache.

    def process_bind_param(self, value: Optional[List[Any]], dialect: Dialect) -> Any:
        """
        Receive a bound parameter value to be converted. Goes from the target data type (list) to a bound parameter in
        the SQL statement (string representation of JSON).

        Args:
            value: The data to be converted.
            dialect: The Dialect in use.

        Returns:
            A bound SQL parameter.
        """
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value: Optional[Any], dialect: Dialect) -> Optional[List[Any]]:
        """
        Receive a result-row column value to be converted. Goes from the data in the result row
        (string representation of JSON) to the target data type (list).

        Args:
            value: The data to be converted.
            dialect: The Dialect in use.

        Returns:
            A List[Any] data type.
        """
        if value is not None:
            value = json.loads(value)
        return value

    def process_literal_param(self, value: Optional[List[Any]], dialect: Dialect) -> str:
        """
        Receive a literal parameter value to be rendered inline within a statement. Goes from the target data (list)
        to a string.
        """
        return str(self.process_bind_param(value, dialect))

    @property
    def python_type(self) -> Type:
        """
        Returns the python type this class represents.
        """
        return list
