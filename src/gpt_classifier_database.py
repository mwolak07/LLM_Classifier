from __future__ import annotations
from sqlalchemy import Integer, String, Boolean, Table, Column, MetaData,\
    create_engine, insert, update, select, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.types import TypeDecorator, String as StringType
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.engine import Engine
from typing import Any, Optional, List, Tuple, Dict, Type
from collections.abc import Sequence
from dataclasses import dataclass
import json
from src import MSMarcoDataset


# Calling the sqlalchemy factory function for creating a base class for our tables.
Base = declarative_base()


@dataclass
class GPTClassifierRow:
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


class GPTClassifierDatabase(Sequence):
    """
    Responsible for providing an easy-to-use interface for creating and accessing the database for storing the data for
    the GPT LLM classification problem.
    Uses sqlalchemy for storing and loading the relevant info to/from an sqlite3 database.
    """
    _engine: Engine
    _session: Session
    _gpt_classifier_table: Table
    _metadata: MetaData

    def __init__(self, db_loc: str):
        """
        Initializes a new GPTClassifierDataset and connects to the database. The database will be created at the
        specified location if it is not already there.

        Args:
            db_loc: The location on disk or the network of the database.
        """
        self._metadata = MetaData()
        self._engine, self._session, self._table = self._connect_to_db(db_loc)

    def _connect_to_db(self, db_loc: str) -> Tuple[Engine, Session, Table]:
        """
        Connects to the database at the location specified. If the gpt_classifier_data table is not already there,
        creates it.

        Args:
            db_loc: The location on disk or the network of the database.

        Returns:
            The Session created after connecting to the database.
        """
        engine = create_engine(db_loc)  # Connecting to our database, creating the file if needed.
        BoundSession = sessionmaker(bind=engine)  # Creating a Session class bound to our engine.
        session = BoundSession()
        table = self._get_table(engine)
        return engine, session, table

    def _get_table(self, engine: Engine) -> Table:
        """
        Gets the Table object from the database
        If the gpt_classifier_data table exists, gets the Table object from the database.
        If not, creates the gpt_classifier_data table in the database.

        Args:
            engine: The sqlalchemy engine that handles our database.

        Returns:
            The Table object corresponding to the gpt_classifier_data table in the database.
        """
        table_name = 'gpt_classifier_data'
        # Trying to get the table from the database.
        try:
            table = Table(table_name, self._metadata, autoload_with=engine)
        except NoSuchTableError:
            table = Table(table_name, self._metadata,
                          Column('id', Integer, primary_key=True),
                          Column('query', String),
                          Column('passages', _JsonList),
                          Column('chosen_passages', _JsonList),
                          Column('prompt', String),
                          Column('human_answer', String),
                          Column('llm_answer', String),
                          Column('has_answer', Boolean)
                          )
            self._metadata.create_all(engine)
        return table

    def __del__(self):
        """
        Deletes this GPTClassifierDataset object. Makes sure we commit any pending changes and disconnect from the
        database.
        """
        self._session.commit()
        self._session.close()

    def add_ms_marco_dataset(self, dataset: MSMarcoDataset) -> None:
        """
        Adds the elements from the given MS Marco dataset to the database. This will populate all of the columns except
        for 'llm_answer', which it will leave blank. The id will be the index in the dataset.

        Assume:
            This should run first, only when the table is empty.

        Args:
            dataset: The MS Marco dataset object we will be getting the MS Marco data from.

        Raises:
            RuntimeError: When the table is not already empty.
        """
        # Should only run when the table is empty.
        if len(self) != 0:
            raise RuntimeError('Table is not empty!')
        # Unpacking the rows for each element in the dataset into a large list of rows. A flattening step is needed.
        values = [row for i in range(len(dataset)) for row in self._ms_marco_item_to_rows(i, dataset)]
        statement = insert(self._table).values(values)
        self._session.execute(statement)
        self._session.commit()

    @staticmethod
    def _ms_marco_item_to_rows(index: int, dataset: MSMarcoDataset) -> List[Dict[str, Any]]:
        """
        Converts the MS Marco element at the given index in the dataset to an dict, which represents the value for each
        column in the gpt_classifier_data table. Creates multiple rows for multiple answers. Omits llm_answer.

        Args:
            index: The index of the element in the MS Marco Dataset.
            dataset: The MS Marco dataset object we will be getting the MS Marco data from.

        Returns:
            A dict representing the row in the gpt_classifier_data table.
        """
        output = []
        # Grabbing the item.
        item = dataset[index]
        # Generating the prompt.
        prompt = dataset.prompt(index)
        # Going through each answer.
        for answer in item.answers:
            output.append({'id': index,
                           'query': dataset[index].query,
                           'passages': dataset[index].passages,
                           'chosen_passages': dataset[index].chosen_passages,
                           'prompt': prompt,
                           'human_answer': answer,
                           'llm_answer': None,
                           'has_answer': True})
        # There was no answer.
        if len(item.answers) == 0:
            output.append({'id': index,
                           'query': dataset[index].query,
                           'passages': dataset[index].passages,
                           'chosen_passages': dataset[index].chosen_passages,
                           'prompt': prompt,
                           'human_answer': dataset.no_answer_phrase,
                           'has_answer': False})
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
        Clears the gpt_classifier_dataset table. Asks the user for confirmation before proceeding.
        """
        choice = input('WARNING: You are deleting the database. Are you SURE this is your intention? (y/n): ')
        if choice != 'y':
            return
        self._session.query(self._table).delete()

    def __getitem__(self, index: int) -> GPTClassifierRow:
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
        Returns the number of rows in the gpt_classifier_data table.

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
