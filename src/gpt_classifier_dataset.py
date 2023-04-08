from __future__ import annotations
from sqlalchemy import Integer, String, Boolean,Table, Column, MetaData,\
    create_engine, insert, update, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.types import TypeDecorator, String as StringType
from sqlalchemy.engine import Engine
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass
import json


# Calling the sqlalchemy factory function for creating a base class for our tables.
Base = declarative_base()


@dataclass
class GPTClassifierItem:
    """
    Represents one element from the dataset for the GPT LLM classification problem.
    """
    pass


class GPTClassifierDataset:
    """
    Responsible for providing a performant and easy-to-use interface for dataset for the GPT LLM classification problem.
    Uses GPTClassifierDatabase internally to store the dataset on disk.
    """
    pass


class _GPTClassifierDatabase:
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
        except Exception:
            table = Table(table_name, self._metadata,
                          Column('id', Integer, primary_key=True),
                          Column('query', String),
                          Column('passages', _JsonList),
                          Column('prompt', String),
                          Column('human_answer', String),
                          Column('gpt_answer', String),
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

    def add_ms_marco_item(self, item: Dict[str, Any]) -> None:
        pass

    def add_ms_marco_items(self, item: List[Dict[str, Any]]) -> None:
        pass

    def add_gpt_answer(self, index: int) -> None:
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class _JsonList(TypeDecorator):
    """
    Allows us to store python lists as JSON objects in our sqlalchemy database.
    """
    impl = StringType  # Specifies which class we inherit behavior from.
    cache_ok = True  # Specifies that this type is OK to cache.

    def process_bind_param(self, value: Optional[List[Any]], dialect: Dialect) -> Any:
        """
        Receive a bound parameter value to be converted. Goes from the target data type to a bound parameter in the SQL
        statement.

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
        Receive a result-row column value to be converted. Goes from the data in the result row to the target data type.

        Args:
            value: The data to be converted.
            dialect: The Dialect in use.

        Returns:
            A List[Any] data type.
        """
        if value is not None:
            value = json.loads(value)
        return value
