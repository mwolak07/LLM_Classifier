from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, select, text
from sqlalchemy import Integer, String, Boolean
from sqlalchemy import Column, TypeDecorator
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.types import TypeEngine
from typing import Any, Optional, Type, List, Dict
import json


# Calling the sqlalchemy factory function for creating a base class for our tables.
Base = declarative_base()


class GPTClassifierDataset:
    """
    Responsible for providing a performant and easy-to-use interface for dataset for the GPT LLM classification problem.
    Uses GPTClassifierDatabase internally to store the dataset on disk.
    """
    pass


class GPTClassifierDatabase:
    """
    Responsible for providing an easy-to-use interface for creating and accessing the database for storing the data for
    the GPT LLM classification problem.
    Uses sqlalchemy for storing and loading the relevant info to/from an sqlite3 database.
    """
    session: Session

    def __init__(self, db_loc: str):
        """
        Initializes a new GPTClassifierDataset and connects to the database. The database will be created at the
        specified location if it is not already there.

        Args:
            db_loc: The location on disk or the network of the database.
        """
        self.session = self._connect_to_db(db_loc)

    def _connect_to_db(self, db_loc: str) -> Session:
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
        if self._gpt_classifier_data_exists(session):
            self._create_gpt_classifier_data_table(engine)

    @staticmethod
    def _gpt_classifier_data_exists(session: Session) -> bool:
        """
        Checks if the gpt_classifier_data table exists. Uses this.session to connect to the database.

        Args:
            session: The session that is connected to our database.

        Returns:
            True if the database exists, False if it does not.
        """
        table_name = 'gpt_classifier_data'
        # Querying the sqlite_master table for our table name.
        result = session.execute(
            select(text(f"1 FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
        ).scalar()
        # Checking if the query came back with any data.
        if result:
            return True
        else:
            return False

    @staticmethod
    def _create_gpt_classifier_data_table(engine: TypeEngine) -> None:
        """
        Creates the GPTClassifierTable in the database using the given engine.

        Args:
            engine: The engine that is connected to our database.

        Returns:
            The Session created after connecting to the database, after it is created.
        """
        Base.metadata.create_all(engine)

    def __del__(self):
        """
        Deletes this GPTClassifierDataset object. Makes sure we commit any pending changes and disconnect from the
        database.
        """
        self.session.commit()
        self.session.close()

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


class JsonList(TypeDecorator):
    """
    Allows us to store python lists as JSON objects in our sqlalchemy database.

    TODO: Add comments for what each method does.
    """
    impl = String

    def process_bind_param(self, value: Optional[List[Any]], dialect: Dialect) -> Optional[str]:
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value: Optional[str], dialect: Dialect) -> Optional[List[Any]]:
        if value is not None:
            value = json.loads(value)
        return value

    def process_literal_param(self, value: Optional[List[Any]], dialect: Dialect) -> Optional[str]:
        return self.process_bind_param(value, dialect)

    @property
    def python_type(self) -> Type:
        return list


class GPTClassifierTable(Base):
    """
    Defines the mapping for the table that will store the GPT classifier dataset for use with sqlalchemy.
    """
    __tablename__ = 'gpt_classifier_data'
    id = Column(Integer, primary_key=True)
    query = Column(String)
    passages = Column(JsonList)
    prompt = Column(String)
    human_answer = Column(String)
    gpt_answer = Column(String)
    has_answer = Column(Boolean)
