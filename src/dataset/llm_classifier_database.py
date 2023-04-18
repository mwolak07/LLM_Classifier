from typing import Any, Iterable, List, Dict, Tuple, Union
from sqlite3 import connect, Connection, Cursor
from collections.abc import Sequence
from dataclasses import dataclass
import json
from src.dataset import MSMarcoDataset


@dataclass
class LLMClassifierRow:
    """
    Represents one item from the dataset for the GPT LLM classification problem.
    """
    query: str
    passages: List[str]
    prompt: str
    human_answer: str
    llm_answer: str
    has_answer: bool


class LLMClassifierDatabase(Sequence):
    """
    Responsible for providing an easy-to-use interface for creating and accessing the database for storing the data for
    the GPT LLM classification problem.
    Uses sqlalchemy for storing and loading the relevant info to/from an sqlite3 database.

    Attributes:
        _table: (class attribute) The name of the table where we will store the data.
        _connection: The connection to the database.
        _cursor: The cursor for the database, attached to the connection.
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
        self._create_table()

    def __del__(self):
        """
        Deletes this GPTClassifierDataset object. Makes sure we commit any pending changes and disconnect from the
        database.
        """
        self.commit()
        self.close()

    def __getitem__(self, index: int) -> LLMClassifierRow:
        """
        Returns the row at 'id' index.

        Args:
            index: The index of the row to retrieve. This corresponds to the row 'id'.

        Returns:
            A GPTClassifierItem, which represents one row in our database.
        """
        statement = f'SELECT (:cols) FROM {self._table} WHERE id=:id;'
        values = {'cols': self.columns_json_extract(), 'id': index}
        result = self.execute(statement, values).fetchone()
        return self.decode_row(result)

    def __len__(self) -> int:
        """
        Returns the number of rows in the llm_classifier_data table.

        Returns:
            The number of rows in the table.
        """
        statement = f'SELECT COUNT(*) FROM {self._table};'
        result = self.execute(statement).fetchone()
        return result

    def execute(self, sql: str, parameters: Iterable[Any] = ()) -> Cursor:
        """
        Uses the internal self._cursor to execute an SQL statement on the database. Supports exactly the same
        functionality as sqlite3.Cursor.execute().

        Args:
            sql: The SQL statement to execute.
            parameters: The parameters to safely insert into the SQL statement, if needed.

        Returns:
            The Cursor object containing the results of the query.
        """
        return self._connection.execute(sql, parameters)

    def executemany(self, sql: str, parameters: Iterable[Iterable[Any]] = ()) -> Cursor:
        """
        Uses the internal self._cursor to execute an SQL statement multiple times for a list of parameters on the
        database. Supports exactly the same functionality as sqlite3.Cursor.executemany().

        Args:
            sql: The SQL statement to execute.
            parameters: The list of sets of parameters to safely insert into the SQL statement, if needed.

        Returns:
            The Cursor object containing the results of the query.
        """
        return self._connection.executemany(sql, parameters)

    def commit(self):
        """
        Uses the internal self._connection to commit any pending changes to the database. Supports exactly the same
        functionality as sqlite3.Connection.commit().
        """
        self._connection.commit()

    def close(self):
        """
        Uses the internal self._connection to close the connection the database. Supports exactly the same functionality
         as sqlite3.Connection.close().
        """
        self._connection.close()

    @staticmethod
    def columns_json_extract() -> str:
        """
        Gets a string of all of the columns with json_extract added appropriately in the table. Intended for SELECT.

        Returns:
            The columns of the table, with json_extract added where needed.
        """
        return 'id, ' \
               'query, ' \
               'json_extract(passages, "$"), ' \
               'json_extract(chosen_passages, "$"), ' \
               'prompt, ' \
               'human_answer, ' \
               'llm_answer, ' \
               'has_answer'

    @staticmethod
    def decode_row(row: Tuple[int, str, str, str, str, str, int]) -> LLMClassifierRow:
        """
        Takes a row from the database, and converts it into python types. Decodes the stringified JSON where needed,
        and converts everything into an LLMClassifierRow.
        All NULL values will be None in the LLMClassifierRow.

        Args:
            row: A row from self._table: (id, query, passages, chosen_passages, prompt, human_answer, llm_answer,
                                          has_answer)

        Returns:
            A pythonic LLMClassifierRow representation of the database row.
        """
        # Getting mutable list from the Tuple row.
        values = list(row)
        # Decoding the JSON strings.
        values[2] = list(json.loads(values[2]))
        # Converting int to boolean.
        values[6] = bool(values[6])
        # Converting to a LLMClassifierRow. Does not include id.
        return LLMClassifierRow(values[1], values[2], values[3], values[4], values[5], values[6])

    def _table_exists(self) -> bool:
        """
        Determines if self._table has already been created in the database.

        Returns:
            True if self._table is already in the database, False if it is not.
        """
        statement = f'SELECT name FROM sqlite_schema WHERE type="table" AND name={self._table};'
        result = self.execute(statement).fetchone()
        return result > 0

    def _create_table(self) -> None:
        """
        Creates self._table in the database if it does not already exist.
        """
        # Setting up oir statement.
        statement = f'CREATE TABLE {self._table} (' \
                    f'  id INTEGER AUTOINCREMENT PRIMARY KEY NOT NULL,' \
                    f'  query TEXT,' \
                    f'  passages TEXT,' \
                    f'  prompt TEXT,' \
                    f'  human_answer TEXT,' \
                    f'  llm_answer TEXT,' \
                    f'  has_answer INTEGER' \
                    f');'
        # Checking if the table exists
        if not self._table_exists():
            self.execute(statement)
            self.commit()

    def add_ms_marco_dataset(self, dataset: MSMarcoDataset, short_prompts: bool) -> None:
        """
        Adds the elements from the given MS Marco dataset to the database. This will populate all of the columns except
        for 'llm_answer', which it will leave blank. The id will be the index in the dataset.

        Assume:
            This should run first, only when the table is empty.

        Args:
            dataset: The MS Marco dataset object we will be getting the MS Marco data from.
            short_prompts: True if we are using short prompts. If this is true, we don't include rows with no answer,
                           and use chosen_passages instead of passages.

        Raises:
            RuntimeError: When the table is not already empty.
        """
        # Should only run when the table is empty.
        if len(self) != 0:
            raise RuntimeError('Table is not empty!')
        # Creating the statement to be executed many times
        statement = f'INSERT INTO {self._table}' \
                    f'(id, query, passages, chosen_passages, prompt, human_answer, has_answer)' \
                    f'VALUES (' \
                    f'   :id, ' \
                    f'   :query, ' \
                    f'   json_array(:passages), ' \
                    f'   :human_answer,' \
                    f'   :has_answer' \
                    f');'
        # Getting the values by flattening the output of the values dicts for each row from self._ms_marco_item_to_rows.
        values_list = [values for i in range(len(dataset))
                       for values in self._ms_marco_item_to_rows(i, dataset, short_prompts)]
        # Executing the statement for all values in the list.
        self.executemany(statement, values_list)
        self.commit()

    @staticmethod
    def _ms_marco_item_to_rows(index: int, dataset: MSMarcoDataset, short_prompts: bool) \
            -> List[Dict[str, Union[int, str]]]:
        """
        Converts the MS Marco element at the given index in the dataset to an dict, which represents the value for each
        column in the llm_classifier_data table. Creates multiple rows for multiple answers. Omits llm_answer.

        Args:
            index: The index of the element in the MS Marco Dataset.
            dataset: The MS Marco dataset object we will be getting the MS Marco data from.
            short_prompts: True if we are using short prompts. If this is true, we don't include rows with no answer,
                           and use chosen_passages instead of passages.

        Returns:
            A dict representing the row in the llm_classifier_data table.
        """
        output = []
        # Grabbing the item.
        item = dataset[index]
        # Grabbing the passages based on short_prompts.
        passages = dataset[index].passages if not short_prompts else dataset[index].chosen_passages
        # Going through each answer, and making a values object. This will not create any for prompts with no answer.
        for answer in item.answers:
            # Adding the new values as a values dict to our output.
            values = {
                'query': dataset[index].query,
                'passages': json.dumps(passages),
                'answer': answer,
                'has_answer': int(True)
            }
            output.append(values)
        # Generating an entry with has_answer = False, if there is no answer, and we are not using short prompts.
        if len(item.answers) == 0 and not short_prompts:
            values = {
                'query': dataset[index].query,
                'passages': json.dumps(passages),
                'answer': None,
                'has_answer': int(False)
            }
            output.append(values)
        return output

    def add_prompts(self, prompts: List[str]) -> None:
        """
        Takes a list of generated prompts, in the order of row 'id', and inserts them.

        Assume:
            This should run only after add_ms_marco_dataset, when the table is not empty.

        Args:
            prompts: A list of prompts generated based on the MS_MARCO data, in order of row 'id'.

        Raises:
            RuntimeError: When the table is empty.
        """
        # Should only run when the table is not empty.
        if len(self):
            raise RuntimeError('Table is empty!')
        # Creating the statement.
        statement = f'UPDATE {self._table} SET prompt = :prompt WHERE id = :id;'
        # Getting a list of values objects
        values_list = [{'prompt': prompts[i], 'id': i} for i in range(len(prompts))]
        # Executing the statement for all values in the list
        self.executemany(statement, values_list)
        self.commit()

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
        # Should only run when the table is not empty.
        if len(self):
            raise RuntimeError('Table is empty!')
        # Creating the statement.
        statement = f'UPDATE {self._table} SET llm_answer = :llm_answer WHERE id = :id;'
        # Getting a list of values objects
        values_list = [{'llm_answer': answers[i], 'id': i} for i in range(len(answers))]
        # Executing the statement for all values in the list
        self.executemany(statement, values_list)
        self.commit()

    def human_answers(self) -> List[str]:
        """
        Returns a list of all the human answers, in order by increasing row 'id'.

        Returns:
            A list of human answers, ordered by increasing row 'id'.
        """
        statement = f'SELECT (human_answer) FROM {self._table} ORDER BY id;'
        result = self.execute(statement).fetchall()
        return list(result)

    def prompts(self) -> List[str]:
        """
        Returns a list of all of the prompts, in order by increasing row 'id'.

        Returns:
            A list of prompts, ordered by increasing row 'id'.
        """
        statement = f'SELECT (prompt) FROM {self._table} ORDER BY id;'
        result = self.execute(statement).fetchall()
        return list(result)

    def llm_answers(self) -> List[str]:
        """
        Returns a list of all the llm answers, in order by increasing row 'id'.

        Returns:
            A list of llm answers, ordered by increasing row 'id'.
        """
        statement = f'SELECT (llm_answer) FROM {self._table} ORDER BY id;'
        result = self.execute(statement).fetchall()
        return list(result)

    def clear(self) -> None:
        """
        Clears the llm_classifier_data table. Asks the user for confirmation before proceeding.
        """
        choice = input('WARNING: You are deleting the database. Are you SURE this is your intention? (y/n): ')
        if choice != 'y':
            return
        statement = f'DROP TABLE {self._table};'
        self.execute(statement)
        self.commit()
