from typing import Any, Iterable, List, Dict, Tuple, Union, Optional
from sqlite3 import connect, Connection, Cursor
from collections.abc import Sequence
from dataclasses import dataclass
from numpy import ndarray
import numpy as np
import json
from src.dataset import MSMarcoDataset


@dataclass
class LLMClassifierRow:
    """
    Represents one item from the dataset for the GPT LLM classification problem.
    """
    query: Optional[str]
    passages: Optional[List[str]]
    prompt: Optional[str]
    human_answer: Optional[str]
    llm_answer: Optional[str]
    has_answer: Optional[bool]
    human_answer_fasttext: Optional[ndarray[ndarray[float]]]
    llm_answer_fasttext: Optional[ndarray[ndarray[float]]]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LLMClassifierRow):
            return False
        same_query = self.query == other.query
        same_passages = self.passages == other.passages
        same_prompt = self.prompt == other.prompt
        same_human_answer = self.human_answer == other.human_answer
        same_llm_answer = self.llm_answer == other.llm_answer
        same_has_answer = self.has_answer == other.has_answer
        same_human_answer_fasttext = self.human_answer_fasttext == other.human_answer_fasttext
        same_llm_answer_fasttext = self.llm_answer_fasttext == other.llm_answer_fasttext
        return same_query and same_passages and same_prompt and same_human_answer and same_llm_answer \
               and same_has_answer and same_human_answer_fasttext and same_llm_answer_fasttext


class LLMClassifierDatabase(Sequence):
    """
    Responsible for providing an easy-to-use interface for creating and accessing the database for storing the data for
    the GPT LLM classification problem.
    Uses sqlalchemy for storing and loading the relevant info to/from an sqlite3 database.

    Attributes:
        table_name: (class attribute) The name of the table where we will store the data.
        _connection: The connection to the database.
        _cursor: The cursor for the database, attached to the connection.
    """
    table_name: str = 'llm_classifier_data'
    _connection: Connection
    _cursor: Cursor

    def __init__(self, db_path: str):
        """
        Initializes a new GPTClassifierDataset and connects to the database. The database will be created at the
        specified location if it is not already there.

        Args:
            db_path: The location on disk or the network of the database.
        """
        self._connection = connect(db_path, check_same_thread=False)
        self._cursor = self._connection.cursor()
        self.create_table()

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
        if index >= len(self):
            raise IndexError(f'index {index} is out of range of database of length {len(self)}')
        statement = f'SELECT {self.columns()} FROM {self.table_name} WHERE id=:id;'
        values = {'cols': self.columns(), 'id': index + 1}
        result = self.execute(statement, values).fetchone()
        return self.decode_row(result)

    def __len__(self) -> int:
        """
        Returns the number of rows in the llm_classifier_data table.

        Returns:
            The number of rows in the table.
        """
        if not self.table_exists():
            return 0
        statement = f'SELECT COUNT(*) FROM {self.table_name};'
        result = self.execute(statement).fetchone()
        return result[0]

    def tolist(self) -> List[LLMClassifierRow]:
        """
        Converts the entire database to a list of rows, moving the entire thing into memory.
        """
        if not self.table_exists():
            return []
        statement = f'SELECT * FROM {self.table_name};'
        result = self.execute(statement).fetchall()
        return [self.decode_row(row) for row in result]

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
    def columns() -> str:
        """
        Gets a string of all of the columns. Intended for SELECT, INSERT, or UPDATE.

        Returns:
            The columns of the table.
        """
        return 'id, ' \
               'query, ' \
               'passages, ' \
               'prompt, ' \
               'human_answer, ' \
               'llm_answer, ' \
               'has_answer, ' \
               'human_answer_fasttext, ' \
               'llm_answer_fasttext'

    @staticmethod
    def decode_row(row: Tuple[int, str, str, str, str, str, int, str, str]) -> LLMClassifierRow:
        """
        Takes a row from the database, and converts it into python types. Decodes the stringified JSON where needed,
        and converts everything into an LLMClassifierRow.
        All NULL values will be None in the LLMClassifierRow.

        Args:
            row: A row from self.table_name: (id, query, passages, prompt, human_answer, llm_answer,
                                              has_answer, human_answer_fasttext, llm_answer_fasttext)

        Returns:
            A pythonic LLMClassifierRow representation of the database row.
        """
        # Getting mutable list from the Tuple row.
        values = list(row)
        # Decoding the JSON strings.
        values[2] = list(json.loads(values[2])) if values[2] is not None else None
        # Converting int to boolean.
        values[6] = bool(values[6]) if values[6] is not None else None
        # Decoding the 2D numpy arrays.
        values[7] = np.array(json.loads(values[7])) if values[7] is not None else None
        values[8] = np.array(json.loads(values[8])) if values[8] is not None else None
        # Converting to a LLMClassifierRow. Does not include id.
        return LLMClassifierRow(values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8])

    def table_exists(self) -> bool:
        """
        Determines if self.table_name has already been created in the database.

        Returns:
            True if self.table_name is already in the database, False if it is not.
        """
        statement = f'SELECT COUNT(*) FROM sqlite_schema WHERE type="table" AND name="{self.table_name}";'
        result = self.execute(statement).fetchone()
        return result[0] > 0

    def create_table(self) -> None:
        """
        Creates self.table_name table and index in the database if it does not already exist.
        """
        # Setting up our statement for creating the table.
        table_statement = f'CREATE TABLE {self.table_name} (' \
                          f'  id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,' \
                          f'  query TEXT,' \
                          f'  passages TEXT,' \
                          f'  prompt TEXT,' \
                          f'  human_answer TEXT,' \
                          f'  llm_answer TEXT,' \
                          f'  has_answer INTEGER,' \
                          f'  human_answer_fasttext TEXT,' \
                          f'  llm_answer_fasttext TEXT' \
                          f');'
        # Setting up our statement for creating the index.
        index_statement = f'CREATE INDEX {self.table_name}_id_idx ON {self.table_name} (id);'
        # Checking if the table exists
        if not self.table_exists():
            self.execute(table_statement)
            self.execute(index_statement)
            self.commit()

    def add_rows(self, rows: List[LLMClassifierRow]) -> None:
        """
        Inserts the given rows into the table.

        Args:
            rows: The rows to insert into the database
        """
        statement = f'INSERT INTO {self.table_name}' \
                    f'(query, passages, prompt, human_answer, llm_answer, has_answer, human_answer_fasttext, ' \
                    f'llm_answer_fasttext) ' \
                    f'VALUES (' \
                    f'  :query, ' \
                    f'  :passages, ' \
                    f'  :prompt, ' \
                    f'  :human_answer, ' \
                    f'  :llm_answer, ' \
                    f'  :has_answer, ' \
                    f'  :human_answer_fasttext, ' \
                    f'  :llm_answer_fasttext ' \
                    f')'
        # Encoding rows
        values_list = [{'query': row.query, 'passages': json.dumps(row.passages), 'prompt': row.prompt,
                        'human_answer': row.human_answer, 'llm_answer': row.llm_answer,
                        'has_answer': int(row.has_answer),
                        'human_answer_fasttext': json.dumps([word.tolist() for word in row.human_answer_fasttext]),
                        'llm_answer_fasttext': json.dumps([word.tolist() for word in row.llm_answer_fasttext])}
                       for row in rows]
        # Executing the statement for all values in the list.
        self.executemany(statement, values_list)
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
        statement = f'INSERT INTO {self.table_name}' \
                    f'(query, passages, human_answer, has_answer)' \
                    f'VALUES (' \
                    f'  :query, ' \
                    f'  :passages, ' \
                    f'  :human_answer,' \
                    f'  :has_answer' \
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
                'human_answer': answer,
                'has_answer': int(True)
            }
            output.append(values)
        # Generating an entry with has_answer = False, if there is no answer, and we are not using short prompts.
        if len(item.answers) == 0 and not short_prompts:
            values = {
                'query': dataset[index].query,
                'passages': json.dumps(passages),
                'human_answer': None,
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
        if len(self) == 0:
            raise RuntimeError('Table is empty!')
        # Creating the statement.
        statement = f'UPDATE {self.table_name} SET prompt = :prompt WHERE id = :id;'
        # Getting a list of values objects
        values_list = [{'prompt': prompts[i], 'id': i + 1} for i in range(len(prompts))]
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
        if len(self) == 0:
            raise RuntimeError('Table is empty!')
        # Creating the statement.
        statement = f'UPDATE {self.table_name} SET llm_answer = :llm_answer WHERE id = :id;'
        # Getting a list of values objects
        values_list = [{'llm_answer': answers[i], 'id': i + 1} for i in range(len(answers))]
        # Executing the statement for all values in the list
        self.executemany(statement, values_list)
        self.commit()

    def add_llm_answer(self, answer: str, index: int) -> None:
        """
        Takes a single answer from the LLM, and inserts it at the given index.

        Assume:
            This should run only after add_ms_marco_dataset, when the table is not empty.

        Args:
            answer: The answer the LLM gave to the prompt at the given index.
            index: The index of the prompt the LLM was r

        Raises:
            RuntimeError: When the table is empty.
        """
        # Should only run when the table is not empty.
        if len(self) == 0:
            raise RuntimeError('Table is empty!')
        # Creating the statement.
        statement = f'UPDATE {self.table_name} SET llm_answer = :llm_answer WHERE id = :id;'
        # Getting the value object.
        value = {'llm_answer': answer, 'id': index + 1}
        # Executing the statement.
        self.execute(statement, value)
        self.commit()

    def add_fasttext(self, human_answer: ndarray[ndarray[float]], llm_answer: ndarray[ndarray[float]],
                     index: int) -> None:
        """
        Takes a single fasttext vectorized answer from the human, and inserts it at the given index.

        Assume:
            This should run only after add_ms_marco_dataset, when the table is not empty.

        Args:
            human_answer: The vectorized answer the human gave for this prompt.
            llm_answer: The vectorized answer the LLM gave for this prompt.
            index: The index of the prompt the LLM was r.

        Raises:
            RuntimeError: When the table is empty.
        """
        # Should only run when the table is not empty.
        if len(self) == 0:
            raise RuntimeError('Table is empty!')
        # Creating the statement.
        statement = f'UPDATE {self.table_name} SET human_answer_fasttext = :human_answer_fasttext,' \
                    f'                             llm_answer_fasttext = :llm_answer_fasttext' \
                    f' WHERE id = :id;'
        # Converting from numpy arrays to python lists and then JSON stings.
        human_answer = json.dumps([word.tolist() for word in human_answer])
        llm_answer = json.dumps([word.tolist() for word in llm_answer])
        # Getting the value object.
        value = {'human_answer_fasttext': human_answer, 'llm_answer_fasttext': llm_answer, 'id': index + 1}
        # Executing the statement.
        self.execute(statement, value)
        self.commit()

    def human_answers(self) -> List[str]:
        """
        Returns a list of all the human answers, in order by increasing row 'id'.

        Returns:
            A list of human answers, ordered by increasing row 'id'.
        """
        statement = f'SELECT human_answer FROM {self.table_name} ORDER BY id;'
        result = self.execute(statement).fetchall()
        return [row[0] for row in result]

    def prompts(self) -> List[str]:
        """
        Returns a list of all of the prompts, in order by increasing row 'id'.

        Returns:
            A list of prompts, ordered by increasing row 'id'.
        """
        statement = f'SELECT prompt FROM {self.table_name} ORDER BY id;'
        result = self.execute(statement).fetchall()
        return [row[0] for row in result]

    def llm_answers(self) -> List[str]:
        """
        Returns a list of all the llm answers, in order by increasing row 'id'.

        Returns:
            A list of llm answers, ordered by increasing row 'id'.
        """
        statement = f'SELECT llm_answer FROM {self.table_name} ORDER BY id;'
        result = self.execute(statement).fetchall()
        return [row[0] for row in result]

    def clear(self) -> None:
        """
        Clears the llm_classifier_data table. Asks the user for confirmation before proceeding.
        """
        choice = input('WARNING: You are deleting the database. Are you SURE this is your intention? (y/n): ')
        if choice != 'y':
            return
        statement = f'DROP TABLE {self.table_name};'
        self.execute(statement)
        self.commit()

    def clear_empty_rows(self) -> None:
        """
        Deletes rows that have NULL for llm_answer.
        """
        statement = f'DELETE FROM {self.table_name} WHERE llm_answer is NULL;'
        self.execute(statement)
        self.commit()
