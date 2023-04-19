from unittest.mock import patch
from typing import List
import unittest
import json
from src.dataset import LLMClassifierDatabase, LLMClassifierRow, MSMarcoDataset


class TestLLMClassifierDatabase(unittest.TestCase):
    """
    Tests that the LLMClassifierDatabase class works correctly.

    Attributes:
        mock_dataset_file: (class attribute) The file where we store our mock dataset.
        db: The database object that we are using for our tests.
    """
    ms_marco_file: str = 'mock_data_ms_marco.json'
    db: LLMClassifierDatabase

    def setUp(self):
        """
        Sets up the database for testing. Does this in memory, with the special :memory: flag.
        """
        self.db = LLMClassifierDatabase(':memory:')

    def test_getitem(self):
        """
        Tests that the database __getitem__ method works correctly.
        """
        # Raises an error when there are no items.
        with self.assertRaises(IndexError):
            element = self.db[0]
        # Creating the table and putting some elements in.
        self.db.create_table()
        statement = f'INSERT INTO {self.db.table_name} ' \
                    f'(query, passages, prompt, human_answer, llm_answer, has_answer) ' \
                    f'VALUES (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?);'
        values_1 = ['query1', json.dumps(["passage1, passage2"]), 'prompt1', 'hanswer1', 'lanswer1', True]
        values_2 = ['query2', json.dumps(["passage3, passage4"]), 'prompt2', '', 'lanswer2', False]
        self.db.execute(statement, values_1 + values_2)
        # Ensuring we get the elements back correctly.
        element_0 = LLMClassifierRow("query1", ["passage1, passage2"], "prompt1", "hanswer1", "lanswer1", True)
        element_1 = LLMClassifierRow("query2", ["passage3, passage4"], "prompt2", "", "lanswer2", False)
        self.assertEqual(element_0, self.db[0])
        self.assertEqual(element_1, self.db[1])
        # Ensuring we can't request elements that are not there.
        with self.assertRaises(IndexError):
            element = self.db[2]

    def test_len(self):
        """
        Tests that the database __len__ method works correctly.
        """
        self.db.create_table()
        self.assertEqual(0, len(self.db))
        statement = f'INSERT INTO {self.db.table_name} ' \
                    f'(query, passages, prompt, human_answer, llm_answer, has_answer) ' \
                    f'VALUES (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?);'
        values_1 = ['query1', json.dumps(["passage1, passage2"]), 'prompt1', 'hanswer1', 'lanswer1', True]
        values_2 = ['query2', json.dumps(["passage3, passage4"]), 'prompt2', '', 'lanswer2', False]
        self.db.execute(statement, values_1 + values_2)
        # Ensuring we get the db length correctly.
        self.assertEqual(2, len(self.db))
        # Ensures length is still 0 when we clear.
        # Clears if 'y' is in input.
        with patch('builtins.input', return_value='y'):
            self.db.clear()
        self.assertTrue(len(self.db) == 0)

    def test_table_exists(self):
        """
        Tests that table_exists correctly can tell if the table exists.
        """
        with patch('builtins.input', return_value='y'):
            self.db.clear()
        self.assertFalse(self.db.table_exists())
        self.db.execute(f'CREATE TABLE {self.db.table_name} ('
                        f'   id INTEGER PRIMARY KEY NOT NULL, '
                        f'   data TEXT'
                        f');')
        self.assertTrue(self.db.table_exists())

    def test_create_table(self):
        """
        Tests that create_table works correctly.
        """
        with patch('builtins.input', return_value='y'):
            self.db.clear()
        # Creating a table in the database will work.
        self.assertFalse(self.db.table_exists())
        self.db.create_table()
        self.assertTrue(self.db.table_exists())
        # Will not try to re-create the table if it is already there.
        self.db.create_table()
        self.assertTrue(self.db.table_exists())

    def test_add_ms_marco_dataset_short_prompts(self):
        """
        Tests that the mock ms marco dataset is correctly added to the database with short prompts.
        """
        # Adding the mock ms marco dataset
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=True)
        # Checking that the rows in the database match up with what we expect.
        expected_rows = [
            LLMClassifierRow(dataset[2].query, dataset[2].chosen_passages, None, dataset[2].answers[0], None, True),
            LLMClassifierRow(dataset[2].query, dataset[2].chosen_passages, None, dataset[2].answers[1], None, True),
            LLMClassifierRow(dataset[3].query, dataset[3].chosen_passages, None, dataset[3].answers[0], None, True),
            LLMClassifierRow(dataset[3].query, dataset[3].chosen_passages, None, dataset[3].answers[1], None, True),
            LLMClassifierRow(dataset[4].query, dataset[4].chosen_passages, None, dataset[4].answers[0], None, True),
            LLMClassifierRow(dataset[5].query, dataset[5].chosen_passages, None, dataset[5].answers[0], None, True),
            LLMClassifierRow(dataset[6].query, dataset[6].chosen_passages, None, dataset[6].answers[0], None, True),
            LLMClassifierRow(dataset[7].query, dataset[7].chosen_passages, None, dataset[7].answers[0], None, True),
            LLMClassifierRow(dataset[8].query, dataset[8].chosen_passages, None, dataset[8].answers[0], None, True),
            LLMClassifierRow(dataset[9].query, dataset[9].chosen_passages, None, dataset[9].answers[0], None, True),
        ]
        db_rows = [row for row in self.db]
        for i in range(len(self.db)):
            self.assertEqual(expected_rows[i], db_rows[i])

    def test_add_ms_marco_dataset_long_prompts(self):
        """
        Tests that the mock ms marco dataset is correctly added to the database with long prompts.
        """
        # Adding the mock ms marco dataset
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=False)
        # Checking that the rows in the database match up with what we expect.
        expected_rows = [
            LLMClassifierRow(dataset[0].query, dataset[0].passages, None, None, None, False),
            LLMClassifierRow(dataset[1].query, dataset[1].passages, None, None, None, False),
            LLMClassifierRow(dataset[2].query, dataset[2].passages, None, dataset[2].answers[0], None, True),
            LLMClassifierRow(dataset[2].query, dataset[2].passages, None, dataset[2].answers[1], None, True),
            LLMClassifierRow(dataset[3].query, dataset[3].passages, None, dataset[3].answers[0], None, True),
            LLMClassifierRow(dataset[3].query, dataset[3].passages, None, dataset[3].answers[1], None, True),
            LLMClassifierRow(dataset[4].query, dataset[4].passages, None, dataset[4].answers[0], None, True),
            LLMClassifierRow(dataset[5].query, dataset[5].passages, None, dataset[5].answers[0], None, True),
            LLMClassifierRow(dataset[6].query, dataset[6].passages, None, dataset[6].answers[0], None, True),
            LLMClassifierRow(dataset[7].query, dataset[7].passages, None, dataset[7].answers[0], None, True),
            LLMClassifierRow(dataset[8].query, dataset[8].passages, None, dataset[8].answers[0], None, True),
            LLMClassifierRow(dataset[9].query, dataset[9].passages, None, dataset[9].answers[0], None, True),
        ]
        db_rows = [row for row in self.db]
        for i in range(len(self.db)):
            self.assertEqual(expected_rows[i], db_rows[i])

    def test_add_prompts(self):
        """
        Tests that prompts are correctly added to the database.
        """
        # Adding the mock ms marco dataset.
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=True)
        # Checking that the prompts are added correctly.
        prompts = [f'prompt{i}' for i in range(len(self.db))]
        self.db.add_prompts(prompts)
        db_prompts = [row.prompt for row in self.db]
        for i in range(len(self.db)):
            self.assertEqual(prompts[i], db_prompts[i])

    def test_add_llm_answers(self):
        """
        Tests that llm answers are correctly added to the database.
        """
        # Adding the mock ms marco dataset.
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=True)
        # Checking that the answers are added correctly.
        answers = [f'answer{i}' for i in range(len(self.db))]
        self.db.add_llm_answers(answers)
        db_answers = [row.llm_answer for row in self.db]
        for i in range(len(self.db)):
            self.assertEqual(answers[i], db_answers[i])

    def test_add_llm_answer(self):
        """
        Tests that llm answers are correctly added to the database, if we go one at a time.
        """
        # Adding the mock ms marco dataset.
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=True)
        # Checking that the answers are added correctly.
        answers = [f'answer{i}' for i in range(len(self.db))]
        for i in range(len(self.db)):
            self.db.add_llm_answer(f'answer{i}', i)
        db_answers = [row.llm_answer for row in self.db]
        for i in range(len(self.db)):
            self.assertEqual(answers[i], db_answers[i])

    def fill_db_short_prompts(self):
        """
        Fills self.db with short prompts from the mock ms marco dataset.
        """
        with patch('builtins.input', return_value='y'):
            if self.db.table_exists():
                self.db.clear()
        self.db.create_table()
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=True)
        prompts = [f'prompt{i}' for i in range(len(self.db))]
        self.db.add_prompts(prompts)
        answers = [f'answer{i}' for i in range(len(self.db))]
        self.db.add_llm_answers(answers)

    def fill_db_long_prompts(self):
        """
        Fills self.db with long prompts from the mock ms marco dataset.
        """
        with patch('builtins.input', return_value='y'):
            if self.db.table_exists():
                self.db.clear()
        self.db.create_table()
        dataset = MSMarcoDataset(self.ms_marco_file)
        self.db.add_ms_marco_dataset(dataset, short_prompts=False)
        prompts = [f'prompt{i}' for i in range(len(self.db))]
        self.db.add_prompts(prompts)
        answers = [f'answer{i}' for i in range(len(self.db))]
        self.db.add_llm_answers(answers)

    def get_ms_marco_answers(self) -> List[str]:
        """
        Reads the answers from the ms_marco_file directly.
        """
        output = []
        with open(self.ms_marco_file, 'r') as f:
            data = json.load(f)
            answers_dict = data['answers']
            well_formed_answers_dict = data['wellFormedAnswers']
            for key in answers_dict.keys():
                answers = answers_dict[key]
                well_formed_answers = well_formed_answers_dict[key]
                if well_formed_answers != '[]':
                    output += well_formed_answers
                elif answers == ['No Answer Present.']:
                    output += [None]
                else:
                    output += answers
            return output

    def test_human_answers(self):
        """
        Tests that the human_answers() method gets all of the human answers correctly.
        """
        # Testing it works correctly with short prompts.
        self.fill_db_short_prompts()
        answers = [answer for answer in self.get_ms_marco_answers() if answer is not None]
        db_answers = self.db.human_answers()
        for i in range(len(self.db)):
            self.assertEqual(answers[i], db_answers[i])

        # Testing it works correctly with long prompts.
        self.fill_db_long_prompts()
        answers = self.get_ms_marco_answers()
        db_answers = self.db.human_answers()
        for i in range(len(self.db)):
            self.assertEqual(answers[i], db_answers[i])

    def test_prompts(self):
        """
        Tests that the prompts() method gets all of the prompts correctly.
        """
        # Testing it works correctly with short prompts.
        self.fill_db_short_prompts()
        prompts = [f'prompt{i}' for i in range(len(self.db))]
        db_prompts = self.db.prompts()
        for i in range(len(self.db)):
            self.assertEqual(prompts[i], db_prompts[i])

        # Testing it works correctly with long prompts.
        self.fill_db_long_prompts()
        prompts = [f'prompt{i}' for i in range(len(self.db))]
        db_prompts = self.db.prompts()
        for i in range(len(self.db)):
            self.assertEqual(prompts[i], db_prompts[i])

    def test_llm_answers(self):
        """
        Tests that the llm_answers() method gets all of the prompts correctly.
        """
        # Testing it works correctly with short prompts.
        self.fill_db_short_prompts()
        answers = [f'answer{i}' for i in range(len(self.db))]
        db_answers = self.db.llm_answers()
        for i in range(len(self.db)):
            self.assertEqual(answers[i], db_answers[i])

        # Testing it works correctly with long prompts.
        self.fill_db_long_prompts()
        answers = [f'answer{i}' for i in range(len(self.db))]
        db_answers = self.db.llm_answers()
        for i in range(len(self.db)):
            self.assertEqual(answers[i], db_answers[i])

    def test_clear(self):
        """
        Tests that the clear method works correctly.
        """
        self.fill_db_short_prompts()
        # Does not clear if 'n' is in input.
        with patch('builtins.input', return_value='n'):
            self.db.clear()
        self.assertTrue(self.db.table_exists())
        self.assertTrue(len(self.db) > 0)
        # Does not clear if random string is in input.
        with patch('builtins.input', return_value='a'):
            self.db.clear()
        self.assertTrue(self.db.table_exists())
        self.assertTrue(len(self.db) > 0)
        # Clears if 'y' is in input.
        with patch('builtins.input', return_value='y'):
            self.db.clear()
        self.assertFalse(self.db.table_exists())
        self.assertTrue(len(self.db) == 0)
