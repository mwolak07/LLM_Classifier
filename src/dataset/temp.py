import random
from src.dataset import LLMClassifierDatabase


database = LLMClassifierDatabase('../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3')
indices = list(range(len(database)))
random_indices = random.sample(indices, 10)
rows = [database[i] for i in random_indices]
small_db = LLMClassifierDatabase('../../data/bloom_1_1B/dev_v2.1_short_prompts_small.sqlite3')
small_db.add_rows(rows)
