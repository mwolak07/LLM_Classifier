from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from src.dataset import LLMClassifierDataset
from src.util import cd_to_executing_file


cd_to_executing_file(__file__)
db_path = '../../data/bloom_1_1B/'
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
model = TFAutoModelForSequenceClassification.from_pretrained('facebook/opt-125m', num_labels=2)
dataset = LLMClassifierDataset(db_path)


