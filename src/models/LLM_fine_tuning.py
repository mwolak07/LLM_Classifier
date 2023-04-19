import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from src.dataset import LLMClassifierDataset

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
model = TFAutoModelForSequenceClassification.from_pretrained('facebook/opt-125m', num_labels=2)


