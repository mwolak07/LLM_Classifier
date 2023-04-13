from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset_llms import InferenceLLM


class InferenceGPT2XL(InferenceLLM):
    """
    Represents a GPT 2-XL that we can use for inference.

    Attributes:
        _temperature: (class attribute) The temperature for GPT 2-XL.
        _model: The model we will use to perform inference.
        _tokenizer: The model we will use to transform the input strings into vectors.
    """

    def __init__(self):
        """s
        Initializes the model and tokenizer with the appropriate parameters for inference.
        """
        self._model = AutoTokenizer.from_pretrained("gpt2-xl", device_map='auto')
        self.model_to_gpu()
        self._tokenizer = AutoModelForCausalLM.from_pretrained("gpt2-xl")
