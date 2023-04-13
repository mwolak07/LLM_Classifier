from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset_llms import InferenceLLM


class InferenceGPTNeo27B(InferenceLLM):
    """
    Represents a GPT Neo-2.7B instance that we can use for inference.

    Attributes:
        _temperature: (class attribute) The temperature for GPT Neo-2.7B.
        _model: The model we will use to perform inference.
        _tokenizer: The model we will use to transform the input strings into vectors.
    """

    def __init__(self):
        """
        Initializes the model and tokenizer with the appropriate parameters for inference.
        """
        self._model = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", device_map='auto')
        self.model_to_gpu()
        self._tokenizer = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
