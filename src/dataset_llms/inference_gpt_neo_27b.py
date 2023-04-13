from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset_llms import InferenceLLM


class InferenceGPTNeo27B(InferenceLLM):
    """
    Represents a GPT Neo-2.7B instance that we can use for inference.

    Attributes:
        min_ram_gb: (class attribute) The amount of RAM, in gb, we need to initialize the model.
        _temperature: (class attribute) The temperature for GPT Neo-2.7B.
        _model: The model we will use to perform inference.
        _tokenizer: The model we will use to transform the input strings into vectors.
    """
    min_ram_gb: float = 9.9

    def __init__(self):
        """
        Initializes the model and tokenizer with the appropriate parameters for inference.
        """
        self.check_ram(self.min_ram_gb)
        self._model = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", device_map='auto')
        self.model_to_gpu()
        self._tokenizer = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
