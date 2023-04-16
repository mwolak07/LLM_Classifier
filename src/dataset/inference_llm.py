



class InferenceLLM(ABC):
    """
    Abstract class, representing an interface for LLMs which we can use for inference when generating the dataset.

    Attributes:
        _temperature: (class attribute) The temperature for the LLM.
        _tokenizer: The model we will use to transform the input strings into vectors.
        _model: The model we will use to perform inference.
    """
    _temperature: str = 0.9
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self, model_name: str, run_on: str = 'cpu', torch_dtype: type = float32):
        """
        Initializes the model and tokenizer with the appropriate parameters for inference.

        Args:
            model_name: The name of the model to use. (ie. EleutherAI/gpt-neo-1.3B).
            run_on: The device to run the model on. Can be 'cpu' or 'gpu'.
            torch_dtype: The floating point type to use in the model. float16 can be specified for lower RAM use.
        """
        self.check_ram()
        self._model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=float16)
        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.model_to_gpu()

    @staticmethod
    def check_ram(min_ram_gb: float) -> None:
        """
        Check if there is not enough RAM to initialize the model.

        Args:
            min_ram_gb: The minimum amount of RAM needed.

        Raises:
            RuntimeError if there is not enough RAM.
        """
        if get_ram_gb() < min_ram_gb:
            raise RuntimeError(f'Need at least {min_ram_gb} GB of RAM to initialize model! '
                               f'You have {get_ram_gb()} GB!')

    def model_to_gpu(self) -> None:
        """
        Tries to assign the current model to the gpu.
        """
        if torch.cuda.is_available():
            self._model.to(torch.device('cuda'))
        else:
            print('WARNING! Could not move model to GPU. Inference will run on CPU.')
        
    def answer(self, question: str) -> str:
        """
        Generates an answer based on the given question.

        Args:
            question: The question the LLM should respond to.

        Returns:
            The answer given by the LLM.
        """
        # Tokenizing the input.
        input_token_ids = self._tokenizer(question, return_tensors="pt").input_ids
        # Moving the input IDs to GPU if possible.
        if torch.cuda.is_available():
            input_token_ids = input_token_ids.to(torch.device('cuda'))
        # Performing the inference.
        output_tokens = self._model.generate(
            input_token_ids,
            do_sample=True,
            temperature=self._temperature,
            max_length=len(question),
        )
        # Getting text back from the tokenized output.
        answer = self._tokenizer.batch_decode(output_tokens)[0]
        return answer

    def answers(self, questions: List[str]) -> List[str]:
        """
        Generates a list of answers based on the given list of questions.

        Args:
            questions: The list of questions the LLM should respond to.

        Returns:
            The answers given by the LLM.
        """
        # Tokenizing the inputs.
        input_token_ids = self._tokenizer(questions, return_tensors='pt').input_ids
        # Moving the input IDs to GPU if possible.
        if torch.cuda.is_available():
            input_token_ids = input_token_ids.to(torch.device('cuda'))
        # Performing the batch inference.
        output_tokens = self._model.generate(
            input_token_ids,
            do_sample=True,
            temperature=self._temperature
        )
        # Getting text back from the tokenized output.
        answers = self._tokenizer.batch_decode(output_tokens)
        return answers
