from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch import device, cuda, float16
from typing import List
from abc import ABC
import nltk
import json
import time
from src.util import cd_to_executing_file, get_ram_gb, get_vram_gb


# Downloading the nltk sentence tokenizer
nltk.download('punkt')


class InferenceLLM(ABC):
    """
    Represents an interface for LLMs which we can use for inference when generating the dataset. The particular LLM
    used is determined when the object is created.

    Attributes:
        _model_ram_file: (class attribute) The file containing a map of model: RAM Requirement.
        _tokenizer: The model we will use to transform the input strings into vectors.
        _model: The model we will use to perform inference.
    """
    _model_ram_file: str = '../models/model_ram.json'
    _use_gpu: bool
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self, model_name: str, use_gpu: bool = True, torch_dtype: type = float16):
        """
        Initializes the model and tokenizer with the appropriate parameters for inference.

        Args:
            model_name: The name of the model to use. (ie. EleutherAI/gpt-neo-1.3B).
            use_gpu: True if we want to use the GPU (if available), False if we do not.
            torch_dtype: The floating point type to use in the model. float32 can be specified for better performance
                         at the cost of higher RAM use.
        """
        # Ensuring we are in the correct directory.
        cd_to_executing_file(__file__)
        # Checking if the gpu is available.
        if use_gpu and cuda.is_available():
            self._use_gpu = True
        elif use_gpu and not cuda.is_available():
            print('WARNING: Unable to use GPU, CUDA is not available.')
            self._use_gpu = False
        else:
            self._use_gpu = False
        # Checking if we have enough memory.
        self.check_ram(self._model_ram_file, model_name, self._use_gpu)
        # Instantiating the model based on model_name and torch_dtype.
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        # Moving the model to the GPU if necessary.
        self._model = self._model.to(device('cuda')) if self._use_gpu else self._model
        # Instantiating the tokenizer associated with the model.
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def check_ram(model_ram_file: str, model_name: str, use_gpu: bool) -> None:
        """
        Check if there is not enough RAM to initialize the model, based on the amount of RAM needed for each model and
        the device we will be running on.

        Args:
            model_ram_file: The file containing the minimum amounts of RAM needed.
            model_name: The name of the LLM model we will be using for lookup in the file.
            use_gpu: True if we are using the GPU (check VRAM instead of RAM)

        Raises:
            RuntimeError if there is not enough RAM.
        """
        with open(model_ram_file, 'r') as f:
            model_ram_data = json.load(f)
        ram = get_ram_gb()
        vram = get_vram_gb()
        model_ram = model_ram_data[model_name]
        if use_gpu and vram < model_ram:
            raise RuntimeError(f'Need at least {model_ram} GB of VRAM to initialize model! '
                               f'You have {min(ram, vram)} GB!')
        elif ram < model_ram:
            raise RuntimeError(f'Need at least {model_ram} GB of RAM to initialize model! '
                               f'You have {ram} GB!')

    @staticmethod
    def postprocess_answer(answer: str) -> str:
        """
        Post-processes an answer. This involves removing the last sentence if it cut off part way through, and removing
        any repeated sentences at the end of the output.
        """
        # Removing the context prompt.
        answer = answer.split('The answer, in complete sentences, to the question:')[1].split(', is:')[1]
        # Getting the sentences.
        sentences = nltk.tokenize.sent_tokenize(answer)
        if len(sentences) > 1:
            # Removing the last sentence if it does not end in correct punctuation (cut off answer).
            if sentences[-1][-1] not in '.!?':
                sentences = sentences[:-1]
        else:
            # Adding a period to a partial answer if there was only 1 sentence.
            if sentences[-1][-1] not in '.!?':
                sentences[-1].append('.')
        # Looking for repeats one after the other, and removing them.
        last_sentence = ''
        for i in range(len(sentences)):
            sentence = sentences[i]
            if sentence == last_sentence:
                sentences.pop(i)
            last_sentence = sentence
        # Concatenating the result and returning it.
        answer = ''.join(sentences)
        # Add spaces after punctuation, but not after the last one.
        answer = answer.replace('.', '. ').replace('!', '! ').replace('?', '? ')
        answer = answer[:-1]
        return answer

    def answers(self, questions: List[str], max_answer_len: int) -> List[str]:
        """
        Generates a list of answers based on the given list of questions.

        Args:
            questions: The list of questions the LLM should respond to.
            max_answer_len: The maximum length of an answer.

        Returns:
            The answers given by the LLM.
        """
        answers = []
        for i in range(len(questions)):
            t = time.time()
            answers.append(self.answer(questions[i], max_answer_len))
            print(f'Generated sample {i}/{len(questions)} in {time.time() - t}s')

    def answer(self, question: str, max_answer_len: int) -> str:
        """
        Generates an answer based on the given question.

        Args:
            question: The question the LLM should respond to.
            max_answer_len: The maximum length of an answer.

        Returns:
            The answer given by the LLM.
        """
        # Encoding the input.
        encoded_input = self._tokenizer(question, return_tensors="pt", return_attention_mask=True)
        # Unpacking the input and moving tensors to the GPU if needed.
        input_ids = encoded_input.input_ids.to(device('cuda')) if self._use_gpu \
            else encoded_input.input_ids
        attention_mask = encoded_input.attention_mask.to(device('cuda')) if self._use_gpu \
            else encoded_input.attention_mask
        # Performing the inference.
        encoded_output = self._model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=max_answer_len,
            attention_mask=attention_mask
        )
        # Getting text back from the tokenized output.
        answer = self._tokenizer.batch_decode(encoded_output)[0]
        # Post-processing the answer.
        processed_answer = self.postprocess_answer(answer)
        return processed_answer
