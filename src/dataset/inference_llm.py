from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch import device, cuda, float16
from typing import List, Tuple
from numpy import ndarray
from abc import ABC
import numpy as np
import json
import time
from src.util import cd_to_executing_file, get_ram_gb, get_vram_gb


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

    def init_extra_steps(self, model_name: str) -> None:
        """
        Makes extra adjustments to the initialization of the model as needed for each model.
        """
        if model_name in ['bigscience/bloom-1b1']:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if model_name in ['EleutherAI/gpt-neo-1.3B']:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

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

        Args:
            answer: The answer we are post-processing.

        Returns:
            The post-processed answer.
        """
        # If we got an empty string as our answer, return it.
        if answer == '' or answer is None:
            return ''
        # Removing the context prompt.
        no_context = answer.split('The short answer, in complete sentences, to the question:')[1]
        answer = no_context.split(', is:')[1]
        # Getting the sentences.
        sentences = InferenceLLM.get_sentences(answer)
        # If we have no sentences, return ''.
        if len(sentences) == 0:
            return ''
        # If we started with punctuation, remove it.
        if sentences[0] in '.!?':
            sentences = sentences[1:]
        # We got multiple sentences.
        if len(sentences) > 1:
            # Removing the last sentence if it does not end in correct punctuation (cut off answer).
            if sentences[-1][-1] not in '.!?':
                sentences = sentences[:-1]
        # We only have one sentence, so do not remove it.
        elif len(sentences) == 1:
            # Adding a period to a partial answer if there was only 1 sentence.
            if sentences[-1][-1] not in '.!?':
                sentences[-1] += '.'
        # Looking for repeats one after the other, and removing them.
        last_sentence = ''
        new_sentences = []
        for sentence in sentences:
            # Skip repeats, unless it is punctuation.
            if sentence != last_sentence or sentence in '.!?':
                new_sentences.append(sentence)
            last_sentence = sentence
        sentences = new_sentences
        # Concatenating the result and returning it.
        answer = ''.join(sentences)
        # Add spaces after punctuation.
        answer = answer.replace('.', '. ').replace('!', '! ').replace('?', '? ')
        # Remove spaces between double punctuation.
        while '. .' in answer or '! !' in answer or '? ?' in answer:
            answer = answer.replace('. .', '..').replace('! !', '!!').replace('? ?', '??')
        # Remove space after the last bit of punctuation.
        answer = answer[:-1]
        return answer

    @staticmethod
    def get_sentences(text: str) -> List[str]:
        """
        Splits a string into sentences, using punctuation: .!?
        Keeps the punctuation at the end of each sentence.
        """
        # Processing the text with a stack.
        text_stack = [text]
        # Iterating through each punctuation mark.
        for p in ['.', '!', '?']:
            # Going through each item on the text stack, for this punctuation mark, and creating a new text stack for
            # the next punctuation mark.
            new_text_stack = []
            for t in text_stack:
                # Splitting t on the punctuation mark and removing whitespace.
                p_split = [sentence.strip() for sentence in t.split(p)]
                p_list = [p] * (len(p_split) - 1)
                # text ended with p. We have to remove the trailing '', and add the punctuation back in with zip.
                if p_split[-1] == '':
                    p_split = p_split[:-1]
                    new_text_stack += [sentence + punctuation for (sentence, punctuation) in zip(p_split, p_list)]
                # text cut off. We have to add the punctuation back in with zip, and add the cut off text back on,
                # because zip() will get rid of the last thing in p_split.
                else:
                    last_sentence = p_split[-1]
                    new_text_stack += [sentence + punctuation for (sentence, punctuation) in zip(p_split, p_list)]
                    new_text_stack.append(last_sentence)
            # Loading the text stack for the next punctuation mark,
            text_stack = new_text_stack
        return text_stack

    def answer(self, question: str, max_answer_len: int) -> Tuple[str, int]:
        """
        Generates an answer based on the given question. Guards against CUDA errors, and will try to get the answer
        again up to three times if it gets no response from the LLM.

        Args:
            question: The question the LLM should respond to.
            max_answer_len: The maximum length of an answer.

        Returns:
            The answer given by the LLM, and the number of tries it took.
        """
        answers, tries = self.answer_batch(np.array([question]), np.array(['']), max_answer_len)
        return answers[0], tries

    def answers(self, questions: List[str], max_answer_len: int, batch_size: int = 1) -> Tuple[List[str], List[int]]:
        """
        Generates a list of answers based on the given list of questions.

        Args:
            questions: The list of questions the LLM should respond to.
            max_answer_len: The maximum length of an answer.
            batch_size: The size of the batches of questions to ask the LLM.

        Returns:
            The answers given by the LLM, and the number of tries each answer took.
        """
        print('\nAnswers:')
        print(f'Num questions: {len(questions)}')
        answers = np.full((len(questions),), '')
        tries_list = np.full((len(questions),), -1)
        questions = np.array(questions)
        question_batches = self.get_batches(questions, batch_size)
        print(f'Batch lengths: {[len(batch) for batch in question_batches]}')
        answer_index = 0
        for i in range(len(question_batches)):
            t = time.time()
            question_batch = question_batches[i]
            answer_batch = np.full((len(question_batch),), '')
            answer_batch, tries = self.answer_batch(question_batch, answer_batch, max_answer_len)
            for answer in answer_batch:
                answers[answer_index] = answer
                tries_list[answer_index] = tries
                answer_index += 1
            print(f'Generated batch {i + 1}/{len(question_batches)} in {time.time() - t}s '
                  f'{(i + 1) / len(question_batches) * 100}%)')
        print(f'Num answers: {len(answers)}')
        return answers.tolist(), tries_list.tolist()

    @staticmethod
    def get_batches(questions: ndarray[str], batch_size: int) -> ndarray[ndarray[str]]:
        """
        Splits the questions into batches of size batch_size for batch inference, and converts to numpy arrays.

        Args:
            questions: The list of questions to split into batches.
            batch_size: The size of each batch of questions we want to be generating.

        Returns:
            A list of lists of questions, representing batches for the model, as a numpy array.
        """
        batches = np.array_split(questions, np.arange(batch_size, len(questions), batch_size))
        return np.array([np.array(batch) for batch in batches])

    def answer_batch(self, question_batch: ndarray[str], answer_batch: ndarray[str], max_answer_len: int,
                     _current_try: int = 0, _max_try: int = 3) -> Tuple[ndarray[str], int]:
        """
        Generates an batch of answers based on a batch of questions. Guards against CUDA errors, and will try to get the
        answer again up to three times if it gets no response from the LLM

        Args:
            question_batch: The batch of questions the LLM should respond to.
            answer_batch: The batch of answers this function will fill in. Should start out being filled in with empty
                          strings: ''
            max_answer_len: The maximum length of an answer.
            _current_try: Recursive parameter used to track which try we are currently executing.
            _max_try: Recursive parameter used to specify how many times we should re-try for an answer.

        Returns:
            The answer batch given by the LLM, and the number of tries it took.

        Raises:
            cuda.OutOfMemoryError: When the batch size or data is too large, and does not fit on the GPU.
            cuda.CudaError: When some other error occurs in CUDA.
            RuntimeError: the length of one of the input sequences was longer than the maximum for the model.
            ValueError: the answer_batch and question_batch are not the same dimensions.
        """
        # We ran out of tries. Return the answers we have.
        if _current_try == _max_try:
            return answer_batch, _current_try + 1

        # Checking dimensions
        if len(answer_batch) != len(question_batch):
            raise ValueError(f'Answer and question batch dimensions do not match!: '
                             f'{len(answer_batch)} != {len(question_batch)}')

        # Getting the indices of the questions with no answers, and the questions with no answers.
        empty_indices = np.where(answer_batch == '')
        llm_questions = question_batch[empty_indices]

        # Checking the llm questions, and generating the answers for them. Ensures the answers go to the indices that
        # had the empty elements.
        self.check_question_lengths(llm_questions)
        answer_batch[empty_indices] = self.generate_batch(llm_questions, max_answer_len)

        # Re-trying for empty answers if we got any.
        empty_indices = np.where(answer_batch == '')
        if len(empty_indices[0]) > 0:
            return self.answer_batch(question_batch, answer_batch, max_answer_len, _current_try + 1)
        # Answer is good, return the output we got.
        else:
            return answer_batch, _current_try + 1

    def check_question_lengths(self, questions: ndarray[str]) -> str:
        """
        Checks that the length of each of the questions is not more than the maximum of the tokenizer.
        Note: The output from np.where looks a little confusing, it is a tuple: Tuple[List[indices], dtype].

        Args:
            questions: The batch of questions the LLM should respond to.

        Raises:
            RuntimeError: the length of one of the input sequences was longer than the maximum for the model.
        """
        long_questions = np.where(len(questions) > self._tokenizer.model_max_length)
        if len(long_questions[0]) > 0:
            raise RuntimeError(f'Input sequence is longer than maximum for the model, '
                               f'{len(questions[long_questions[0][0]])} > {self._tokenizer.model_max_length}, '
                               f'question: {questions[long_questions[0][0]]}')

    def generate_batch(self, questions: ndarray[str], max_answer_len: int) -> ndarray[str]:
        """
        Generates a batch of answers from a batch of questions using the LLM.

        Args:
            questions: The batch of questions the LLM should respond to.
            max_answer_len: The maximum length of an answer.

        Raises:
            cuda.OutOfMemoryError: When the batch size or data is too large, and does not fit on the GPU.
            cuda.CudaError: When some other error occurs in CUDA.
        """
        # Converting the questions to a list and tokenizing them.
        questions = questions.tolist()
        encoded_inputs = self._tokenizer(questions, return_tensors="pt", return_attention_mask=True, padding=True)
        # Unpacking the input and moving tensors to the GPU if needed.
        input_ids = encoded_inputs.input_ids.to(device('cuda')) if self._use_gpu \
            else encoded_inputs.input_ids
        attention_mask = encoded_inputs.attention_mask.to(device('cuda')) if self._use_gpu \
            else encoded_inputs.attention_mask
        # Performing the inference.
        encoded_output = self._model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=max_answer_len,
            attention_mask=attention_mask
        )
        # Getting text back from the tokenized output.
        answers = np.array(self._tokenizer.batch_decode(encoded_output))
        # Post-processing the answers.
        postprocess_fn = np.vectorize(self.postprocess_answer)
        return postprocess_fn(answers)
