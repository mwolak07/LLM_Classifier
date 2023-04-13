from transformers import AutoTokenizer, AutoModelForCausalLM
import time


def download_huggingface_model(model_path: str) -> None:
    """
    Downloads a pretrained huggingface model using the given model path.

    Args:
        model_path: The huggingface path to the model.
    """
    print(f'Downloading {model_path}...')
    t = time.time()
    AutoTokenizer.from_pretrained(model_path)
    AutoModelForCausalLM.from_pretrained(model_path)
    print(f'Done in {time.time() - t}s')


def download_huggingface_models() -> None:
    """
    Downloads the pretrained huggingface models used in this project. These models are:
    - gpt-neox-20b | 20B params
    - gpt-j-6b     | 6B  params
    - gpt-neo-2.7B | 2.7B params
    - gpt2-xl      | 1.5B params
    - gpt-neo-1.3B | 1.3B params
    - gpt2-large   | 774M params
    - gpt2-medium  | 355M params
    - gpt-neo-125m | 125M params
    - gpt2         | 124M params
    """
    download_huggingface_model('EleutherAI/gpt-neox-20b')
    download_huggingface_model('EleutherAI/gpt-j-6b')
    download_huggingface_model('EleutherAI/gpt-neo-2.7B')
    download_huggingface_model('gpt2-xl')
    download_huggingface_model('EleutherAI/gpt-neo-1.3B')
    download_huggingface_model('gpt2-large')
    download_huggingface_model('gpt2-medium')
    download_huggingface_model('EleutherAI/gpt-neo-125m')
    download_huggingface_model('gpt2')


if __name__ == '__main__':
    download_huggingface_models()
