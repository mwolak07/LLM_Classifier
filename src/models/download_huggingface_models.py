from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os


def model_path_in_cache(model_path: str) -> bool:
    """
    Checks if the given model_path has already been downloaded in the huggingface cache.

    Args:
        model_path: The huggingface path to the model.

    Returns:
        Is model_path in the huggingface cache?
    """
    user_path = os.path.expanduser('~')
    cache = os.path.join(user_path, '.cache/huggingface/hub')
    model_folder = 'models--' + model_path.replace('/', '--')
    return model_folder in os.listdir(cache)


def download_huggingface_model(model_path: str) -> None:
    """
    Downloads a pretrained huggingface model using the given model path.

    Args:
        model_path: The huggingface path to the model.
    """
    # Using different tokenizer and model for llama.
    if not model_path_in_cache(model_path):
        print(f'Downloading {model_path}...')
        t = time.time()
        # Model might be too big for memory, we want to continue in this case.
        try:
            AutoTokenizer.from_pretrained(model_path)
            AutoModelForCausalLM.from_pretrained(model_path)
        except RuntimeError as e:
            if 'memory' in str(e):
                print(f'Warning: {model_path} could not be initialized, because there was not enough system RAM!')
            else:
                raise e

        print(f'Done in {time.time() - t}s')
    else:
        print(f'{model_path} already downloaded!')


def download_huggingface_models() -> None:
    """
    Downloads the pretrained huggingface models used in this project. These models are:
    - gpt-neo-2.7B | 2.7B params | 9.9 GB
    - gpt2-xl      | 1.5B params | 6.0 GB
    - bloom-3b     | 3B params   | 5.6 GB
    - opt-2.7b     | 2.7B params | 5.3 GB
    - gpt-neo-1.3B | 1.3B params | 4.9 GB
    - gpt2-large   | 774M params | 3.0 GB
    - opt-1.3b     | 1.3B params | 2.5 GB
    - bloom-1b1    | 1.1B params | 2.0 GB
    - gpt2-medium  | 355M params | 1.4 GB
    - bloom-560m   | 560M params | 1.1 GB
    - opt-350m     | 350M params | 0.6 GB
    - gpt-neo-125m | 125M params | 0.5 GB
    - gpt2         | 124M params | 0.5 GB
    - opt-125m     | 125M params | 0.3 GB
    """
    download_huggingface_model('EleutherAI/gpt-neo-2.7B')
    download_huggingface_model('gpt2-xl')
    download_huggingface_model('bigscience/bloom-3b')
    download_huggingface_model('facebook/opt-2.7b')
    download_huggingface_model('EleutherAI/gpt-neo-1.3B')
    download_huggingface_model('gpt2-large')
    download_huggingface_model('facebook/opt-1.3b')
    download_huggingface_model('bigscience/bloom-1b1')
    download_huggingface_model('gpt2-medium')
    download_huggingface_model('bigscience/bloom-560m')
    download_huggingface_model('facebook/opt-350m')
    download_huggingface_model('EleutherAI/gpt-neo-125m')
    download_huggingface_model('gpt2')
    download_huggingface_model('facebook/opt-125m')


if __name__ == '__main__':
    download_huggingface_models()
