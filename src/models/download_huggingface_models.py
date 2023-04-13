from transformers import AutoTokenizer, AutoModelForCausalLM


def download_huggingface_models() -> None:
    """
    Downloads the pretrained huggingface models used in this project. These models are:
    - gpt-neox-20b
    - gpt-j-6b
    - gpt-neo-2.7B
    - gpt-neo-1.3B
    - gpt-neo-125m
    """
    # gpt-neox-20b
    AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
    # gpt-j-6b
    AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
    # gpt-neo-2.7B
    AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # gpt-neo-1.3B
    AutoTokenizer.from_pretrained("El")



if __name__ == '__main__':
    download_huggingface_models()
