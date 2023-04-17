# LLM Classifier 2023

This repo contains the code for a large language model (LLM) output classification project. The goal is to investigate ways of classifying if a piece of text comes from a human, or a large language model. This is our final project for CS 4400 at NEU for the spring 2023 semester.

## Commands:

This is a list of useful commands. These are to be executed from the project root directory.
- Generating env requirements:
  - `conda env export > environment.yml`
- Installing from generated env requirements:
  - `conda env update --file environment.yml`
- Generating the databases in /data/llm_classifier:
  - `python src/datasets/generate_datasets.py`
- Generating the test prompts:
  - `python test/datasets/generate_test_questions.py`
- Downloading pretrained huggingface models:
  - `python src/models/download_huggingface_models.py`

## Environment:

The specs for the env are stored in environment.yml. This makes it easy to use with conda. If you would like to install
everything manually, the command is:
- `conda install python=3.9 pytorch=2.0 torchvision torchaudio pytorch-cuda=11.8 tensorflow=2.9 transformers 
huggingface_hub tokenizers pytorch-lightning matplotlib numpy pandas nltk scikit-learn gensim -c pytorch -c nvidia 
-c HuggingFace -c anaconda -c conda-forge`

## Refences:
@software{gpt-neox-library,
  title = {{GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch}},
  author = {Andonian, Alex and Anthony, Quentin and Biderman, Stella and Black, Sid and Gali, Preetham and Gao, Leo and Hallahan, Eric and Levy-Kramer, Josh and Leahy, Connor and Nestler, Lucas and Parker, Kip and Pieler, Michael and Purohit, Shivanshu and Songz, Tri and Phil, Wang and Weinbach, Samuel},
  url = {https://www.github.com/eleutherai/gpt-neox},
  doi = {10.5281/zenodo.5879544},
  month = {8},
  year = {2021},
  version = {0.0.1},
}

@inproceedings{gpt-neox-20b,
  title={{GPT-NeoX-20B}: An Open-Source Autoregressive Language Model},
  author={Black, Sid and Biderman, Stella and Hallahan, Eric and Anthony, Quentin and Gao, Leo and Golding, Laurence and He, Horace and Leahy, Connor and McDonell, Kyle and Phang, Jason and Pieler, Michael and Prashanth, USVSN Sai and Purohit, Shivanshu and Reynolds, Laria and Tow, Jonathan and Wang, Ben and Weinbach, Samuel},
  booktitle={Proceedings of the ACL Workshop on Challenges \& Perspectives in Creating Large Language Models},
  url={https://arxiv.org/abs/2204.06745},
  year={2022}
}

@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
