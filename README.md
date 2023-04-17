# LLM Classifier 2023
This repo contains the code for a large language model (LLM) output classification project. The goal is to investigate ways of classifying if a piece of text comes from a human, or a large language model. This is our final project for CS 4400 at NEU for the spring 2023 semester.

## Commands:
This is a list of useful commands. These are to be executed from the project root directory.
- Generating env requirements:
  - Conda: conda env export > environment.yml
- Installing from generated env requirements:
  - Conda: conda env update --file environment.yml
- Generating the databases in /data/llm_classifier:
  - python3 src/datasets/generate_datasets.py
- Generating the test prompts:
  - python3 test/datasets/generate_test_questions.py
- Downloading pretrained huggingface models:
  - python3 src/models/download_huggingface_models.py

## Refences:
@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
