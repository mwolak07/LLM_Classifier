# LLM Classifier
This repo contains the code for a large language model (LLM) output classification project. The goal is to investigate ways of classifying if a piece of text comes from a human, or a large language model. This is our final project for CS 4400 at NEU for the spring 2023 semester.

## Viewing the docs:
Open the index.html file for the docs. This is under: docs/_build/html/index.html

## Commands:
This is a list of useful commands. These are to be executed from the project root directory.
- Generating env requirements:
  - `$ conda env export > environment.yml`
- Installing from generated env requirements:
  - `$ conda env update --file environment.yml`
- Generating the databases in /data/llm_classifier:
  - `$ python src/datasets/generate_datasets.py`
- Generating the mock ms marco dataset:
  - `$ python ./test/datasets/generate_mock_ms_marco_data.py`
- Downloading pretrained huggingface models:
  - `$ python ./llm_classifier/models/download_huggingface_models.py`

## Environment:
The specs for the env are stored in environment.yml. This makes it easy to use with conda. If you would like to install
everything manually, the command is:
- `$ conda install python=3.9 pytorch=2.0 torchvision torchaudio pytorch-cuda=11.8 tensorflow=2.9 sphinx=5 transformers 
  huggingface_hub tokenizers pytorch-lightning matplotlib numpy pandas nltk scikit-learn seaborn gensim psutil 
  sphinx_rtd_theme -c pytorch -c nvidia -c HuggingFace -c anaconda -c conda-forge`
  
## Generating Documentation:
Documentation is generated using sphinx. Below are the steps for how to re-generate the documentation.

If you just want to update the docs after modifying code:
- `$ cd docs`
- `$ make clean html`
- `$ make html`

If you want to re-generate the sphinx files:
- `$ pip install sphinx-rtd-theme`
- `$ cd docs`
- `$ sphinx-quickstart`
  - Separate source and build directories (y/n) [n]: n
  - Project name: LLM-Classifier
  - Author name(s): Mateusz Wolak and Alexander Malakov
  - Project release []: 1.0
  - Project language [en]: en
- Modify conf.py:
  - Add the following lines under "Path setup":
    ```python
    import os
    import sys
    sys.path.insert(0, os.path.abspath('..'))
    ```
  - Add extensions under "General configuration":
    ```python
    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
        'sphinx.ext.napoleon'
    ]
    autodoc_default_options = {
        'private-members': False,
        'special-members': True,
    }
    ```
  - Change html_theme under "Options for HTML output":
    ```python
    html_theme = 'sphinx-rtd-theme'
    ```
- `$ cd ..`    
- `$ sphinx-apidoc -o docs src`
- Modify index.rst:
  - Modify the following lines under "Welcome to LLM-Classifier's documentation!":
    ```text
    .. toctree::
       :maxdepth: 2
       :caption: Contents:

       modules
    ```
- `$ cd docs`
- `$ make html`



## Refences:
@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
