# LLM Classifier
This repo contains the code for a large language model (LLM) output classification project. The goal is to investigate ways of classifying if a piece of text comes from a human, or a large language model. This is our final project for CS 4400 at NEU for the spring 2023 semester.

## Directory map:
- data: all the data, large files not uploaded to github.
  - bloom_1_1B: the database files for the Bloom 1.1B generated LLM data.
  - MS_MARCO: the json file for the MS MARCO human data.
  - fasttext: the files for the pre-trained fasttext weights.
- data_analysis: exploratory analysis results.
  - MS_MARCO: exploratory analysis results for MS MARCO.
- docs: The documentation.
  - build
    - html: index.html is here.
- src: all the source code
  - dataset: the dataset creation and processing code.
  - models: the implementations of the various ML models.
  - util: misc. utilities used in dataset, models, and test.
  - logs: the training logs from the NN models.
  - model_results: the results from the training of each model.
  - model_weights: the saved weights for each model. Omitted due to large file size.
- test: all the test code.
  - dataset: the tests for the code in src/dataset
  
## Viewing the docs:
Open the index.html file for the docs. This is under: docs/_build/html/index.html

## Commands:
This is a list of useful commands. These are to be executed from the project root directory.
- Generating env requirements:
  - `$ conda env export > environment.yml`
- Installing from generated env requirements:
  - `$ conda env update --file environment.yml`
- Generating the databases in /data/llm_classifier:
  - `$ python -m src.datasets.generate_datasets`
- Generating the mock ms marco dataset:
  - `$ python -m test.datasets.generate_mock_ms_marco_data.generate_mock_ms_marco_data`
- Downloading pretrained huggingface models:
  - `$ python -m src.models.download_huggingface_models'
- View NN training with tensorboard:
  - `$ tensorboard --logdir=src/logs/<model_name>'
  - Open localhost:6006 in browser

## Environment:
The specs for the env are stored in environment.yml. This makes it easy to use with conda. If you would like to install
everything manually, the command is:
- `$ conda install python=3.9 pytorch=2.0 torchvision torchaudio pytorch-cuda=11.8 transformers=4.28.1 
  huggingface_hub tokenizers pytorch-lightning sphinx=5 matplotlib numpy pandas nltk scikit-learn seaborn gensim psutil 
  sphinx_rtd_theme -c pytorch -c nvidia -c HuggingFace -c anaconda -c conda-forge`
- `$ pip install tensorflow==2.10`
  
## Generating Documentation:
Documentation is generated using sphinx. Below are the steps for how to re-generate the documentation.

If you just want to update the docs after modifying code:
- `$ cd docs`
- `$ make clean html`
- `$ make html`

If you changed the file or directory structure:
- delete docs/src.*
- delete docs/modules.rst  
- `$ sphinx-apidoc -o docs src`
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
