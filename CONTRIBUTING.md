# Contributing Guide

Thank you for considering contributing to this repository! We appreciate all efforts that make this repository a great tool. There are many ways to contribute:

- Submitting bug reports or feature requests.
- Writing tutorials or blog posts, improving the documentation.
- Developing features which can be incorporated into this repository.

## Workflow

We adopt the workflow derived from [GitFLow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) with multiple branches:

- `main` serves as the most stable branch. It will not be updated frequently, hence, it may not include the latest projects or support the latest features.
- `develop/main` serves as the convergence of other developing branches, and is treated as the candidate for `main`.
- `develop/<project_name>` stands for a long-term ongoing project, which will be merged to `develop/main` after fully tested. To enable the latest features, developing branches should consider **rebasing on `develop/main` regularly**. Each develop branch can maintain their own feature branches.
- `feature/<feature_name>` stands for a short-term feature used by one or shared by many projects, which will be merged to `develop/<project_name>` or `develop/main` (depending on its audience) after fully tested. Ideally, feature branches should be **mutually exclusive with each other**.
- `hotfix/<bug_name>` stands for a quick fix of bugs existing in any branches. Bug-fix branches will **be reviewed with the first priority**.

## Code Review

Any branch merging will ONLY be accepted after review, which includes the check on its rationality, necessity, implementation logic, and the coding style.

## Developing Tools

Several tools are recommended to facilitate the developing, which can be installed via

```shell
pip install -r requirements/develop.txt
```

## Coding Style

We adopt coding style derived from [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). A recommended `.pylintrc` is provided. Please analyze the code with

```shell
pylint $(git ls-files '*.py')
```

**NOTE**: Any lint warning or error is unacceptable.

## File Structure

This repository is organized as follows:

```shell
./
├── configs/                    # Training configurations of all supported projects.
├── converters/                 # Tools to convert models released by other repositories.
├── datasets/                   # Datasets, defining how each data entry should be processed.
│   ├── data_loaders/           # Tools to group data entries into batches.
│   ├── file_readers/           # Tools to read data from disk.
│   └── transformations/        # Tools to transform data.
│       └── utils/              # Utilities used for data transformation.
├── docs/                       # Documentations.
├── metrics/                    # Evaluation metrics.
├── models/                     # Deep models.
│   └── utils/                  # Utilities used to build deep models.
├── requirements/               # Environment (mainly Python) requirements.
├── runners/                    # Running workflows (i.e., training procedures).
│   ├── augmentations/          # Augmentation pipelines used in training.
│   ├── controllers/            # Tools to control the training process.
│   ├── losses/                 # Loss definitions, each of which is for a particular project.
│   └── utils/                  # Utilities to facilitate the running process.
├── scripts/                    # Job launching / testing scripts.
│   └── training_demos/         # Demo training scripts for reproduction.
├── third_party/                # Third-party dependencies.
├── utils/                      # Utility functions.
│   ├── file_transmitters/      # Tools to transmit files across file systems.
│   ├── loggers/                # Tools for logging.
│   └── visualizers/            # Tools to visualizing results.
├── .gitignore                  # List of files without version control.
├── .gitmodules                 # List of git submodules.
├── .pylintrc                   # Pylint configuration, which helps check the coding style.
├── LICENSE                     # License.
├── CONTRIBUTING.md             # Contributing Guide.
├── README.md                   # Repository description.
├── convert_model.py            # Main entry for model conversion, regarding `./converters/`.
├── dump_command_args.py        # Main entry to export available arguments of all projects, regarding `./configs/`.
├── prepare_dataset.py          # Main entry for data preparation.
├── test_metrics.py             # Main entry for metric evaluation, regarding `./metrics/`.
├── train.py                    # Main entry for model training, regarding `./runners/`.
└── unit_tests.py               # Main entry for unit tests, regarding `./models/` and `./utils/`.
```

To develop a new project (or a new approach), a `runner` (which defines the training procedure, e.g., the data forwarding flow, the optimizing order of various loss terms, how the model(s) should be updated, etc.), a `loss` (which defines the computation of each loss term), and a `config` (which collects all configurations used in the project) are necessary. You may also need to design your own `dataset` (including data `transformation`), `model` structure, evaluation `metric`, `augmentation` pipeline, and running `controller` if they are not supported yet. **NOTE:** All these modules are almost independent from each other. Hence, once a new feature (e.g., `dataset`, `model`, `metric`, `augmentation`, or `controller`) is developed, it can be shared to others with minor effort. It you are interested in sharing your work, we really appreciate your contribution to these modules.
