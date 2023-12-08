# A Comparative Study of Genetic-Based Approaches for Enhanced Hourly Temperature Predictions

[[Report]](/doc/report.pdf)


Nowadays, time series forecasting is widely applied across various domains. However, addressing the challenges of time series forecasting proves difficult due to factors that extend beyond the data. In this context, approaches based on deep learning and metaheuristics emerge to confront the complexities of time series forecasting by harnessing the inherent data. This study examines two strategies for predicting the outdoor temperature in the upcoming hour: a deep learning model, optimized through Genetic Algorithms, and an interpretable alternative employing Genetic Programming. The analysis aims to compare the results attained through these differing methodologies.

## Requirements
Before getting started, make sure you have the following dependencies installed:
torch, and [DEAP](https://deap.readthedocs.io).
To install the DEAP and PyTorch, run the following commands:
```bash
$ pip install deap torch
```

## Usage

### Source Code
The implementation of the proposed approaches can be found in the `src` folder. Explore this directory to understand the code structure and access the underlying logic.

### Notebooks
In the `notebooks` folder, you will find Jupyter notebooks that invoke the implemented routines. These notebooks contain various tests showcasing the results obtained from the Genetic Algorithms-optimized deep learning model and the Genetic Programming-based alternative. Review these notebooks for a comprehensive understanding of the experimental outcomes.

## Dataset
Refer to the [data folder](data) for information on the dataset used in this study. Ensure that you have the necessary data before running the code.

Feel free to explore, modify, and build upon the provided codebase to suit your specific requirements or extend the research in this domain. If you encounter any issues or have questions, please refer to the documentation of the dependencies or reach out to the project contributors for assistance.
