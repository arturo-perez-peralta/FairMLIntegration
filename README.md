# FairMLIntegration

## Code structures and dependencies

### `Processors.ipynb`

This notebook implements multistage and logical processors. The results are stored in `results/sweep` and `results/best`.

### `Figures.ipynb`

This notebook processes the results of the `Processors.ipynb` and produces the visualizations of the paper. The figures are saved in `results/Figures`.

### `utils.py`

Auxiliary functions neccessary for the correct use of `Processors.ipynb`.

### Python Version
The code is designed to work with Python 3.9.18.
It is important to ensure that the correct Python version is installed to avoid compatibility issues.

### Installing Dependencies
To reproduce the environment used in this project, follow these steps:

1. **Ensure Python 3.9.18 is Installed**:
2. **Install Dependencies**:
   - The project dependencies are listed in the `requirements.txt` file.
   - Use the following command to install them:
     ```bash
     pip install -r requirements.txt
     ```
