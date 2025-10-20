# FairMLIntegration

## Code structures and dependencies

### `Processors.ipynb`

This notebook implements multistage and logical processors. The results are stored in `results/sweep` and `results/best`.

### `Figures.ipynb`

This notebook processes the results of the `Processors.ipynb` and produces the visualizations of the paper. The figures are saved in `results/Figures`.

### `SimulationStudy.py`

Performs 50 instances of the Simulation Study.

### `MassiveSimulationFigures.ipynb`

Similar to `Figures.ipynb`. Handles the aggregated data of 50 simulations.

### `utils.py`

Auxiliary functions neccessary for the correct use of `Processors.ipynb`.

### Python Version
The code is designed to work with **Python 3.9.18**.
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
### Cite this work
```
@misc{pérezperalta2025merrierlogicalmultistageprocessors,
      title={The more the merrier: logical and multistage processors in credit scoring}, 
      author={Arturo Pérez-Peralta and Sandra Benítez-Peña and Rosa E. Lillo},
      year={2025},
      eprint={2503.23979},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2503.23979}, 
}
```
