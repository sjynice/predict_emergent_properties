# Figures

Scripts used for database generation, model training, and principal analyses.

## Database generation

The scripts `NumStable14.py` and `NumStable15.py` are used to generate databases with different distributions of $\mu$, $\gamma$, and $D$.

For Figure 4 and most of the other analyses, the surrogate models were trained on databases generated using `NumStable14.py`.

To perform parallel simulations using SLURM, use the two Bash scripts provided in `BashScript/Generate_Databases/`.

## Model training

Files whose names contain `MD_transpt3` or `MD_transpt5` are used to train models predicting fate diversification ($\mathcal{H}$) and diversity realization ($\mathcal{D}$), respectively.

Version 3 does not require a survival fraction list, whereas version 5 does. Both `NumStable14.py` and `NumStable15.py` in `GenerateDatabase/` can generate databases containing survival fraction lists. You can delete that list to adapt the specific scenarios.

In addition, the following scripts are used to train surrogate models for other emergent properties:

- `NS_transpt3`: Fate multiplicity models.
- `Fl_transpt3`: Persistent fluctuation models.
- `SS_transpt3`: Transition species models.

Some of these scripts require submission through Bash scripts.

## Model saving and usage

Once trained, the models are saved in the `saved_models/` directory.

An automated pipeline is provided in `tools/` to load and use the saved models. Instructions for using this pipeline are included in the corresponding principal analysis scripts.
