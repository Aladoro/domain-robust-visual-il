# Domain-Robust Visual Imitation Learning with Mutual Information Constraints

This repository contains research code for the *ICLR 2021* paper [*Domain-Robust Visual Imitation Learning with Mutual Information Constraints*](https://sites.google.com/view/disentangail/).

## Requirements

1) To replicate the experiments in this project, you need to install the Mujoco
simulation software with a valid license. You can find instructions [here](https://github.com/openai/mujoco-py).

2) The rest of the requirements can be installed with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html),
by utilizing the provided environment file:
```setup
conda env create -f environment.yml
conda activate autonomous_imitation
```

## Collect data/Training

We explain how to collect expert/prior data and perform *observational* imitation to
replicate the paper's experiments in the notebook *training_notebook.ipynb*.
This notebook can be accessed via executing `jupyter notebook` after activating the *conda* environment.

## Plotting

To reproduce the plots, we provide two functions to process and display the results collected:

* plot.py - to obtain a plot of multiple algorithms for a single *observational* imitation problem.

* multi_plot.py - to compare plots for multiple *observational* imitation problems partially sharing the same axis.

e.g. to plot the results for *1-Linked Inverted Pendulum* -> *1-Linked Colored Inverted Pendulum*:
```setup
python plot.py experiments_data/InvertedPendulum_to_colored --min_score 5 --max_score 50 --logdir . --epochs 20 --cumulative --show_legend --yaxis 'Scaled cumulative reward'
```

For further details on their usage, please run:
```setup
python plot.py -h
```

```setup
python multi_plot.py -h
```

## Citation

```
@inproceedings{cetin2021domainrobust,
               title={Domain-Robust Visual Imitation Learning with Mutual Information Constraints},
               author={Edoardo Cetin and Oya Celiktutan},
               booktitle={International Conference on Learning Representations},
               year={2021},
               url={https://openreview.net/forum?id=QubpWYfdNry}
}
```
