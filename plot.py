import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
from matplotlib.ticker import MaxNLocator

LINE_MARKERS = ['o', 's', 'D', '^', 'v', '*', '.', ',']


def cum_max_array(arr):
    """Return array with the cumulative maximum value at each index."""
    for i in range(arr.values.shape[0] - 1):
        arr.values[i + 1] = np.maximum(arr.values[i + 1], arr.values[i])
    return arr


def subplot_data(data, ax, value="mean", title=None, min_score=None, max_score=None):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    if min_score is not None and max_score is not None:
        scale = float(max_score) - float(min_score)
        data[value] = (data[value] - float(min_score)) / scale
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition", ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    if title is not None:
        plt.title(title)
    if min_score is not None and max_score is not None:
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.get_legend().remove()
    return ax


def get_datasets(fpath, condition=None, show_random=False, epochs=100, cumulative=False):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            if condition is not None:
                exp_name = condition
            else:
                param_path = open(os.path.join(root, 'params.json'))
                params = json.load(param_path)
                exp_name = params['exp']['exp_name']

            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            if not show_random:
                experiment_data = experiment_data[1:]
            if cumulative:
                for col in experiment_data.columns:
                    if col != 'Iteration':
                        experiment_data[col] = cum_max_array(experiment_data[col])
            n_data_points = len(experiment_data.index)
            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                np.tile(exp_name,
                        n_data_points)
            )
            if n_data_points <= epochs:
                new_exp_data = pd.DataFrame(np.concatenate([experiment_data.values,
                                                            np.repeat(experiment_data.tail(1).values,
                                                                      epochs - n_data_points, axis=0)]))
            else:
                new_exp_data = pd.DataFrame(experiment_data.values[:epochs])
            new_exp_data.columns = experiment_data.columns
            new_exp_data['Iteration'] = (np.arange(epochs) + 1).astype('int32')
            datasets.append(new_exp_data)
            unit += 1

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot data from experiments in a single plot.')
    parser.add_argument('folder', default='experiments_data', help='Directory containing all the data logs for an '
                                                                   'experiment.')
    parser.add_argument('--logdir', nargs='*', help='Names of sub-directories containing the experiment data logs for '
                                                    'the different tested algorithms .')
    parser.add_argument('--legend', nargs='*', help='Names of the different tested algorithms for plot legend.')
    parser.add_argument('--value', default='mean', help='Value to plot.')
    parser.add_argument('--xaxis', default='Epoch', help='X-axis label.')
    parser.add_argument('--yaxis', default='Reward', help='Y-axis label.')
    parser.add_argument('--title', default=None, help='Plot title.')
    parser.add_argument('--show_random', action='store_true', help='Show the performance at timestep 0 with '
                                                                   'randomly initialized agent.')
    parser.add_argument('--epochs', default=100, help='Number of epochs to plot.')
    parser.add_argument('--min_score', default=None, help='Minimum score for normalization, usually score of random '
                                                          'behavior.')
    parser.add_argument('--max_score', default=None, help='Maximum score for normalization, usually score of expert '
                                                          'behavior.')
    parser.add_argument('--cumulative', action='store_true', help='Plot the maximum cumulative score at each epoch.')
    parser.add_argument('--show_legend', action='store_true', help='Display legend.')

    args = parser.parse_args()
    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    sns.set(style="darkgrid", font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True,
                           gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(osp.join(args.folder, logdir), legend_title,
                                 show_random=args.show_random,
                                 epochs=int(args.epochs),
                                 cumulative=args.cumulative)
    else:
        for logdir in args.logdir:
            data += get_datasets(osp.join(args.folder, logdir),
                                 show_random=args.show_random,
                                 epochs=int(args.epochs),
                                 cumulative=args.cumulative)

    ax = subplot_data(data, ax=ax, value=args.value, title=args.title, min_score=args.min_score,
                      max_score=args.max_score)
    for i, line in enumerate(ax.lines):
        line.set_marker(LINE_MARKERS[i])
        line.set_markevery(int(args.epochs) // 20)

    handles, labels = ax.get_legend_handles_labels()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', left='off', right='off', top='off', bottom='off')
    plt.grid(False)
    plt.xlabel(args.xaxis)
    plt.ylabel(args.yaxis)
    plt.tight_layout()
    if args.show_legend:
        fig.legend(handles, labels, loc='lower center', mode='expand', ncol=5)
    plt.show()


if __name__ == "__main__":
    main()
