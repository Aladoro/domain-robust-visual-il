import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
from matplotlib.ticker import MaxNLocator

from plot import cum_max_array

LINE_MARKERS = ['o', 's', 'D', '^', 'v', '*', '.', ',']


def subplot_data(data, ax, value="mean", title=None, min_score=None, max_score=None):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    if min_score is not None and max_score is not None:
        scale = float(max_score) - float(min_score)
        data[value] = (data[value] - float(min_score))/scale
        min_data = np.min(data[value])
        max_data = np.max(data[value])
        min_data_q = np.round(min_data / 0.2)*0.2
        max_data_q = np.round(max_data / 0.2)*0.2
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition", ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    if title is not None:
        plt.title(title)
    if min_score is not None and max_score is not None:
        ax.set_yticks(np.arange(min_data_q, max_data_q+0.2, 0.2))
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
                print('drop')
                experiment_data = experiment_data[1:]
            if cumulative:
                for col in experiment_data.columns:
                    if col != 'Iteration':
                        experiment_data[col] = cum_max_array(experiment_data[col])
            n_data_points = len(experiment_data.index)
            assert n_data_points <= epochs, "The maximum number of timesteps specified is less than data length"
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
            new_exp_data = pd.DataFrame(np.concatenate([experiment_data.values,
                                                        np.repeat(experiment_data.tail(1).values,
                                                                  epochs - n_data_points, axis=0)]))
            new_exp_data.columns = experiment_data.columns
            new_exp_data['Iteration'] = (np.arange(epochs)+1).astype('int32')
            datasets.append(new_exp_data)
            unit += 1
    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot data from multiple experiments on the same axis.')
    parser.add_argument('folders', nargs='*', help='List of directories containing the data logs for the different '
                                                   'experiments.')
    parser.add_argument('--logdir', nargs='*', help='Names of sub-directories containing the experiment data logs for '
                                                    'the different tested algorithms in each experiment.')
    parser.add_argument('--legend', nargs='*', help='Names of the different tested algorithms for plot legend.')
    parser.add_argument('--rows', default=2, help='Number of subplot rows, needs to be consistent with number of '
                                                  'experiments.')
    parser.add_argument('--columns', default=4, help='Number of subplot columns, needs to be consistent with number of '
                                                     'experiments.')
    parser.add_argument('--value', default='mean', help='Value to plot.')
    parser.add_argument('--xaxis', default='Epoch', help='X-axis label.')
    parser.add_argument('--yaxis', default='Reward', help='Y-axis label.')
    parser.add_argument('--title', default=None, help='Plot title.')
    parser.add_argument('--show_random', action='store_true', help='Show the performance at timestep 0 with '
                                                                   'randomly initialized agent.')
    parser.add_argument('--epochs', default=100, help='Number of epochs to plot.')
    parser.add_argument('--min_scores', nargs='*', help='List of minimum scores for normalization in each experiment, '
                                                        'usually score of random behavior.')
    parser.add_argument('--max_scores', nargs='*', help='List of maximum scores for normalization in each experiment, '
                                                        'usually score of expert behavior.')
    parser.add_argument('--cumulative', action='store_true', help='Plot the maximum cumulative score at each epoch.')
    parser.add_argument('--long', action='store_true', help='Stretch plots to aid visualization.')
    parser.add_argument('--show_legend', action='store_true', help='Display legend.')

    args = parser.parse_args()
    rows = int(args.rows)
    columns = int(args.columns)

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    assert len(args.folders) == rows*columns, \
        "The specified number of rows and columns must match the number of " \
        "experiments folders"
    if isinstance(args.min_scores, list) and isinstance(args.max_scores, list):
        if len(args.min_scores) == 1 and len(args.max_scores) == 1:
            min_scores = [args.min_scores[0] for i in range(rows * columns)]
            max_scores = [args.max_scores[0] for i in range(rows * columns)]
        else:
            assert len(args.min_scores) == rows*columns, \
                'min_scores must be the same length as the # of graphs'
            assert len(args.max_scores) == rows * columns, \
                'max_scores must be the same length as the # of graphs'
            min_scores = args.min_scores
            max_scores = args.max_scores
    else:
        min_scores = [args.min_scores for i in range(rows*columns)]
        max_scores = [args.max_scores for i in range(rows*columns)]

    sns.set(style="darkgrid", font_scale=1)
    if args.long:
        fig, axs = plt.subplots(rows, columns, figsize=(5 * columns, 3 * rows),
         sharex=True, sharey=True, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    else:
        fig, axs = plt.subplots(rows, columns, figsize=(4*columns, 4*rows),
        sharex=True, sharey=True, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    for ax, folder, min_score, max_score in zip(axs.flatten(), args.folders, min_scores, max_scores):
        data = []
        if use_legend:
            for logdir, legend_title in zip(args.logdir, args.legend):
                data += get_datasets(osp.join(folder, logdir), legend_title,
                                     show_random=args.show_random,
                                     epochs=int(args.epochs),
                                     cumulative=args.cumulative)
        else:
            for logdir in args.logdir:
                data += get_datasets(osp.join(folder, logdir),
                                     show_random=args.show_random,
                                     epochs=int(args.epochs),
                                     cumulative=args.cumulative)

        ax = subplot_data(data, ax=ax, value=args.value, title=args.title, min_score=min_score,
                          max_score=max_score)
        for i, line in enumerate(ax.lines):
            line.set_marker(LINE_MARKERS[i])

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
