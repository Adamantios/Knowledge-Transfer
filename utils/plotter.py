from itertools import cycle
from os.path import join
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
from numpy import asarray


def plot_results(results: List[Dict], epochs: int, save_folder: Optional[str]) -> None:
    """
    Plots the KT results.

    :param results: the results for each one of the KT method applied.
    :param epochs: the number of epochs the experiment ran for.
    :param save_folder: the folder in which the plots will be saved.
    """
    # Plot every metric result for every KT method.
    for result in results:
        # Do not plot teacher, since we don't have training history.
        if result['method'] != 'Teacher':

            for metric, history in result['history'].items():
                # Plot only validation metric results.
                if metric.startswith('val_'):
                    # Create subplot for current metric.
                    fig, ax = plt.subplots(figsize=(12, 10))
                    ax.plot(history)
                    ax.set_title(result['method'], fontsize='x-large')
                    ax.set_xlabel('epoch', fontsize='large')
                    ax.set_ylabel(metric, fontsize='large')
                    plt.show()

                    if save_folder is not None:
                        filepath = join(save_folder, result['method'] + '_' + metric + '_vs_epoch' + '.png')
                        fig.savefig(filepath)

    # Plot KT methods comparison for each metric.
    # Do not compare for PKT.
    n_methods = 0
    for result in results:
        if result['method'] != 'Teacher' and result['method'] != 'Probabilistic Knowledge Transfer':
            n_methods += 1

    if bool(n_methods):
        linestyles = ['--', '-.', ':']
        i = 0
        for metric_index, metric in enumerate(results[0]['history'].keys()):
            # Plot only validation metric results.
            if metric.startswith('val_') and 'loss' not in metric:
                i += 1
                linestyles_pool = cycle(linestyles)
                # Create subplot for overall KT methods comparison for the current metric.
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.set_title('KT Methods Comparison', fontsize='x-large')
                ax.set_xlabel('epoch', fontsize='large')
                ax.set_ylabel(metric, fontsize='large')
                # For every method.
                for result in results:
                    if result['method'] == 'Teacher':
                        # Plot teacher baseline.
                        baseline = asarray([result['evaluation'][i] for _ in range(epochs)])
                        ax.plot(baseline, label=result['method'], linestyle='-')
                    elif result['method'] == 'Probabilistic Knowledge Transfer':
                        continue
                    else:
                        # Plot method's current metric results.
                        ax.plot(list(result['history'].values())[metric_index], label=result['method'],
                                linestyle=next(linestyles_pool))

                ax.legend(loc='best', fontsize='large')
                plt.show()
                if save_folder is not None:
                    filepath = join(save_folder, 'KT_Methods_Comparison_' + metric + '_vs_epoch' + '.png')
                    fig.savefig(filepath)
