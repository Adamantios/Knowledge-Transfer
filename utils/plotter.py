from itertools import cycle
from os.path import join
from typing import List, Optional, Dict

import matplotlib.pyplot as plt


def plot_results(results: List[Dict], save_folder: Optional[str]) -> None:
    """
    Plots the KT results.

    :param results: the results for each one of the KT method applied.
    :param save_folder: the folder in which the plots will be saved.
    """
    # Plot every metric result for every KT method.
    for result in results:
        # Do not plot teacher, since we don't have training history.
        if result['method'] != 'Teacher':

            for metric, history in result['history'].items():
                # Create subplot for current metric.
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.plot(history)
                ax.set_title(result['method'], fontsize='x-large')
                ax.set_xlabel('epoch', fontsize='large')
                ax.set_ylabel(metric, fontsize='large')
                fig.show()

                if save_folder is not None:
                    filepath = join(save_folder, result['method'] + '_' + metric + '_vs_epoch' + '.png')
                    fig.savefig(filepath)

    # Plot KT methods comparison for each metric.
    linestyles = ['-', '-.', ':']
    linestyles_pool = cycle(linestyles)
    for metric_index, metric in enumerate(results[0]['history'].keys()):
        # Create subplot for overall KT methods comparison for the current metric.
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title('KT Methods Comparison', fontsize='x-large')
        ax.set_xlabel('epoch', fontsize='large')
        ax.set_ylabel(metric, fontsize='large')
        # For every method.
        for result in results:
            if result['method'] == 'Teacher':
                # Plot teacher baseline.
                ax.plot(result['evaluation'], label=result['method'], linestyle='--')
            else:
                # Plot method's current metric results.
                ax.plot(result['history'].values()[metric_index], label=result['method'],
                        linestyle=next(linestyles_pool))

        ax.legend(loc='best', fontsize='large')
        fig.show()
        if save_folder is not None:
            filepath = join(save_folder, 'KT_Methods_Comparison_' + metric + '_vs_epoch' + '.png')
            fig.savefig(filepath)
