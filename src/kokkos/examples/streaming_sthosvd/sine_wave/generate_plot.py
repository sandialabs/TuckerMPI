"""Generate rank and error plots."""


import numpy as np
import matplotlib.pyplot as plt


def matplotlib_adjust_fontsize(small, medium, large):
    """Adjust font size in matplotlib figures."""
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=large)   # fontsize of the figure title
    # plt.rc('text', usetex=True)         # use LaTeX rendering


def stats_file_name(noise, error):
    """Create stats file name."""
    return 'noise_{:s}_tol_{:s}_stream/stats_stream.txt'.format(noise, error)


def main():
    """Generate rank and error plots."""
    color_list = ['tab:orange', 'tab:olive', 'tab:purple']
    noise_list = ['9e-4', '7e-4', '5e-4']
    error_list = ['1e-3', '2e-3']

    sthosvd_rank1 = [[58, 11],
                     [32, 11],
                     [11, 11]]
    sthosvd_rank2 = [[30, 11],
                     [11, 11],
                     [11, 11]]
    sthosvd_rank3 = [[11, 11],
                     [11, 11],
                     [11, 11]]
    sthosvd_error = [[0.00089855, 0.00089937],
                     [0.00069942, 0.00069951],
                     [0.00049965, 0.00049965]]

    matplotlib_adjust_fontsize(16, 18, 20)

    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(9.6, 7.2))
    for i, (noise, color) in enumerate(zip(noise_list, color_list)):
        label = '$\\eta = ' + noise[0] + ' \\times 10^{' + noise[-2:] + '}$'

        for j, error in enumerate(error_list):
            data = np.loadtxt(stats_file_name(noise, error), skiprows=1)

            size3 = data[:, 2]
            rank1 = data[:, 4]
            rank2 = data[:, 5]
            rank3 = data[:, 6]
            error = data[:, 11]

            ax[0, j].plot(size3, rank1, color=color, linewidth=1, label=label)
            ax[0, j].plot(size3, sthosvd_rank1[i][j] * np.ones_like(size3),
                          linewidth=2, linestyle='dotted', color=color)

            ax[1, j].plot(size3, rank2, color=color, linewidth=1, label=label)
            ax[1, j].plot(size3, sthosvd_rank2[i][j] * np.ones_like(size3),
                          linewidth=2, linestyle='dotted', color=color)

            ax[2, j].plot(size3, rank3, color=color, linewidth=1, label=label)
            ax[2, j].plot(size3, sthosvd_rank3[i][j] * np.ones_like(size3),
                          linewidth=2, linestyle='dotted', color=color)

            ax[3, j].semilogy(size3, error, color=color, linewidth=1,
                              label=label)
            ax[3, j].semilogy(size3, sthosvd_error[i][j] * np.ones_like(size3),
                              linewidth=2, linestyle='dotted', color=color)

    ax[0, 0].set_title('$\\tau = 10^{-3}$', fontsize=16)
    ax[0, 1].set_title('$\\tau = 2 \\times 10^{-3}$', fontsize=16)

    ax[3, 0].set_yticks([5.0e-04, 1.0e-03], minor=True)

    ax[0, 0].legend(bbox_to_anchor=(0, 1.3, 2.35, 0.2),
                    loc='lower left',
                    mode='expand',
                    ncol=3)

    ax[3, 0].set_xlabel('Number of Snapshots')
    ax[3, 1].set_xlabel('Number of Snapshots')

    ax[0, 0].set_ylabel('Rank $R_1$')
    ax[0, 0].yaxis.set_label_coords(-0.3, 0.5)

    ax[1, 0].set_ylabel('Rank $R_2$')
    ax[1, 0].yaxis.set_label_coords(-0.3, 0.5)

    ax[2, 0].set_ylabel('Rank $R_3$')
    ax[2, 0].yaxis.set_label_coords(-0.3, 0.5)

    ax[3, 0].set_ylabel('Rel. Error')
    ax[3, 0].yaxis.set_label_coords(-0.3, 0.5)

    plt.tight_layout()

    fig.savefig('../../../../documents/ISC-2023/figs/noisy_wave.pdf')

    plt.close(fig)


if __name__ == '__main__':
    main()
