"""Generate streaming STHOSVD config file."""


import argparse
import os


def main(mode_sizes, num_initial, tolerance, data_dir, config_dir):
    """Create configuration for streaming STHOSVD run."""
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(config_dir + '/compress', exist_ok=True)
    os.makedirs(config_dir + '/reconstruct', exist_ok=True)

    rel_data_dir = os.path.relpath(data_dir, config_dir)

    with open(config_dir + '/compress.txt', 'wt') as f:
        f.write('Automatic rank determination        = true\n')
        f.write('Perform STHOSVD                     = true\n')
        f.write('Write STHOSVD result                = true\n')
        f.write('Print options                       = true\n')
        f.write('SV Threshold                        = {:e}\n'.format(tolerance))
        f.write('Global dims                         =')
        for s in mode_sizes[:-1]:
            f.write(' {:d}'.format(s))
        f.write(' {:d}\n'.format(num_initial))
        f.write('STHOSVD directory                   = compress\n')
        f.write('SV directory                        = compress\n')

        if num_initial >= mode_sizes[-1]:
            f.write('Input file list                     = filenames_initial.txt\n')
            f.write('Preprocessed output file list       = compress/pre.txt\n')
            f.write('Stats file                          = compress/stats.txt\n')
        else:
            f.write('Initial input file list             = filenames_initial.txt\n')
            f.write('Streaming input file list           = filenames_remaining.txt\n')
            f.write('Streaming statistics output file    = compress/stats.txt\n')

    with open(config_dir + '/filenames_initial.txt', 'wt') as f:
        for i in range(num_initial):
            f.write('{:s}/snapshot_{:d}.bin\n'.format(rel_data_dir, i))

    if num_initial < mode_sizes[-1]:
        with open(config_dir + '/filenames_remaining.txt', 'wt') as f:
            num_subsequent = mode_sizes[-1] - num_initial
            for i in range(num_subsequent):
                j = i + num_initial
                f.write('{:s}/snapshot_{:d}.bin\n'.format(rel_data_dir, j))

    with open(config_dir + '/reconstruct.txt', 'wt') as f:
        f.write('Print options                       = true\n')
        f.write('Beginning subscripts                =')
        for s in mode_sizes:
            f.write(' 0')
        f.write('\n')
        f.write('Ending subscripts                   =')
        for s in mode_sizes:
            f.write(' {:d}'.format(s - 1))
        f.write('\n')
        f.write('STHOSVD directory                   = compress\n')
        f.write('Output file list                    = reconstruct/filenames.txt')

    with open(config_dir + '/reconstruct/filenames.txt', 'wt') as f:
        for i in range(mode_sizes[-1]):
            f.write('reconstruct/snapshot_{:d}.bin\n'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode_sizes',
                        type=int,
                        nargs='+',
                        required=True,
                        help='Sizes of tensor modes')
    parser.add_argument('--num_initial',
                        type=int,
                        required=True,
                        help='Number of initial snapshots')
    parser.add_argument('--tolerance',
                        type=float,
                        required=True,
                        help='Tucker ST-HOSVD tolerance')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Path to data directory')
    parser.add_argument('--config_dir',
                        type=str,
                        required=True,
                        help='Path to directory where config will be saved')
    args = parser.parse_args()

    main(args.mode_sizes,
         args.num_initial,
         args.tolerance,
         args.data_dir,
         args.config_dir)
