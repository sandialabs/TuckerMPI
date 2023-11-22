"""Generate noisy wave data."""


import argparse
import os
import numpy as np


def linear_to_centered_cartesian(linear_index, extent):
    """Convert linear index to centered cartesian index."""
    num_dim = len(extent)

    cartesian_index = [None] * num_dim
    temp = linear_index
    for d in range(num_dim):
        cartesian_index[d] = temp % (2 * extent[d] + 1) - extent[d]
        temp = temp // (2 * extent[d] + 1)

    return cartesian_index


def test_linear_to_centered_cartesian():
    """Unit test."""
    assert linear_to_centered_cartesian(101, [5, 5]) == [-3, 4]
    assert linear_to_centered_cartesian(550, [4, 4, 4]) == [-3, 3, 2]


def generate_clean_data(mode_sizes, maximum_frequencies, rng):
    """Generate clean low-rank tensor data."""
    num_dim = len(mode_sizes)
    x = [None] * num_dim
    for d in range(num_dim):
        shape = [1] * num_dim
        shape[d] = -1
        x[d] = np.linspace(0.0, 2.0 * np.pi, mode_sizes[d])
        x[d] = x[d].reshape(tuple(shape))

    num_mode = np.prod([2 * j + 1 for j in maximum_frequencies])
    amplitudes = rng.standard_normal(size=num_mode)

    data = np.zeros(tuple(mode_sizes), dtype=np.float64)
    j_dot_x = np.zeros(tuple(mode_sizes), dtype=np.float64)
    for j_linear in range(num_mode):
        j = linear_to_centered_cartesian(j_linear, maximum_frequencies)

        j_dot_x.fill(0.0)
        for d in range(num_dim):
            j_dot_x += j[d] * x[d]

        data += amplitudes[j_linear] * np.sin(j_dot_x)

    return data


def test_generate_clean_data():
    """Unit test."""
    rng = np.random.default_rng(0)

    def compute_rank(A, epsilon):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return np.sum(s > epsilon)

    data = generate_clean_data([20, 30, 40], [2, 3, 4], rng)

    unfold_0 = data.transpose([0, 1, 2]).reshape(20, -1)
    unfold_1 = data.transpose([1, 0, 2]).reshape(30, -1)
    unfold_2 = data.transpose([2, 0, 1]).reshape(40, -1)

    rank_0 = compute_rank(unfold_0, 1.0e-12)
    rank_1 = compute_rank(unfold_1, 1.0e-12)
    rank_2 = compute_rank(unfold_2, 1.0e-12)

    assert rank_0 == 5
    assert rank_1 == 7
    assert rank_2 == 9


def generate_clean_data_with_rank_increase(mode_sizes, starting_frequencies, frequency_bumps, rng):
    if frequency_bumps == []:
        return generate_clean_data(mode_sizes, starting_frequencies, rng)
        
    halfway = mode_sizes[0] // 2
    mode_sizes_1 = [halfway                 if j == 0 else mode_sizes[j] for j in range(len(mode_sizes))]
    mode_sizes_2 = [mode_sizes[j] - halfway if j == 0 else mode_sizes[j] for j in range(len(mode_sizes))]

    maximum_frequencies_1 = starting_frequencies
    maximum_frequencies_2 = [f0 + f1 for (f0, f1) in zip(starting_frequencies, frequency_bumps)]
    
    data_1 = generate_clean_data(mode_sizes_1, maximum_frequencies_1, rng)
    data_2 = generate_clean_data(mode_sizes_2, maximum_frequencies_2, rng)

    return np.concatenate((data_1, data_2), axis=0)


def compute_noise_scale_factor(data, noise, noise_to_signal_ratio):
    """Compute noise scale factor."""
    data_norm_2 = np.sum(data**2)
    noise_norm_2 = np.sum(noise**2)
    data_dot_noise = np.sum(data * noise)
    eta = noise_to_signal_ratio**2
    return (eta * data_dot_noise +
            np.sqrt(eta**2 * data_dot_noise**2 +
                    eta * (1 - eta) * data_norm_2 * noise_norm_2)) \
        / ((1 - eta) * noise_norm_2)


def test_compute_noise_scale_factor():
    """Unit test."""
    x = 10.0 * np.random.randn(100, 100, 100)
    n = np.random.randn(100, 100, 100)

    for r in [0.4, 0.2, 0.1, 0.05, 0.025]:
        s = compute_noise_scale_factor(x, n, r)
        y = x + s * n
        e = y - x
        assert abs(np.sqrt(np.sum(e**2)) / np.sqrt(np.sum(y**2)) - r) < 1.0e-12


def main(mode_sizes, maximum_frequencies, frequency_bumps, noise_to_signal_ratio, data_dir, verbose):
    """Create data from noisy wave data source."""
    os.makedirs(data_dir, exist_ok=True)

    mode_sizes.reverse()
    maximum_frequencies.reverse()
    frequency_bumps.reverse()

    rng = np.random.default_rng(0)

    data = generate_clean_data_with_rank_increase(mode_sizes, maximum_frequencies, frequency_bumps, rng)

    noise = rng.standard_normal(size=data.shape)

    for i in range(mode_sizes[0]):
        scale = compute_noise_scale_factor(data[i, :], noise[i, :],
                                           noise_to_signal_ratio)
        data[i, :] += scale * noise[i, :]

        file_name = data_dir + '/snapshot_' + str(i) + '.bin'
        if verbose:
            print('Writing file {:s}'.format(file_name))

        with open(file_name, 'wb') as f:
            f.write(data[i, :].tobytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode_sizes',
                        type=int,
                        nargs='+',
                        required=True,
                        help='Sizes of tensor modes')
    parser.add_argument('--maximum_frequencies',
                        type=int,
                        nargs='+',
                        required=True,
                        help='Maximum frequencies along tensor modes')
    parser.add_argument('--frequency_bumps',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Added frequency halfway through along tensor modes')
    parser.add_argument('--noise_to_signal_ratio',
                        type=float,
                        required=True,
                        help='Fraction of white noise to add to the signal')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Path to directory where data will be saved')
    parser.add_argument('--verbose',
                        help='Verbose output')
    args = parser.parse_args()

    main(args.mode_sizes,
         args.maximum_frequencies,
         args.frequency_bumps,
         args.noise_to_signal_ratio,
         args.data_dir,
         args.verbose)
