"""Generate error statistics."""


import argparse
import itertools
import os
import sys

import numpy as np


def compute_errors(dir_name, rel_tol, abs_tol):
    with open('{:s}/reconstruct_stats.txt'.format(dir_name), 'wt') as f_out:
        f_out.write('{:>6s} {:>12s} {:>12s} {:>12s} {:>12s}\n'.format('step', 'abserr', 'relerr', 'abserr_track', 'relerr_track'))

        if os.path.isfile('{:s}/filenames_remaining.txt'.format(dir_name)):
            filenames_exact_initial = open('{:s}/filenames_initial.txt'.format(dir_name), 'rt')
            filenames_exact_remaining = open('{:s}/filenames_remaining.txt'.format(dir_name), 'rt')
            filenames_exact = itertools.chain(filenames_exact_initial, filenames_exact_remaining)
        else:
            filenames_exact = open('{:s}/filenames_initial.txt'.format(dir_name), 'rt')

        filenames_apprx = open('{:s}/reconstruct/filenames.txt'.format(dir_name), 'rt')

        errnrm2_tracking = 0.0
        tnsnrm2_tracking = 0.0

        passed = True

        for i, (fn_exact, fn_apprx) in enumerate(zip(filenames_exact, filenames_apprx)):
            tns_exact = np.fromfile('{:s}/{:s}'.format(dir_name, fn_exact.strip()), '<f8')
            tns_apprx = np.fromfile('{:s}/{:s}'.format(dir_name, fn_apprx.strip()), '<f8')

            abserr = np.linalg.norm(tns_exact - tns_apprx)
            tnsnrm = np.linalg.norm(tns_exact)
            relerr = abserr / tnsnrm

            errnrm2_tracking += abserr**2
            tnsnrm2_tracking += tnsnrm**2

            abserr_tracking = np.sqrt(errnrm2_tracking)
            relerr_tracking = np.sqrt(errnrm2_tracking / tnsnrm2_tracking)

            f_out.write('{:6d} {:12.6e} {:12.6e} {:12.6e} {:12.6e}\n'.format(i, abserr, relerr, abserr_tracking, relerr_tracking))

            if abserr > abs_tol:
                print('Absolute reconstruction error {:.3e} for slice {:d} exceeded tolerance {:.3e}'.format(abserr, i, abs_tol))
                passed = False
            if relerr > rel_tol:
                print('Relative reconstruction error {:.3e} for slice {:d} exceeded tolerance {:.3e}'.format(relerr, i, rel_tol))
                passed = False

        if os.path.isfile('{:s}/filenames_remaining.txt'.format(dir_name)):
            filenames_exact_initial.close()
            filenames_exact_remaining.close()
        else:
            filenames_exact.close()

        filenames_apprx.close()
        return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',
                        type=str,
                        required=True,
                        help='Path to directory where config will be saved')
    parser.add_argument('--rel_tol',
                        type=float,
                        default=1.0e-10,
                        required=False,
                        help='Tolerance for relative error in reconstruction indicating pass or fail')
    parser.add_argument('--abs_tol',
                        type=float,
                        default=1.0e-10,
                        required=False,
                        help='Tolerance for absolute error in reconstruction indicating pass or fail')
    args = parser.parse_args()

    passed = compute_errors(args.config_dir, args.rel_tol, args.abs_tol)
    if passed:
        sys.exit(0)
    sys.exit(1)
