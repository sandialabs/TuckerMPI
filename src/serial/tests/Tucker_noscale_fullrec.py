#!/usr/bin/python

import os
import subprocess

# Generate the data
os.chdir("/home/amklinv/TuckerMPI/build/serial/tests/input_files/system/serial_noscale_fullrec/generate")
print "/home/amklinv/TuckerMPI/build/serial/drivers/bin/Tucker_generate"
subprocess.check_call('/home/amklinv/TuckerMPI/build/serial/drivers/bin/Tucker_generate')

# Compress the data
os.chdir("../sthosvd")
print "/home/amklinv/TuckerMPI/build/serial/drivers/bin/Tucker_sthosvd"
subprocess.check_call('/home/amklinv/TuckerMPI/build/serial/drivers/bin/Tucker_sthosvd')

# Make sure the data is sufficiently compressed
with open("compressed/sthosvd_ranks.txt") as f:
    computed_ranks = [int(x) for x in f]
true_ranks = [40, 20, 30, 9, 25]
print "True ranks are ", true_ranks
print "Computed ranks are ", computed_ranks
assert(not cmp(true_ranks,computed_ranks))

# Un-compress the data
os.chdir("../reconstruct")
print "/home/amklinv/TuckerMPI/build/serial/drivers/bin/Tucker_reconstruct"
subprocess.check_call('/home/amklinv/TuckerMPI/build/serial/drivers/bin/Tucker_reconstruct')

# Compare the outputs
print "/home/amklinv/TuckerMPI/build/serial/compare/bin/compare"
subprocess.check_call(['/home/amklinv/TuckerMPI/build/serial/compare/bin/compare', '55000000', '../generate/generated_tensor.mpi', 'reconstructed.mpi', '1e-6'])