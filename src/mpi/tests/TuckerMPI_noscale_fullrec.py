#!/usr/bin/python

import os
import subprocess
import sys

# Generate the data
os.chdir("input_files/system/mpi_noscale_fullrec/generate")
cmd = sys.argv[1:]+['../../../../../drivers/bin/generate']
print "Command line: ", str(cmd)
subprocess.check_call(cmd)

# Compress the data
os.chdir("../sthosvd")
cmd = sys.argv[1:]+['../../../../../drivers/bin/sthosvd']
print "Command line: ", str(cmd)
subprocess.check_call(cmd)

# Make sure the data is sufficiently compressed
with open("compressed/sthosvd_ranks.txt") as f:
    computed_ranks = [int(x) for x in f]
true_ranks = [40, 20, 30, 9, 25]
print "True ranks are ", true_ranks
print "Computed ranks are ", computed_ranks
assert(not cmp(true_ranks,computed_ranks))

# Un-compress the data
os.chdir("../reconstruct")
cmd = sys.argv[1:]+['../../../../../drivers/bin/reconstruct']
print "Command line: ", str(cmd)
subprocess.check_call(cmd)

# Compare the outputs using serial compare
print "../../../../../../serial/compare/bin/compare"
subprocess.check_call(['../../../../../../serial/compare/bin/compare', '55000000', '../generate/generated_tensor.mpi', 'reconstructed.mpi', '1e-6'])
