
import os, sys
import numpy as np
from argparse import ArgumentParser

# -------------------------------------------------------------------
def extract_datatensor_rank(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("Global dims = "):
                ss = line.split("=")[1].strip()
                arr = np.fromstring(ss, sep=" ")
                return len(arr)

# -------------------------------------------------------------------
def extract_single_line_value_from_paramfile(line_in_paramfile, paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith(line_in_paramfile):
                # print(line.split("="))
                return line.split("=")[1].strip()

# -------------------------------------------------------------------
def foundAtLeastOneFile(workDir, substring):
	files = [workDir+'/'+f for f in os.listdir(workDir) if substring in f]
	return bool(files)

# -------------------------------------------------------------------
def compare_eigenvalues(workDir, modeCount):
	print("1. comparator: comparing eigenvalues")

	if not foundAtLeastOneFile(workDir, "goldEigvals_"):
		sys.exit("comparator: aborting because no eigenvalues files were found!")

	tols = { "rel": 1e-8, "abs": 1e-8 }

	# figure out where sing values are written to
	svPath = extract_single_line_value_from_paramfile("SV directory = ", workDir+"/paramfile.txt")
	svPrefix = extract_single_line_value_from_paramfile("SV file prefix = ", workDir+"/paramfile.txt")

	for i in range(0, modeCount):
		print("\tComparing eigenvalues for mode = {}".format(i))

		gold = np.loadtxt(workDir+"/goldEigvals_"+str(i)+".txt")
		computed = np.loadtxt(svPath + "/" + svPrefix + "_mode_"+str(i)+".txt")

		assert (len(gold) == len(computed)), \
		    "Failing due to mismatching extents!"
		assert (np.allclose(gold, computed, rtol=tols['rel'], atol=tols['abs'])), \
		    "Failing due to numerical differences!"

# -------------------------------------------------------------------
def compare_normalizedtensor(workDir):
	print("2. comparator: comparing normalized tensor")

	tols = {"rel": 1e-7, "abs": 1e-8}

	gold = np.loadtxt(workDir+"/goldNormalizedTensor.txt")
	computed = np.fromfile(workDir+"/normalized_tensor.bin")

	assert(len(gold) == len(computed)), \
		"Failing due to mismatching extents!"
	assert (np.allclose(gold, computed, rtol=tols['rel'], atol=tols['abs'])), \
		"Failing due to numerical differences!"

# -------------------------------------------------------------------
def compare_coretensor(workDir):
	print("3. comparator: comparing core tensor")

	tols = {"rel": 1e-8, "abs": 1e-8}

	# figure out where sthosvd values are written to
	sthosvdPath   = extract_single_line_value_from_paramfile("STHOSVD directory = ", workDir+"/paramfile.txt")
	sthosvdPrefix = extract_single_line_value_from_paramfile("STHOSVD file prefix = ", workDir+"/paramfile.txt")

	computedFile = sthosvdPath + "/" + sthosvdPrefix + "_core.mpi"
	goldFile = workDir+"/goldCoreTensor.txt"
	print("\t computed: ", computedFile)
	print("\t gold    : ", goldFile)

	computed = np.fromfile(computedFile)
	np.savetxt(workDir+"/computedCoreTensor.txt", computed)
	gold = np.loadtxt(workDir+"/goldCoreTensor.txt")

	assert(len(gold) == len(computed)), \
		"Failing due to mismatching extents!"
	assert (np.allclose(np.abs(gold), np.abs(computed), rtol=tols['rel'], atol=tols['abs'])), \
		"Failing due to numerical differences!"

# -------------------------------------------------------------------
def compare_metrics(workDir):
	print("4. comparator: comparing metrics")

	tols = {"rel": 1e-4, "abs": 1e-6}

	gold = np.loadtxt(workDir+"/goldMetrics.txt", skiprows=3)
	computed = np.loadtxt(workDir+"/stats.txt", skiprows=3)

	assert(len(gold) == len(computed)), \
		"Failing due to mismatching extents!"
	assert(np.allclose(gold, computed, rtol=tols['rel'], atol=tols['abs'])), \
		"Failing due to numerical differences!"

# -------------------------------------------------------------------
def compare_factors(workDir, modeCount):
	print("5. comparator: comparing factor matrices")

	tols = {"rel": 1e-8, "abs": 1e-8}

	sthosvdPath   = extract_single_line_value_from_paramfile("STHOSVD directory = ", "./paramfile.txt")
	sthosvdPrefix = extract_single_line_value_from_paramfile("STHOSVD file prefix = ", "./paramfile.txt")
	# convert
	for i in range(0, modeCount):
		binary = np.fromfile(sthosvdPath + "/" + sthosvdPrefix + "_mat_" + str(i) + ".mpi")
		np.savetxt("factorMatrix_" + str(i) + ".txt", binary)

	# compare
	for i in range(0, modeCount):
		print("\tComparing fator for mode = {}".format(i))

		gold = np.loadtxt(workDir+"/goldFactorMatrix_" + str(i) + ".txt")
		computed = np.loadtxt(workDir+"/factorMatrix_" + str(i) + ".txt")

		assert (len(gold) == len(computed)), \
			"Failing due to mismatching extents!"
		assert (np.allclose(np.abs(gold), np.abs(computed), rtol=tols['rel'], atol=tols['abs'])), \
			"Failing due to numerical differences!"

# -------------------------------------------------------------------
if __name__== "__main__":
# -------------------------------------------------------------------
	# we have as many modes to read as the ranks of the tensor
	modeCount = extract_datatensor_rank("./paramfile.txt")
	print("\n")

	workDir = "."
	compare_eigenvalues(workDir, modeCount)

	if foundAtLeastOneFile(workDir, "goldNormalizedTensor"):
		compare_normalizedtensor(workDir)

	if foundAtLeastOneFile(workDir, "goldCoreTensor"):
		compare_coretensor(workDir)

	if foundAtLeastOneFile(workDir, "goldMetrics"):
		compare_metrics(workDir)

	if foundAtLeastOneFile(workDir, "goldFactorMatrix"):
		compare_factors(workDir, modeCount)
