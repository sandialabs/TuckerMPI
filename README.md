This is the GIT repo for the work on building a parallel Tucker for
combustion data.                                                   

For more information:
Alicia Klinvex (amklinv@sandia.gov)
Grey Ballard   (ballard@wfu.edu)
Tamara Kolda   (tgkolda@sandia.gov)
Woody Austin   (austinwn@cs.utexas.edu)
Hemanth Kolla  (hnkolla@sandia.gov)

WARNING
-------
This code is not ready for public release, but is ready for evaluation by friendly expert users.  Please contact us if you have any questions, or submit an issue if you find a bug or wish to request a new feature.

Requirements
------------
MPI implementation (We use openMPI MPICH2, and MVAPICH2)
BLAS implementation                                     
LAPACK implementation                                   
C++11 or greater

Documentation
-------------
Please see https://tensors.gitlab.io/TuckerMPI

Papers
------
Parallel Tensor Compression for Large-Scale Scientific Data
Woody Austin, Grey Ballard, and Tamara G. Kolda

Funding Statement
-----------------
The development of this software was supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program and a Sandia Truman Postdoctoral Fellowship (LDRD funding). Sandia National Laboratories is a multi-program laboratory managed and operated by Sandia Corporation, a wholly owned subsidiary of Lockheed Martin Corporation, for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE–AC04–94AL85000.