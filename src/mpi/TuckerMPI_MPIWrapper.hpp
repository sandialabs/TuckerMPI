/*
 * TuckerMPI_MPIWrapper.hpp
 *
 *  Created on: Dec 22, 2020
 *      Author: ballard
 */

#ifndef MPI_TUCKER_MPIWRAPPER_HPP_
#define MPI_TUCKER_MPIWRAPPER_HPP_

#include <iostream>
#include "mpi.h"

namespace TuckerMPI
{

// Overloaded wrappers
void MPI_Recv_(float*, int, int, int, MPI_Comm, MPI_Status*);
void MPI_Recv_(double*, int, int, int, MPI_Comm, MPI_Status*);

void MPI_Send_(float*, int, int, int, MPI_Comm);
void MPI_Send_(double*, int, int, int, MPI_Comm);

void MPI_Bcast_(float*, int, int, MPI_Comm);
void MPI_Bcast_(double*, int, int, MPI_Comm);

void MPI_Isend_(const float*, int, int, int, MPI_Comm, MPI_Request*);
void MPI_Isend_(const double*, int, int, int, MPI_Comm, MPI_Request*);

void MPI_Irecv_(float*, int, int, int, MPI_Comm, MPI_Request*);
void MPI_Irecv_(double*, int, int, int, MPI_Comm, MPI_Request*);

void MPI_Reduce_(const float*, float*, int, MPI_Op, int, MPI_Comm);
void MPI_Reduce_(const double*, double*, int, MPI_Op, int, MPI_Comm);

void MPI_Allreduce_(const float*, float*, int, MPI_Op, MPI_Comm);
void MPI_Allreduce_(const double*, double*, int, MPI_Op, MPI_Comm);

void MPI_Gather_(const float*, int, float*, int, int, MPI_Comm);
void MPI_Gather_(const double*, int, double*, int, int, MPI_Comm);

void MPI_Gatherv_(const float*, int , float*, const int*, const int*,
     int root, MPI_Comm comm);
void MPI_Gatherv_(const double*, int , double*, const int*, const int*,
     int root, MPI_Comm comm);

void MPI_Allgatherv_(const float*, int, float*, const int*, 
     const int*, MPI_Comm);
void MPI_Allgatherv_(const double*, int, double*, const int*, 
     const int*, MPI_Comm);

void MPI_Reduce_scatter_(const float*, float*, int*, MPI_Op, MPI_Comm);
void MPI_Reduce_scatter_(const double*, double*, int*, MPI_Op, MPI_Comm);

void MPI_Alltoallv_(const float*, const int*, const int*, float*, 
     const int*, const int*, MPI_Comm);
void MPI_Alltoallv_(const double*, const int*, const int*, double*, 
     const int*, const int*, MPI_Comm);

int MPI_File_read_(MPI_File, float*, int, MPI_Status*);
int MPI_File_read_(MPI_File, double*, int, MPI_Status*);

int MPI_File_read_all_(MPI_File, float*, int, MPI_Status*);
int MPI_File_read_all_(MPI_File, double*, int, MPI_Status*);

int MPI_File_write_(MPI_File, const float*, int, MPI_Status*);
int MPI_File_write_(MPI_File, const double*, int, MPI_Status*);

int MPI_File_write_all_(MPI_File, const float*, int, MPI_Status*);
int MPI_File_write_all_(MPI_File, const double*, int, MPI_Status*);

// Specialized template wrappers
template <class scalar_t>
void MPI_Type_create_subarray_(int, const int [], const int [], const int [],
     int, MPI_Datatype*);

template <class scalar_t>
void MPI_File_set_view_(MPI_File, MPI_Offset, MPI_Datatype, const char*, 
     MPI_Info);

template <class scalar_t>
int MPI_Get_count_(const MPI_Status* status, int *count);

} // end namespace TuckerMPI

#endif /* MPI_TUCKER_MPIWRAPPER_HPP_ */
