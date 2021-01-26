/*
 * TuckerMPI_MPIWrapper.cpp
 *
 *  Created on: Dec 23, 2020
 *      Author: ballard
 */

#include "TuckerMPI_MPIWrapper.hpp"

namespace TuckerMPI
{

void MPI_Isend_(const float* buf, int count, int dest, int tag,
     MPI_Comm comm, MPI_Request* request)
{
  MPI_Isend(buf, count, MPI_FLOAT, dest, tag, comm, request);
}

void MPI_Isend_(const double* buf, int count, int dest, int tag,
     MPI_Comm comm, MPI_Request* request)
{
  MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, comm, request);
}

void MPI_Irecv_(float* buf, int count, int source, int tag, 
     MPI_Comm comm, MPI_Request* request)
{
    MPI_Irecv(buf, count, MPI_FLOAT, source, tag, comm, request);
}

void MPI_Irecv_(double* buf, int count, int source, int tag, 
     MPI_Comm comm, MPI_Request* request)
{
    MPI_Irecv(buf, count, MPI_DOUBLE, source, tag, comm, request);
}

void MPI_Reduce_(const float* sendbuf, float* recvbuf, int count, 
     MPI_Op op, int root, MPI_Comm comm)
{
  MPI_Reduce(sendbuf, recvbuf, count, MPI_FLOAT, op, root, comm);
}

void MPI_Reduce_(const double* sendbuf, double* recvbuf, int count, 
     MPI_Op op, int root, MPI_Comm comm)
{
  MPI_Reduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, root, comm);
}

void MPI_Allreduce_(const float* sendbuf, float* recvbuf, int count, 
     MPI_Op op, MPI_Comm comm)
{
  MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, op, comm);
}

void MPI_Allreduce_(const double* sendbuf, double* recvbuf, int count, 
     MPI_Op op, MPI_Comm comm)
{
  MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
}

void MPI_Gather_(const float* sendbuf, int sendcount, float* recvbuf, 
     int recvcount, int root, MPI_Comm comm)
{
    MPI_Gather(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, 
    MPI_FLOAT, root, comm);
}

void MPI_Gather_(const double* sendbuf, int sendcount, double* recvbuf, 
     int recvcount, int root, MPI_Comm comm)
{
    MPI_Gather(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, 
    MPI_DOUBLE, root, comm);
}

void MPI_Gatherv_(const float* sendbuf, int sendcount, float* recvbuf, 
     const int* recvcounts, const int* displs, int root, MPI_Comm comm)
{
    MPI_Gatherv(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcounts, displs,
                MPI_FLOAT, root, comm);
}

void MPI_Gatherv_(const double* sendbuf, int sendcount, double* recvbuf, 
     const int* recvcounts, const int* displs, int root, MPI_Comm comm)
{
    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcounts, displs,
                MPI_DOUBLE, root, comm);
}

void MPI_Allgatherv_(const float* sendbuf, int sendcount, 
     float* recvbuf, const int *recvcounts, const int* displs,MPI_Comm comm)
{
    MPI_Allgatherv(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcounts, displs,
    MPI_FLOAT, comm);
}

void MPI_Allgatherv_(const double* sendbuf, int sendcount, 
     double* recvbuf, const int *recvcounts, const int* displs,MPI_Comm comm)
{
    MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcounts, displs,
    MPI_DOUBLE, comm);
}

void MPI_Reduce_scatter_(const float* sendbuf, float* recvbuf, int *recvcounts, 
     MPI_Op op, MPI_Comm comm)
{
    MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPI_FLOAT, op, comm);
}

void MPI_Reduce_scatter_(const double* sendbuf, double* recvbuf, int *recvcounts, 
     MPI_Op op, MPI_Comm comm)
{
    MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPI_DOUBLE, op, comm);
}

void MPI_Alltoallv_(const float* sendbuf, const int* sendcounts, const int* sdispls, 
     float* recvbuf, const int* recvcounts, const int* rdispls, MPI_Comm comm)
{
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_FLOAT, recvbuf, recvcounts, 
    rdispls, MPI_FLOAT, comm);
}

void MPI_Alltoallv_(const double* sendbuf, const int* sendcounts, const int* sdispls, 
     double* recvbuf, const int* recvcounts, const int* rdispls, MPI_Comm comm)
{
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_DOUBLE, recvbuf, recvcounts, 
    rdispls, MPI_DOUBLE, comm);
}

int MPI_File_read_(MPI_File fh, float* buf, int count, MPI_Status* status)
{
    return MPI_File_read(fh, buf, count, MPI_FLOAT, status);
}

int MPI_File_read_(MPI_File fh, double* buf, int count, MPI_Status* status)
{
    return MPI_File_read(fh, buf, count, MPI_DOUBLE, status);
}

int MPI_File_read_all_(MPI_File fh, float* buf, int count, MPI_Status* status)
{
    return MPI_File_read_all(fh, buf, count, MPI_FLOAT, status);
}

int MPI_File_read_all_(MPI_File fh, double* buf, int count, MPI_Status* status)
{
    return MPI_File_read_all(fh, buf, count, MPI_DOUBLE, status);
}

int MPI_File_write_(MPI_File fh, const float* buf, int count, MPI_Status* status)
{
    return MPI_File_write(fh, buf, count, MPI_FLOAT, status);
}

int MPI_File_write_(MPI_File fh, const double* buf, int count, MPI_Status* status)
{
    return MPI_File_write(fh, buf, count, MPI_DOUBLE, status);
}

int MPI_File_write_all_(MPI_File fh, const float* buf, int count, 
    MPI_Status* status)
{
    return MPI_File_write_all(fh, buf, count, MPI_FLOAT, status);
}

int MPI_File_write_all_(MPI_File fh, const double* buf, int count, 
    MPI_Status* status)
{
    return MPI_File_write_all(fh, buf, count, MPI_DOUBLE, status);
}

template <class scalar_t>
void MPI_Type_create_subarray_(int ndims, const int array_of_sizes[],
     const int array_of_subsizes[], const int array_of_starts[],
     int order, MPI_Datatype *newtype)
{
    // Using template specialization instead of overloading because type 
    // cannot be inferred from argument list; generic implementation fails
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank == 0) {
      std::cerr << "ERROR: Unknown type for MPI_Type_create_subarray.\n" <<
      "Calling MPI_Abort...\n";
    }
    MPI_Abort(MPI_COMM_WORLD,1);
}

template <>
void MPI_Type_create_subarray_<float>(int ndims, const int array_of_sizes[],
     const int array_of_subsizes[], const int array_of_starts[],
     int order, MPI_Datatype *newtype)
{
    MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
    array_of_starts, order, MPI_FLOAT, newtype);
}

template <>
void MPI_Type_create_subarray_<double>(int ndims, const int array_of_sizes[],
     const int array_of_subsizes[], const int array_of_starts[],
     int order, MPI_Datatype *newtype)
{
    MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
    array_of_starts, order, MPI_DOUBLE, newtype);
}

template <class scalar_t>
void MPI_File_set_view_(MPI_File fh, MPI_Offset disp, MPI_Datatype filetype, 
     const char *datarep, MPI_Info info)
{
    // Using template specialization instead of overloading because type 
    // cannot be inferred from argument list; generic implementation fails
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank == 0) {
      std::cerr << "ERROR: Unknown type for MPI_File_set_view.\n" <<
      "Calling MPI_Abort...\n";
    }
    MPI_Abort(MPI_COMM_WORLD,1);
}

template <>
void MPI_File_set_view_<float>(MPI_File fh, MPI_Offset disp, MPI_Datatype filetype, 
     const char *datarep, MPI_Info info)
{
    MPI_File_set_view(fh, disp, MPI_FLOAT, filetype, datarep, info);
}

template <>
void MPI_File_set_view_<double>(MPI_File fh, MPI_Offset disp, MPI_Datatype filetype, 
     const char *datarep, MPI_Info info)
{
    MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, datarep, info);
}

template <class scalar_t>
int MPI_Get_count_(const MPI_Status* status, int *count)
{
    // Using template specialization instead of overloading because type 
    // cannot be inferred from argument list; generic implementation fails
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank == 0) {
      std::cerr << "ERROR: Unknown type for MPI_Get_count.\n" <<
      "Calling MPI_Abort...\n";
    }
    return MPI_Abort(MPI_COMM_WORLD,1);
}

template <>
int MPI_Get_count_<float>(const MPI_Status* status, int *count)
{
    return MPI_Get_count(status, MPI_FLOAT, count);
}

template <>
int MPI_Get_count_<double>(const MPI_Status* status, int *count)
{
    return MPI_Get_count(status, MPI_DOUBLE, count);
}

} // end namespace TuckerMPI
