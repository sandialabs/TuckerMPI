#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;
using scalar_t = double;

const MPI_Comm comm = MPI_COMM_WORLD;

int mpi_size(){
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  return nprocs;
}

int mpi_rank(){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

template <class scalar_t>
bool checkUTEqual(const scalar_t* arr1, const scalar_t* arr2, int numRows)
{
  const double TOL = 1000 * std::numeric_limits<scalar_t>::epsilon();

  for(int r=0; r<numRows; r++) {
    for(int c=r; c<numRows; c++) {
      int ind = r+c*numRows;
      if(std::abs(arr1[ind]-arr2[ind]) > TOL) {
        if(mpi_rank() == 0){
          std::cerr << "ERROR: The true solution is " << arr2[ind]
                    << ", but the computed solution was " << arr1[ind]
                    << ", a difference of " << std::abs(arr1[ind]-arr2[ind])
                    << std::endl;
        }
        return false;
      }
    }
  }
  return true;
}

bool runSim(std::vector<int> procs)
{
  if(mpi_rank() == 0){
    for(int element: procs)
      std::cout << element << ", ";
  }

  std::vector<int> dims = {3, 5, 7, 11};
  Tensor<scalar_t> T(dims, procs);
  read_tensor_binary(T, "./tensor_data_files/3x5x7x11.bin");

  auto matrix = impl::new_gram(T, 0);

  const scalar_t TRUE_SOLUTION_0[3*3] =
    {30.443306716415002, 2.839326384970902, 3.302846757455287,
     2.839326384970902, 30.254535550071385, -1.525701935166856,
     3.302846757455286,  -1.525701935166856,  33.203090378426815};

  bool matchesTrueSol = checkUTEqual(matrix.data(), TRUE_SOLUTION_0, 3);
  if(!matchesTrueSol)
    return false;
  
  matrix = impl::new_gram(T, 1);

  const scalar_t TRUE_SOLUTION_1[5*5] =
    {19.818638147579669, -1.262410163567289, -0.702854185504209, -1.530487842705398, 0.076595773936855,
     -1.262410163567289, 19.055467182423385, -0.066700982927269,  2.017798574327055, -2.178581571284073,
     -0.702854185504209, -0.066700982927269, 18.565387462362573, -1.776252595103098, 1.511348003921888,
     -1.530487842705398,  2.017798574327055, -1.776252595103098, 19.948082945419085, -1.701919532297699,
      0.076595773936855, -2.178581571284071,  1.511348003921888, -1.701919532297699, 16.513356907128461};

  matchesTrueSol = checkUTEqual(matrix.data(), TRUE_SOLUTION_1, 5);

  if(!matchesTrueSol)
    return false;

  matrix = impl::new_gram(T, 2);

  const scalar_t TRUE_SOLUTION_2[7*7] =
    {15.008467435803363, 0.759679780649146, 0.036286707752565, -0.289429119623268, 0.297697996043826, 1.775651754316252, 0.290583279984489,
     0.759679780649146, 13.534258753086892, 0.216613144804639, -0.909444710671252, -0.440040933466265, -1.316163513868651, 0.331266618564718,
     0.036286707752565, 0.216613144804639, 13.107978854355352, 0.701142578929455, 1.325848667245948, -0.355644067560675, 0.307781943716007,
     -0.289429119623268, -0.909444710671252, 0.701142578929455, 13.223174991133867, 1.085334988154543, 1.098880462506622, 2.998592432064371,
     0.297697996043825, -0.440040933466265, 1.325848667245947, 1.085334988154542, 13.872205816335024, -0.682375996350724, 1.263779626695686,
     1.775651754316251, -1.316163513868651, -0.355644067560674, 1.098880462506623, -0.682375996350724, 12.100552292592960, 0.747070331494893,
     0.290583279984489, 0.331266618564718, 0.307781943716007, 2.998592432064369, 1.263779626695687, 0.747070331494893, 13.054294501605717};

  matchesTrueSol = checkUTEqual(matrix.data(), TRUE_SOLUTION_2, 7);

  if(!matchesTrueSol)
    return false;
  
  matrix = impl::new_gram(T, 3);

  const scalar_t TRUE_SOLUTION_3[11*11] =
    {9.161015785664386, 0.727680925086397, -0.027808461977047, -1.858926623169289, 2.161910031054029, 0.538498274148853, -1.513458574761186, 1.560326634563823, 0.092898181083783, 0.193490784986641, -1.989938728094985,
        0.727680925086397, 8.579166359218990, 0.141184182846715, 0.895099070075687, -0.217980995935905, -0.257633332960250, 0.874112582420890, -1.046151056779062, 0.285393081978074, -0.864025655243411, 0.241418995156843,
        -0.027808461977047, 0.141184182846715, 8.487659393716891, -0.196516290753068, 0.855490962332431, 0.483991332254019, -0.096130754960697, 1.054249161028251, 0.718345150564935, -0.962619165522184, -1.619740748464319,
        -1.858926623169289, 0.895099070075687, -0.196516290753068, 8.222091258079500, -0.657900464867635, -0.649610924287497, 1.661744576911002, -0.193946562374636, 0.072191263966888, 0.586444766094151, -0.340892667710611,
        2.161910031054029, -0.217980995935905, 0.855490962332431, -0.657900464867635, 6.873280265688026, 1.816757520854934, -1.262051827038299, 0.678477183630724, 0.135986851830552, -0.228764676015497, -1.666495122122524,
        0.538498274148853, -0.257633332960250, 0.483991332254019, -0.649610924287497, 1.816757520854934, 8.227968322612568, 0.861291545524726, -0.353270918783090, -0.570666332802020, -1.789631071892280, 0.442503856536748,
        -1.513458574761186, 0.874112582420890, -0.096130754960697, 1.661744576911002, -1.262051827038299, 0.861291545524726, 9.269694148981491, -1.574380517040051, 1.099770875691088, 0.069588373131300, -0.717801546637968,
        1.560326634563823, -1.046151056779062, 1.054249161028251, -0.193946562374636, 0.678477183630724, -0.353270918783090, -1.574380517040051, 9.070501064224349, -0.973741303421328, -0.752651299409004, 0.151673397796920,
        0.092898181083782, 0.285393081978074, 0.718345150564935, 0.072191263966888, 0.135986851830551, -0.570666332802020, 1.099770875691088, -0.973741303421328, 7.925894762690504, 0.447258387553726, -0.591921295268904,
        0.193490784986642, -0.864025655243411, -0.962619165522184, 0.586444766094151, -0.228764676015497, -1.789631071892280, 0.069588373131299, -0.752651299409004, 0.447258387553726, 8.517370143817459, -0.523119424574941,
        -1.989938728094985, 0.241418995156843, -1.619740748464319, -0.340892667710611, -1.666495122122525, 0.442503856536748, -0.717801546637968, 0.151673397796920, -0.591921295268904, -0.523119424574941, 9.566291140219018};

  matchesTrueSol = checkUTEqual(matrix.data(), TRUE_SOLUTION_3, 11);

  if(!matchesTrueSol)
    return false;

  if(mpi_rank() == 0)
    std::cout << ": PASSED" << std::endl;;

  return true;
}

TEST(tuckermpi, ttm_new_gram_nprocs2)
{
  std::vector<int> procs = {-1, -1, -1, -1};

  if(mpi_size() == 2) {
    procs[0] = 2; procs[1] = 1; procs[2] = 1; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 2; procs[2] = 1; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 1; procs[2] = 2; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 1; procs[2] = 1; procs[3] = 2;
    ASSERT_TRUE(runSim(procs));
  }
}

TEST(tuckermpi, ttm_new_gram_nprocs6)
{
  using scalar_t = double;

  std::vector<int> procs = {-1, -1, -1, -1};

  if(mpi_size() == 6){
    procs[0] = 6; procs[1] = 1; procs[2] = 1; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));
    
    procs[0] = 1; procs[1] = 6; procs[2] = 1; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 1; procs[2] = 6; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 1; procs[2] = 1; procs[3] = 6;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 2; procs[1] = 3; procs[2] = 1; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));
    
    procs[0] = 2; procs[1] = 1; procs[2] = 3; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 2; procs[1] = 1; procs[2] = 1; procs[3] = 3;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 2; procs[2] = 3; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 2; procs[2] = 1; procs[3] = 3;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 1; procs[2] = 2; procs[3] = 3;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 3; procs[1] = 2; procs[2] = 1; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));
    
    procs[0] = 3; procs[1] = 1; procs[2] = 2; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 3; procs[1] = 1; procs[2] = 1; procs[3] = 2;
    ASSERT_TRUE(runSim(procs));

    procs[0] = 1; procs[1] = 3; procs[2] = 2; procs[3] = 1;
    ASSERT_TRUE(runSim(procs));
    
    procs[0] = 1; procs[1] = 3; procs[2] = 1; procs[3] = 2;
    ASSERT_TRUE(runSim(procs));
    
    procs[0] = 1; procs[1] = 1; procs[2] = 3; procs[3] = 2;
    ASSERT_TRUE(runSim(procs));
  }
}
