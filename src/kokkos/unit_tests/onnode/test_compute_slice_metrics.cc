#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

TEST(tuckerkokkos, compute_slice_metrics_mode2){
  // Prepare
  using scalar_t = double;

  // Create a 2x2x2 tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  int mode = 2;

  // Fill it with the entries 0:8
  auto tensor_d = tensor.data();
  for(int i=0; i<8; i++){
    tensor_d(i) = i;
  }

  // Create a metric data storing the slice

  const std::vector<Tucker::Metric> metrics{Tucker::Metric::MIN,
					    Tucker::Metric::MAX};
  //Tucker::MEAN+Tucker::VARIANCE;
  //auto metrics = Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE;
  auto metricsData = TuckerOnNode::compute_slice_metrics(tensor, mode, metrics);
  auto metricsData_h = TuckerOnNode::create_mirror(metricsData);
  TuckerOnNode::deep_copy(metricsData_h, metricsData);

  // // True results (compute by hand)
  // int nbr_slice = 2;
  // scalar_t true_min[nbr_slice]   = {0.0, 4.0};
  // scalar_t true_max[nbr_slice]   = {3.0, 7.0};
  // scalar_t true_mean[nbr_slice]  = {1.5, 5.5};
  // scalar_t true_var[nbr_slice]   = {1.25, 1.25};
  // scalar_t true_stdev[nbr_slice] = {1.1180, 1.1180};
  auto minV = metricsData_h.get(Tucker::Metric::MIN);
  std::cout << minV(0) << " " << minV(1) << "\n";


  // // Check the result
  // for(int i=0; i<nbr_slice; i++){
  //   ASSERT_TRUE(metric.getMinData()[i]      == true_min[i]);
  //   ASSERT_TRUE(metric.getMaxData()[i]      == true_max[i]);
  //   ASSERT_TRUE(metric.getMeanData()[i]     == true_mean[i]);
  //   ASSERT_TRUE(metric.getVarianceData()[i] == true_var[i]);
  //   EXPECT_NEAR(sqrt(metric.getVarianceData()[i]), true_stdev[i], 0.0001);
  // }
}

// TEST(tuckerkokkos, compute_slice_metrics_mode0){
//   // Prepare
//   typedef double scalar_t;
//   using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

//   // Create a 2x2x2 tensor
//   std::vector<int> dims = {2, 2, 2};
//   TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);
//   int mode = 0;

//   // Fill it with the entries 0:8
//   auto tensor_d = tensor.data();
//   for(int i=0; i<8; i++){
//     tensor_d(i) = i;
//   }

//   // Create a metric data storing the slice
//   Tucker::MetricData<scalar_t> metric = Tucker::compute_slice_metrics(tensor, (int)mode, Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

//   // True results (compute by hand)
//   int nbr_slice = 2;
//   scalar_t true_min[nbr_slice]   = {0.0, 1.0};
//   scalar_t true_max[nbr_slice]   = {6.0, 7.0};
//   scalar_t true_mean[nbr_slice]  = {3.0, 4.0};
//   scalar_t true_var[nbr_slice]   = {5.0, 5.0};
//   scalar_t true_stdev[nbr_slice] = {2.2360, 2.2360};

//   // Check the result
//   for(int i=0; i<nbr_slice; i++){
//     ASSERT_TRUE(metric.getMinData()[i]      == true_min[i]);
//     ASSERT_TRUE(metric.getMaxData()[i]      == true_max[i]);
//     ASSERT_TRUE(metric.getMeanData()[i]     == true_mean[i]);
//     ASSERT_TRUE(metric.getVarianceData()[i] == true_var[i]);
//     EXPECT_NEAR(sqrt(metric.getVarianceData()[i]), true_stdev[i], 0.0001);
//   }
// }

// TEST(tuckerkokkos, compute_slice_metrics_mode1){
//   // Prepare
//   typedef double scalar_t;
//   using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

//   // Create a 2x2x2 tensor
//   std::vector<int> dims = {2, 2, 2};
//   TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);
//   int mode = 1;

//   // Fill it with the entries 0:8
//   auto tensor_d = tensor.data();
//   for(int i=0; i<8; i++){
//     tensor_d(i) = i;
//   }

//   // Create a metric data storing the slice
//   Tucker::MetricData<scalar_t> metric = Tucker::compute_slice_metrics(tensor, (int)mode, Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

//   // True results (compute by hand)
//   int nbr_slice = 2;
//   scalar_t true_min[nbr_slice]   = {0.0, 2.0};
//   scalar_t true_max[nbr_slice]   = {5.0, 7.0};
//   scalar_t true_mean[nbr_slice]  = {2.5, 4.5};
//   scalar_t true_var[nbr_slice]   = {4.25, 4.25};
//   scalar_t true_stdev[nbr_slice] = {2.0615, 2.0615};

//   // Check the result
//   for(int i=0; i<nbr_slice; i++){
//     ASSERT_TRUE(metric.getMinData()[i]      == true_min[i]);
//     ASSERT_TRUE(metric.getMaxData()[i]      == true_max[i]);
//     ASSERT_TRUE(metric.getMeanData()[i]     == true_mean[i]);
//     ASSERT_TRUE(metric.getVarianceData()[i] == true_var[i]);
//     EXPECT_NEAR(sqrt(metric.getVarianceData()[i]), true_stdev[i], 0.0001);
//   }
// }

// // Originally import test here:
// // https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/serial/tests/Tucker_slice_test_nofile.cpp
// TEST(tuckerkokkos, compute_slice_metrics_2x3x5x7_allmode){
//   // Prepare
//   typedef double scalar_t;
//   using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

//   // Create a 2x3x5x7 tensor
//   std::vector<int> dims = {2, 3, 5, 7};
//   int nbr_dim = 4;
//   TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);

//   // Fill it with the entries 0:210
//   auto tensor_d = tensor.data();
//   for(int i=0; i<210; i++){
//     tensor_d(i) = i+1;
//   }

//   // True values
//   scalar_t trueData[7][4][3];
//   trueData[0][0][0] = 209;
//   trueData[0][0][1] = 1;
//   trueData[0][0][2] = 11025;
//   trueData[1][0][0] = 210;
//   trueData[1][0][1] = 2;
//   trueData[1][0][2] = 11130;
//   trueData[0][1][0] = 206;
//   trueData[0][1][1] = 1;
//   trueData[0][1][2] = 7245;
//   trueData[1][1][0] = 208;
//   trueData[1][1][1] = 3;
//   trueData[1][1][2] = 7385;
//   trueData[2][1][0] = 210;
//   trueData[2][1][1] = 5;
//   trueData[2][1][2] = 7525;
//   trueData[0][2][0] = 186;
//   trueData[0][2][1] = 1;
//   trueData[0][2][2] = 3927;
//   trueData[1][2][0] = 192;
//   trueData[1][2][1] = 7;
//   trueData[1][2][2] = 4179;
//   trueData[2][2][0] = 198;
//   trueData[2][2][1] = 13;
//   trueData[2][2][2] = 4431;
//   trueData[3][2][0] = 204;
//   trueData[3][2][1] = 19;
//   trueData[3][2][2] = 4683;
//   trueData[4][2][0] = 210;
//   trueData[4][2][1] = 25;
//   trueData[4][2][2] = 4935;
//   trueData[0][3][0] = 30;
//   trueData[0][3][1] = 1;
//   trueData[0][3][2] = 465;
//   trueData[1][3][0] = 60;
//   trueData[1][3][1] = 31;
//   trueData[1][3][2] = 1365;
//   trueData[2][3][0] = 90;
//   trueData[2][3][1] = 61;
//   trueData[2][3][2] = 2265;
//   trueData[3][3][0] = 120;
//   trueData[3][3][1] = 91;
//   trueData[3][3][2] = 3165;
//   trueData[4][3][0] = 150;
//   trueData[4][3][1] = 121;
//   trueData[4][3][2] = 4065;
//   trueData[5][3][0] = 180;
//   trueData[5][3][1] = 151;
//   trueData[5][3][2] = 4965;
//   trueData[6][3][0] = 210;
//   trueData[6][3][1] = 181;
//   trueData[6][3][2] = 5865;

//   // Checks
//   for(int i=0; i<nbr_dim; i++) {
//     TuckerOnNode::MetricData<scalar_t> metric = Tucker::compute_slice_metrics(tensor, i, Tucker::MIN + Tucker::MAX + Tucker::SUM);
//     for(int j=0; j<dims[i]; j++) {
//       std::cout << "The maximum of slice " << j << " of mode "
//           << i << " is " << metric.getMaxData()[j] << std::endl;
//       std::cout << "The minimum of slice " << j << " of mode "
//           << i << " is " << metric.getMinData()[j] << std::endl;
//       std::cout << "The sum of slice " << j << " of mode "
//           << i << " is " << metric.getSumData()[j] << std::endl;

//       ASSERT_TRUE(metric.getMaxData()[j] == trueData[j][i][0]);
//       ASSERT_TRUE(metric.getMinData()[j] == trueData[j][i][1]);
//       ASSERT_TRUE(metric.getSumData()[j] == trueData[j][i][2]);
//     }
//   }
// }
