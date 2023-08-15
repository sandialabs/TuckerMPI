#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

template<class MetricsDataHost>
void check(int nSlice,
	   std::vector<double> const & true_min,
	   std::vector<double> const & true_max,
	   std::vector<double> const & true_mean,
	   std::vector<double> const & true_var,
	   MetricsDataHost metrics_h)
{
  auto minV = metrics_h.get(Tucker::Metric::MIN);
  auto maxV = metrics_h.get(Tucker::Metric::MAX);
  auto meanV = metrics_h.get(Tucker::Metric::MEAN);
  auto varV = metrics_h.get(Tucker::Metric::VARIANCE);
  for(int i=0; i<nSlice; i++){
    ASSERT_TRUE(minV[i] == true_min[i]);
    ASSERT_TRUE(maxV[i] == true_max[i]);
    ASSERT_TRUE(meanV[i]== true_mean[i]);
    ASSERT_TRUE(varV[i] == true_var[i]);
  }
}

TEST(tuckerkokkos, compute_slice_metrics_mode0){
  using scalar_t = double;

  constexpr int targetMode = 0;

  // Create & fill tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  auto tensor_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), tensor);
  auto tensor_data = tensor_h.data();
  for(int i=0; i<8; i++){
    tensor_data(i) = i;
  }
  Tucker::deep_copy(tensor, tensor_h);

  // compute metrics
  auto metrics = TuckerOnNode::compute_slice_metrics(tensor, targetMode, Tucker::defaultMetrics);
  auto metrics_h = Tucker::create_mirror(metrics);
  Tucker::deep_copy(metrics_h, metrics);

  // True results (compute by hand)
  int nslices = 2;
  std::vector<scalar_t> true_min {0.0, 1.0};
  std::vector<scalar_t> true_max {6.0, 7.0};
  std::vector<scalar_t> true_mean{3.0, 4.0};
  std::vector<scalar_t> true_var {5.0, 5.0};
  check(nslices, true_min, true_max, true_mean, true_var, metrics_h);
}

TEST(tuckerkokkos, compute_slice_metrics_mode1){
  using scalar_t = double;

  constexpr int targetMode = 1;

  // Create & fill tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  auto tensor_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), tensor);
  auto tensor_data = tensor_h.data();
  for(int i=0; i<8; i++){
    tensor_data(i) = i;
  }
  Tucker::deep_copy(tensor, tensor_h);

  // compute metrics
  auto metrics = TuckerOnNode::compute_slice_metrics(tensor, targetMode, Tucker::defaultMetrics);
  auto metrics_h = Tucker::create_mirror(metrics);
  Tucker::deep_copy(metrics_h, metrics);

  // True results (compute by hand)
  int nslices = 2;
  std::vector<scalar_t> true_min = {0.0, 2.0};
  std::vector<scalar_t> true_max = {5.0, 7.0};
  std::vector<scalar_t> true_mean= {2.5, 4.5};
  std::vector<scalar_t> true_var = {4.25, 4.25};
  check(nslices, true_min, true_max, true_mean, true_var, metrics_h);
}

TEST(tuckerkokkos, compute_slice_metrics_mode2){
  using scalar_t = double;

  constexpr int targetMode = 2;

  // Create & fill tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  auto tensor_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), tensor);
  auto tensor_data = tensor_h.data();
  for(int i=0; i<8; i++){
    tensor_data(i) = i;
  }
  Tucker::deep_copy(tensor, tensor_h);

  // compute metrics
  auto metrics = TuckerOnNode::compute_slice_metrics(tensor, targetMode, Tucker::defaultMetrics);
  auto metrics_h = Tucker::create_mirror(metrics);
  Tucker::deep_copy(metrics_h, metrics);

  // True results (compute by hand)
  int nslices = 2;
  std::vector<scalar_t> true_min = {0.0, 4.0};
  std::vector<scalar_t> true_max = {3.0, 7.0};
  std::vector<scalar_t> true_mean= {1.5, 5.5};
  std::vector<scalar_t> true_var = {1.25, 1.25};
  check(nslices, true_min, true_max, true_mean, true_var, metrics_h);
}


// Originally import test here:
// https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/serial/tests/Tucker_slice_test_nofile.cpp
TEST(tuckerkokkos, compute_slice_metrics_2x3x5x7_allmode)
{
  using scalar_t = double;

  // Create tensor
  TuckerOnNode::Tensor<scalar_t> tensor({2, 3, 5, 7});
  auto tensor_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), tensor);
  auto tensor_data = tensor_h.data();
  for(int i=0; i<tensor.size(); i++){
    tensor_data(i) = i+1;
  }
  Tucker::deep_copy(tensor, tensor_h);

  scalar_t trueData[7][4][3];
  trueData[0][0][0] = 209;
  trueData[0][0][1] = 1;
  trueData[0][0][2] = 11025;
  trueData[1][0][0] = 210;
  trueData[1][0][1] = 2;
  trueData[1][0][2] = 11130;
  trueData[0][1][0] = 206;
  trueData[0][1][1] = 1;
  trueData[0][1][2] = 7245;
  trueData[1][1][0] = 208;
  trueData[1][1][1] = 3;
  trueData[1][1][2] = 7385;
  trueData[2][1][0] = 210;
  trueData[2][1][1] = 5;
  trueData[2][1][2] = 7525;
  trueData[0][2][0] = 186;
  trueData[0][2][1] = 1;
  trueData[0][2][2] = 3927;
  trueData[1][2][0] = 192;
  trueData[1][2][1] = 7;
  trueData[1][2][2] = 4179;
  trueData[2][2][0] = 198;
  trueData[2][2][1] = 13;
  trueData[2][2][2] = 4431;
  trueData[3][2][0] = 204;
  trueData[3][2][1] = 19;
  trueData[3][2][2] = 4683;
  trueData[4][2][0] = 210;
  trueData[4][2][1] = 25;
  trueData[4][2][2] = 4935;
  trueData[0][3][0] = 30;
  trueData[0][3][1] = 1;
  trueData[0][3][2] = 465;
  trueData[1][3][0] = 60;
  trueData[1][3][1] = 31;
  trueData[1][3][2] = 1365;
  trueData[2][3][0] = 90;
  trueData[2][3][1] = 61;
  trueData[2][3][2] = 2265;
  trueData[3][3][0] = 120;
  trueData[3][3][1] = 91;
  trueData[3][3][2] = 3165;
  trueData[4][3][0] = 150;
  trueData[4][3][1] = 121;
  trueData[4][3][2] = 4065;
  trueData[5][3][0] = 180;
  trueData[5][3][1] = 151;
  trueData[5][3][2] = 4965;
  trueData[6][3][0] = 210;
  trueData[6][3][1] = 181;
  trueData[6][3][2] = 5865;

  // Checks
  const std::array<Tucker::Metric, 3> metricIDs{Tucker::Metric::MAX,
  Tucker::Metric::MIN, Tucker::Metric::SUM};

  for(int mode=0; mode<tensor.rank(); mode++) {
    auto metrics = TuckerOnNode::compute_slice_metrics(tensor, mode, metricIDs);
    auto metrics_h = Tucker::create_mirror(metrics);
    Tucker::deep_copy(metrics_h, metrics);

    auto maxV = metrics_h.get(Tucker::Metric::MAX);
    auto minV = metrics_h.get(Tucker::Metric::MIN);
    auto sumV = metrics_h.get(Tucker::Metric::SUM);
    for(int j=0; j<tensor.extent(mode); j++) {
      // std::cout << "The *computed* maximum of slice " << j << " of mode "
      // 		<< mode << " is " << maxV(j) << std::endl;
      EXPECT_TRUE(maxV[j] == trueData[j][mode][0]);
      EXPECT_TRUE(minV[j] == trueData[j][mode][1]);
      EXPECT_TRUE(sumV[j] == trueData[j][mode][2]);
    }
  }
}
