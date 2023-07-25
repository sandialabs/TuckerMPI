#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

TEST(tuckerkokkos, compute_slice_metrics_mode2){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 2x2x2 tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);
  int mode = 2;

  // Fill it with the entries 0:8
  auto tensor_d = tensor.data();
  for(int i=0; i<8; i++){
    tensor_d(i) = i;
  }

  // Create a metric data storing the slice
  Tucker::MetricData<scalar_t> metric = Tucker::compute_slice_metrics(tensor, (int)mode, Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

  // True results (compute by hand)
  int nbr_slice = 2;
  scalar_t true_min[nbr_slice]   = {0.0, 4.0};
  scalar_t true_max[nbr_slice]   = {3.0, 7.0};
  scalar_t true_mean[nbr_slice]  = {1.5, 5.5};
  scalar_t true_var[nbr_slice]   = {1.25, 1.25};
  scalar_t true_stdev[nbr_slice] = {1.1180, 1.1180};

  // Check the result
  for(int i=0; i<nbr_slice; i++){
    ASSERT_TRUE(metric.getMinData()[i]      == true_min[i]);
    ASSERT_TRUE(metric.getMaxData()[i]      == true_max[i]);
    ASSERT_TRUE(metric.getMeanData()[i]     == true_mean[i]);
    ASSERT_TRUE(metric.getVarianceData()[i] == true_var[i]);
    EXPECT_NEAR(sqrt(metric.getVarianceData()[i]), true_stdev[i], 0.0001);
  }
}

TEST(tuckerkokkos, compute_slice_metrics_mode0){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 2x2x2 tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);
  int mode = 0;

  // Fill it with the entries 0:8
  auto tensor_d = tensor.data();
  for(int i=0; i<8; i++){
    tensor_d(i) = i;
  }

  // Create a metric data storing the slice
  Tucker::MetricData<scalar_t> metric = Tucker::compute_slice_metrics(tensor, (int)mode, Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

  // True results (compute by hand)
  int nbr_slice = 2;
  scalar_t true_min[nbr_slice]   = {0.0, 1.0};
  scalar_t true_max[nbr_slice]   = {6.0, 7.0};
  scalar_t true_mean[nbr_slice]  = {3.0, 4.0};
  scalar_t true_var[nbr_slice]   = {5.0, 5.0};
  scalar_t true_stdev[nbr_slice] = {2.2360, 2.2360};

  // Check the result
  for(int i=0; i<nbr_slice; i++){
    ASSERT_TRUE(metric.getMinData()[i]      == true_min[i]);
    ASSERT_TRUE(metric.getMaxData()[i]      == true_max[i]);
    ASSERT_TRUE(metric.getMeanData()[i]     == true_mean[i]);
    ASSERT_TRUE(metric.getVarianceData()[i] == true_var[i]);
    EXPECT_NEAR(sqrt(metric.getVarianceData()[i]), true_stdev[i], 0.0001);
  }
}

TEST(tuckerkokkos, compute_slice_metrics_mode1){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 2x2x2 tensor
  std::vector<int> dims = {2, 2, 2};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);
  int mode = 1;

  // Fill it with the entries 0:8
  auto tensor_d = tensor.data();
  for(int i=0; i<8; i++){
    tensor_d(i) = i;
  }

  // Create a metric data storing the slice
  Tucker::MetricData<scalar_t> metric = Tucker::compute_slice_metrics(tensor, (int)mode, Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

  // True results (compute by hand)
  int nbr_slice = 2;
  scalar_t true_min[nbr_slice]   = {0.0, 2.0};
  scalar_t true_max[nbr_slice]   = {5.0, 7.0};
  scalar_t true_mean[nbr_slice]  = {2.5, 4.5};
  scalar_t true_var[nbr_slice]   = {4.25, 4.25};
  scalar_t true_stdev[nbr_slice] = {2.0615, 2.0615};

  // Check the result
  for(int i=0; i<nbr_slice; i++){
    ASSERT_TRUE(metric.getMinData()[i]      == true_min[i]);
    ASSERT_TRUE(metric.getMaxData()[i]      == true_max[i]);
    ASSERT_TRUE(metric.getMeanData()[i]     == true_mean[i]);
    ASSERT_TRUE(metric.getVarianceData()[i] == true_var[i]);
    EXPECT_NEAR(sqrt(metric.getVarianceData()[i]), true_stdev[i], 0.0001);
  }
}