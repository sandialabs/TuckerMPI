#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

using namespace TuckerOnNode;

struct TuckerTensorFixA : public ::testing::Test
{
protected:
  virtual void TearDown() override{}

  virtual void SetUp() override
  {
    /*
      mode = 0:
      factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      mode = 1:
      factors = [0.9, 1., 1.1, 1.2]

      mode = 2:
      factors = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    */

    factors_ = factors_rank1_view_t("U", 18);
    perModeSlicingInfo_ = slicing_info_view_t("pmsi", 3);

    // modify on host
    auto factors_h = Kokkos::create_mirror(factors_);
    double begin = 0.0;
    for (size_t i=0; i<=17; ++i){
      factors_h(i) = begin + 0.1;
      begin = factors_h(i);
    }
    Kokkos::deep_copy(factors_, factors_h);

    auto & info = perModeSlicingInfo_(0);
    info.startIndex        = 0;
    info.endIndexExclusive = 8;
    info.extent0           = 2;
    info.extent1           = 4;

    auto & info1 = perModeSlicingInfo_(1);
    info1.startIndex        = 8;
    info1.endIndexExclusive = 12;
    info1.extent0           = 2;
    info1.extent1	   = 2;

    auto & info2 = perModeSlicingInfo_(2);
    info2.startIndex        = 12;
    info2.endIndexExclusive = 18;
    info2.extent0           = 2;
    info2.extent1	   = 3;
  }

  using scalar_t = double;
  using core_tensor_t = Tensor<scalar_t>;
  using memory_space = typename core_tensor_t::traits::memory_space;
  using tt_t = Tucker::TuckerTensor<core_tensor_t>;
  using tt_traits = typename tt_t::traits;
  using slicing_info_view_t = Kokkos::View<Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;
  using factors_rank1_view_t = Kokkos::View<scalar_t*, Kokkos::LayoutLeft, memory_space>;

  const int rank = 3;
  core_tensor_t core_;
  factors_rank1_view_t factors_;
  slicing_info_view_t perModeSlicingInfo_;
};

template <class TTensorType>
void verifyDefault(TTensorType T){
  ASSERT_EQ(T.rank(), -1);
  auto factors = T.factorMatrix(0);
  ASSERT_EQ(factors.extent(0), 0);
  ASSERT_EQ(factors.extent(1), 0);
}

template <class TTensorType>
void verify1(TTensorType T)
{
  ASSERT_EQ(T.rank(), 3);
  auto U0 = T.factorMatrix(0);
  auto U1 = T.factorMatrix(1);
  auto U2 = T.factorMatrix(2);
  auto U0_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U0);
  auto U1_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U1);
  auto U2_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U2);

  ASSERT_EQ(U0_h.extent(0), 2);
  ASSERT_EQ(U0_h.extent(1), 4);
  ASSERT_DOUBLE_EQ(U0_h(0,0), 0.1);
  ASSERT_DOUBLE_EQ(U0_h(0,1), 0.3);
  ASSERT_DOUBLE_EQ(U0_h(0,2), 0.5);
  ASSERT_DOUBLE_EQ(U0_h(0,3), 0.7);
  ASSERT_DOUBLE_EQ(U0_h(1,0), 0.2);
  ASSERT_DOUBLE_EQ(U0_h(1,1), 0.4);
  ASSERT_DOUBLE_EQ(U0_h(1,2), 0.6);
  ASSERT_DOUBLE_EQ(U0_h(1,3), 0.8);

  ASSERT_EQ(U1_h.extent(0), 2);
  ASSERT_EQ(U1_h.extent(1), 2);
  ASSERT_DOUBLE_EQ(U1_h(0,0), 0.9);
  ASSERT_DOUBLE_EQ(U1_h(0,1), 1.1);
  ASSERT_DOUBLE_EQ(U1_h(1,0), 1.0);
  ASSERT_DOUBLE_EQ(U1_h(1,1), 1.2);

  ASSERT_EQ(U2_h.extent(0), 2);
  ASSERT_EQ(U2_h.extent(1), 3);
  ASSERT_DOUBLE_EQ(U2_h(0,0), 1.3);
  ASSERT_DOUBLE_EQ(U2_h(0,1), 1.5);
  ASSERT_DOUBLE_EQ(U2_h(0,2), 1.7);
  ASSERT_DOUBLE_EQ(U2_h(1,0), 1.4);
  ASSERT_DOUBLE_EQ(U2_h(1,1), 1.6);
  ASSERT_DOUBLE_EQ(U2_h(1,2), 1.8);
}

template <class TTensorType>
void verify2(TTensorType T)
{
  ASSERT_EQ(T.rank(), 3);

  auto U0 = T.factorMatrix(0);
  auto U1 = T.factorMatrix(1);
  auto U2 = T.factorMatrix(2);
  auto U0_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U0);
  auto U1_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U1);
  auto U2_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U2);

  ASSERT_EQ(U0_h.extent(0), 2);
  ASSERT_EQ(U0_h.extent(1), 4);
  ASSERT_DOUBLE_EQ(U0_h(0,0), 0.1);
  ASSERT_DOUBLE_EQ(U0_h(0,1), 0.3);
  ASSERT_DOUBLE_EQ(U0_h(0,2), 0.5);
  ASSERT_DOUBLE_EQ(U0_h(0,3), 0.7);
  ASSERT_DOUBLE_EQ(U0_h(1,0), 2.0); // changed
  ASSERT_DOUBLE_EQ(U0_h(1,1), 0.4);
  ASSERT_DOUBLE_EQ(U0_h(1,2), 0.6);
  ASSERT_DOUBLE_EQ(U0_h(1,3), 0.8);

  ASSERT_EQ(U1_h.extent(0), 2);
  ASSERT_EQ(U1_h.extent(1), 2);
  ASSERT_DOUBLE_EQ(U1_h(0,0), 0.9);
  ASSERT_DOUBLE_EQ(U1_h(0,1), 1.1);
  ASSERT_DOUBLE_EQ(U1_h(1,0), 10.0); // changed
  ASSERT_DOUBLE_EQ(U1_h(1,1), 1.2);

  ASSERT_EQ(U2_h.extent(0), 2);
  ASSERT_EQ(U2_h.extent(1), 3);
  ASSERT_DOUBLE_EQ(U2_h(0,0), 1.3);
  ASSERT_DOUBLE_EQ(U2_h(0,1), 1.5);
  ASSERT_DOUBLE_EQ(U2_h(0,2), 17.0); // changed
  ASSERT_DOUBLE_EQ(U2_h(1,0), 1.4);
  ASSERT_DOUBLE_EQ(U2_h(1,1), 1.6);
  ASSERT_DOUBLE_EQ(U2_h(1,2), 1.8);
}

template <class TTensorType>
void change2(TTensorType & T)
{
  auto fac0 = T.factorMatrix(0);
  auto fac0_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fac0);
  fac0_h(1,0) = 2.0;
  Kokkos::deep_copy(fac0, fac0_h);

  auto fac1 = T.factorMatrix(1);
  auto fac1_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fac1);
  fac1_h(1,0) = 10.;
  Kokkos::deep_copy(fac1, fac1_h);

  auto fac2 = T.factorMatrix(2);
  auto fac2_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fac2);
  fac2_h(0,2) = 17.0;
  Kokkos::deep_copy(fac2, fac2_h);
}


TEST_F(TuckerTensorFixA, default_cnstr){
  tt_t T;
  verifyDefault(T);
}

TEST_F(TuckerTensorFixA, cnstr1){
  tt_t T(core_, factors_, perModeSlicingInfo_);
  verify1(T);
}

TEST_F(TuckerTensorFixA, copy_cnstr){
  tt_t T(core_, factors_, perModeSlicingInfo_);
  verify1(T);
  auto T2 = T;
  verify1(T2);
}

TEST_F(TuckerTensorFixA, copy_cnstr_has_view_semantics){
  tt_t T(core_, factors_, perModeSlicingInfo_);
  verify1(T);
  auto T2 = T;
  verify1(T2);
  change2(T2);
  verify2(T2);
  verify2(T);
}

TEST_F(TuckerTensorFixA, copy_assign_has_view_semantics){
  tt_t T(core_, factors_, perModeSlicingInfo_);
  verify1(T);
  tt_t T2;
  verifyDefault(T2);
  T2 = T;
  verify1(T2);
  change2(T2);
  verify2(T2);
  verify2(T);
}

TEST_F(TuckerTensorFixA, move_cnstr){
  tt_t T(core_, factors_, perModeSlicingInfo_);
  verify1(T);
  tt_t T2(std::move(T));
  verify1(T2);
  change2(T2);
  verify2(T2);
}

TEST_F(TuckerTensorFixA, move_assign){
  tt_t T(core_, factors_, perModeSlicingInfo_);
  verify1(T);
  tt_t T2;
  T2 = std::move(T);
  verify1(T2);
}

TEST_F(TuckerTensorFixA, copy_cnstr_const_view){
  tt_t T(core_, factors_, perModeSlicingInfo_);

  using core_const_tensor_t = TuckerOnNode::Tensor<const scalar_t>;
  using tt_t = Tucker::TuckerTensor<core_const_tensor_t>;

  tt_t b = T;
  auto f = b.factorMatrix(0);
  //f(0,0) = 1; //this MUST fail to compile for the test to be correct
}

// FINISH
