#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

using namespace TuckerOnNode;

struct TensorGramEigvalsFixA : public ::testing::Test
{
protected:
  virtual void TearDown() override{}

  virtual void SetUp() override
  {
    /*
      mode = 0:
      eigvals = 1., 2., 3., 4.

      mode = 1:
      eigvals = 5., 6.

      mode = 2:
      eigvals = 7., 8., 9.
    */

    eigvals_ = eigvals_rank1_view_t("evals", 9);
    perModeSlicingInfo_ = slicing_info_view_t("pmsi", 3);

    // modify on host
    auto eigvals_h = Kokkos::create_mirror(eigvals_);
    for (size_t i=0; i<=8; ++i){
      eigvals_h(i) = (double) (i+1);
    }
    Kokkos::deep_copy(eigvals_, eigvals_h);

    auto & info = perModeSlicingInfo_(0);
    info.startIndex        = 0;
    info.endIndexExclusive = 4;

    auto & info1 = perModeSlicingInfo_(1);
    info1.startIndex        = 4;
    info1.endIndexExclusive = 6;

    auto & info2 = perModeSlicingInfo_(2);
    info2.startIndex        = 6;
    info2.endIndexExclusive = 9;
  }

  using scalar_t = double;
  using tt_t = TensorGramEigenvalues<scalar_t>;
  using slicing_info_view_t = Kokkos::View<Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;
  using eigvals_rank1_view_t = Kokkos::View<scalar_t*, Kokkos::LayoutLeft>;

  const int rank = 3;
  eigvals_rank1_view_t eigvals_;
  slicing_info_view_t perModeSlicingInfo_;
};

template <class ContType>
void verifyDefault(ContType T){
  ASSERT_EQ(T.rank(), -1);
  auto eigvals = T[0];
  ASSERT_EQ(eigvals.extent(0), 0);
}

template <class ContType>
void verify1(ContType T)
{
  ASSERT_EQ(T.rank(), 3);

  auto evals0 = T[0];
  auto evals1 = T[1];
  auto evals2 = T[2];
  auto evals0_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals0);
  auto evals1_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals1);
  auto evals2_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals2);

  ASSERT_EQ(evals0_h.extent(0), 4);
  ASSERT_DOUBLE_EQ(evals0_h(0), 1.);
  ASSERT_DOUBLE_EQ(evals0_h(1), 2.);
  ASSERT_DOUBLE_EQ(evals0_h(2), 3.);
  ASSERT_DOUBLE_EQ(evals0_h(3), 4.);

  ASSERT_EQ(evals1_h.extent(0), 2);
  ASSERT_DOUBLE_EQ(evals1_h(0), 5.);
  ASSERT_DOUBLE_EQ(evals1_h(1), 6.);

  ASSERT_EQ(evals2_h.extent(0), 3);
  ASSERT_DOUBLE_EQ(evals2_h(0), 7.);
  ASSERT_DOUBLE_EQ(evals2_h(1), 8.);
  ASSERT_DOUBLE_EQ(evals2_h(2), 9.);
}

template <class ContType>
void verify2(ContType T)
{
  ASSERT_EQ(T.rank(), 3);

  auto evals0 = T[0];
  auto evals1 = T[1];
  auto evals2 = T[2];

  auto evals0_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals0);
  auto evals1_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals1);
  auto evals2_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals2);

  ASSERT_EQ(evals0_h.extent(0), 4);
  ASSERT_DOUBLE_EQ(evals0_h(0), 1.);
  ASSERT_DOUBLE_EQ(evals0_h(1), 20.); // this changed
  ASSERT_DOUBLE_EQ(evals0_h(2), 3.);
  ASSERT_DOUBLE_EQ(evals0_h(3), 4.);

  ASSERT_EQ(evals1_h.extent(0), 2);
  ASSERT_DOUBLE_EQ(evals1_h(0), 5.);
  ASSERT_DOUBLE_EQ(evals1_h(1), 6.);

  ASSERT_EQ(evals2_h.extent(0), 3);
  ASSERT_DOUBLE_EQ(evals2_h(0), 7.);
  ASSERT_DOUBLE_EQ(evals2_h(1), 8.);
  ASSERT_DOUBLE_EQ(evals2_h(2), 9.);
}

template <class ContType>
void change2(ContType & T)
{
  auto evals0 = T[0];
  auto evals0_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), evals0);
  evals0_h(1) = 20.;
  Kokkos::deep_copy(evals0, evals0_h);
}

TEST_F(TensorGramEigvalsFixA, default_cnstr){
  tt_t T;
  verifyDefault(T);
}

TEST_F(TensorGramEigvalsFixA, cnstr1){
  tt_t T(eigvals_, perModeSlicingInfo_);
  verify1(T);
}

TEST_F(TensorGramEigvalsFixA, copy_cnstr){
  tt_t T(eigvals_, perModeSlicingInfo_);
  verify1(T);
  auto T2 = T;
  verify1(T2);
}

TEST_F(TensorGramEigvalsFixA, copy_cnstr_has_view_semantics){
  tt_t T(eigvals_, perModeSlicingInfo_);
  verify1(T);
  auto T2 = T;
  verify1(T2);
  change2(T2);
  verify2(T2);
  verify2(T);
}

TEST_F(TensorGramEigvalsFixA, copy_assign_has_view_semantics){
  tt_t T(eigvals_, perModeSlicingInfo_);
  verify1(T);
  tt_t T2;
  verifyDefault(T2);
  T2 = T;
  verify1(T2);
  change2(T2);
  verify2(T2);
  verify2(T);
}

TEST_F(TensorGramEigvalsFixA, move_cnstr){
  tt_t T(eigvals_, perModeSlicingInfo_);
  verify1(T);
  tt_t T2(std::move(T));
  verify1(T2);
  change2(T2);
  verify2(T2);
}

TEST_F(TensorGramEigvalsFixA, move_assign){
  tt_t T(eigvals_, perModeSlicingInfo_);
  verify1(T);
  tt_t T2;
  T2 = std::move(T);
  verify1(T2);
}

TEST_F(TensorGramEigvalsFixA, copy_cnstr_const_view){
  tt_t T(eigvals_, perModeSlicingInfo_);
  TensorGramEigenvalues<const scalar_t> b = T;
  auto f = b[0];
  //f(0) = 1; //this MUST fail to compile for the test to be correct
}

// FINISH
