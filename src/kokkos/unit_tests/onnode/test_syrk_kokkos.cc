
#include <gtest/gtest.h>
#include "./impl/Tucker_syrk_kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::random_device rd;
  std::mt19937 m_gen{rd()};
  dist_type m_dist;

  UnifDist(const double a, const double b) : m_dist(a, b){}
  double operator()() { return m_dist(m_gen); }
};

template<class AlphaType, class BetaType, class A_t, class C_t>
void syrk_gold_solution_opA_N(AlphaType alpha, BetaType beta,
			      const A_t &A, C_t &C)
{
  const auto n = C.extent(0);
  const auto kap = A.extent(1);
  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t i = 0; i < j+1; ++i)
    {
      AlphaType sum = {};
      for (std::size_t k = 0; k < kap; ++k) {
        sum += A(i, k) * A(j, k);
      }
      C(i,j) = beta*C(i,j) + alpha*sum;
    }
  }
}

template<class AlphaType, class BetaType, class A_t, class C_t>
void syrk_gold_solution_opA_T(AlphaType alpha, BetaType beta,
			      const A_t &A, C_t &C)
{
  const auto n = C.extent(0);
  const auto kap = A.extent(0);
  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t i = 0; i < j+1; ++i)
    {
      AlphaType sum = {};
      for (std::size_t k = 0; k < kap; ++k) {
        sum += A(k, i) * A(k, j);
      }
      C(i,j) = beta*C(i,j) + alpha*sum;
    }
  }
}

template<class T>
T clone(T v){
  T clone("clone", v.extent(0), v.extent(1));
  Kokkos::deep_copy(clone, v);
  return clone;
}

TEST(tuckerkokkos_syrk, uplo_U_opA_N)
{
  const std::size_t n = 5;
  const std::size_t k = 13;
  using value_type = double;

  // create A
  Kokkos::View<value_type**> A("A", n, k);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(45662333);
  Kokkos::fill_random(A, pool, -1.1, 5.2);

  // create C
  Kokkos::View<value_type**> C("A", n, n);
  auto C_h = Kokkos::create_mirror(C);
  const auto a = static_cast<value_type>(-1.8);
  const auto b = static_cast<value_type>(2.3);
  UnifDist<value_type> randObj(a, b);
  for (std::size_t i=0; i < n; ++i) {
    for (std::size_t j=i; j < n; ++j) {
      C_h(i,j) = randObj();
    }
  }
  Kokkos::deep_copy(C, C_h);

  auto A_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto A_clone = clone(A);
  auto A_clone_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_clone);

  auto C_clone = clone(C);
  auto C_clone_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C_clone);

  Tucker::impl::syrk_kokkos("U", "N", 1.2, A, 1.6, C);
  Kokkos::deep_copy(C_h, C);
  syrk_gold_solution_opA_N(1.2, 1.6, A_clone_h, C_clone_h);

  for (std::size_t i=0; i < n; ++i) {
    for (std::size_t j=0; j < n; ++j) {
      ASSERT_NEAR(C_h(i,j), C_clone_h(i,j), 1e-12);
      if (j < i){
	ASSERT_EQ(C_h(i,j), 0.);
      }
    }
  }
}

TEST(tuckerkokkos_syrk, uplo_U_opA_T)
{
  const std::size_t n = 5;
  const std::size_t k = 13;
  using value_type = double;

  // create A
  Kokkos::View<value_type**> A("A", k, n);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(45662333);
  Kokkos::fill_random(A, pool, -1.1, 5.2);

  // create C
  Kokkos::View<value_type**> C("A", n, n);
  auto C_h = Kokkos::create_mirror(C);
  const auto a = static_cast<value_type>(-1.8);
  const auto b = static_cast<value_type>(2.3);
  UnifDist<value_type> randObj(a, b);
  for (std::size_t i=0; i < n; ++i) {
    for (std::size_t j=i; j < n; ++j) {
      C_h(i,j) = randObj();
    }
  }
  Kokkos::deep_copy(C, C_h);

  auto A_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto A_clone = clone(A);
  auto A_clone_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_clone);

  auto C_clone = clone(C);
  auto C_clone_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C_clone);

  Tucker::impl::syrk_kokkos("U", "T", 1.2, A, 1.6, C);
  Kokkos::deep_copy(C_h, C);
  syrk_gold_solution_opA_T(1.2, 1.6, A_clone_h, C_clone_h);

  for (std::size_t i=0; i < n; ++i) {
    for (std::size_t j=0; j < n; ++j) {
      ASSERT_NEAR(C_h(i,j), C_clone_h(i,j), 1e-12);
      if (j < i){
	ASSERT_EQ(C_h(i,j), 0.);
      }
    }
  }
}
