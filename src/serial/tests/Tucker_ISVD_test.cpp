/**
 * @file
 * @brief Incremental SVD unit tests
 * @author Saibal De
 */

#include "Tucker_ISVD.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "Tucker.hpp"
#include "Tucker_Matrix.hpp"
#include "Tucker_Util.hpp"
#include "Tucker_Vector.hpp"

#ifdef TEST_SINGLE
typedef float scalar_t;
#else
typedef double scalar_t;
#endif

const scalar_t PI = 4 * std::atan(static_cast<scalar_t>(1));

class DataSource {
public:
  DataSource(int m, scalar_t dt, scalar_t t_cut, int k_max) {
    m_ = m;

    const scalar_t dx = 2 * PI / (m_ - 1);
    x_ = Tucker::MemoryManager::safe_new<Tucker::Vector<scalar_t>>(m_);
    for (int i = 0; i < m_; ++i) {
      x_->data()[i] = i * dx;
    }

    dt_ = dt;
    t_ = 0;
    t_cut_ = t_cut;

    k_max_ = k_max;
  }

  ~DataSource() { Tucker::MemoryManager::safe_delete(x_); }

  void constructInitialMatrix(int n, Tucker::Matrix<scalar_t> *A) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m_; ++i) {
        scalar_t &entry = A->data()[i + j * m_];
        entry = 0;
        for (int k = 0; k < k_max_ / 2; ++k) {
          entry += std::sin(k * (x_->data()[i] - (j + 1) * dt_));
        }
        if (t_ >= t_cut_) {
          for (int k = k_max_ / 2; k < k_max_; ++k) {
            entry += std::sin(k * (x_->data()[i] - (j + 1) * dt_ + t_cut_));
          }
        }
      }
    }

    t_ = n * dt_;
  }

  void constructNextVector(Tucker::Vector<scalar_t> *v) {
    t_ += dt_;

    for (int i = 0; i < m_; ++i) {
      scalar_t &entry = v->data()[i];
      entry = 0;
      for (int k = 0; k < k_max_ / 2; ++k) {
        entry += std::sin(k * (x_->data()[i] - t_));
      }
      if (t_ >= t_cut_) {
        for (int k = k_max_ / 2; k < k_max_; ++k) {
          entry += std::sin(k * (x_->data()[i] - t_ + t_cut_));
        }
      }
    }
  }

private:
  int m_;
  Tucker::Vector<scalar_t> *x_;
  scalar_t dt_;
  scalar_t t_;
  scalar_t t_cut_;
  int k_max_;
};

bool testInitializeFactors() {
  bool success = true;

  const int m = 100;
  const int n = 10;
  const scalar_t dt = 1;
  const scalar_t t_cut = 50;
  const int k_max = 5;
  const scalar_t tolerance = 1.0e-2;

  Tucker::Matrix<scalar_t> *A =
      Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(m, n);

  DataSource data_source(m, dt, t_cut, k_max);
  data_source.constructInitialMatrix(n, A);

  Tucker::Matrix<scalar_t> *G = Tucker::computeGram(A, 1);

  scalar_t *s;
  Tucker::Matrix<scalar_t> *U;
  const scalar_t thresh = tolerance * std::sqrt(A->norm2());
  Tucker::computeEigenpairs(G, s, U, thresh);

  for (int j = 0; j < U->ncols(); ++j) {
    s[j] = std::sqrt(s[j]);
  }

  Tucker::ISVD<scalar_t> isvd;
  isvd.initializeFactors(U, s, A);

  if (isvd.getRelativeError() >= tolerance) {
    std::cout << "failed: relative error estimate check" << std::endl;
    success = false;
  }

  if (isvd.getRightSingularVectorsError() >=
      10 * std::numeric_limits<scalar_t>::epsilon()) {
    std::cout << "failed: right singular vectors orthogonality check"
              << std::endl;
    success = false;
  }

  Tucker::MemoryManager::safe_delete(A);
  Tucker::MemoryManager::safe_delete(G);
  Tucker::MemoryManager::safe_delete_array(s, n);
  Tucker::MemoryManager::safe_delete(U);

  return success;
}

bool testUpdateRightSingularVectors() {
  bool success = true;

  const int m = 100;
  const int n = 10;
  const scalar_t dt = 1;
  const scalar_t t_cut = 50;
  const int k_max = 5;
  const scalar_t tolerance = 1.0e-2;

  Tucker::Matrix<scalar_t> *A =
      Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(m, n);

  DataSource data_source(m, dt, t_cut, k_max);
  data_source.constructInitialMatrix(n, A);

  Tucker::Matrix<scalar_t> *G = Tucker::computeGram(A, 1);

  scalar_t *s;
  Tucker::Matrix<scalar_t> *U;
  const scalar_t thresh = tolerance * std::sqrt(A->norm2());
  Tucker::computeEigenpairs(G, s, U, thresh);

  for (int j = 0; j < U->ncols(); ++j) {
    s[j] = std::sqrt(s[j]);
  }

  Tucker::ISVD<scalar_t> isvd;
  isvd.initializeFactors(U, s, A);

  // TODO: write better test
  isvd.updateRightSingularVectors(0, U, U);

  Tucker::MemoryManager::safe_delete(A);
  Tucker::MemoryManager::safe_delete(G);
  Tucker::MemoryManager::safe_delete_array(s, n);
  Tucker::MemoryManager::safe_delete(U);

  return success;
}

bool testUpdateFactors() {
  bool success = true;

  const int m = 100;
  const int n_init = 10;
  const int n = 10000;
  const scalar_t dt = 1.0e-01;
  const scalar_t t_cut = 50;
  const int k_max = 5;
  const scalar_t tolerance = 1.0e-01;

  Tucker::Matrix<scalar_t> *A =
      Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(m, n_init);

  DataSource data_source(m, dt, t_cut, k_max);
  data_source.constructInitialMatrix(n_init, A);

  Tucker::Matrix<scalar_t> *G = Tucker::computeGram(A, 1);

  scalar_t *s;
  Tucker::Matrix<scalar_t> *U;
  const scalar_t thresh = tolerance * std::sqrt(A->norm2());
  Tucker::computeEigenpairs(G, s, U, thresh);

  for (int j = 0; j < U->ncols(); ++j) {
    s[j] = std::sqrt(s[j]);
  }

  Tucker::ISVD<scalar_t> isvd;
  isvd.initializeFactors(U, s, A);
  if (isvd.getRelativeError() >= tolerance) {
    std::cout << "failed: factor initialization error constraint" << std::endl;
    success = false;
  }

  std::cout << std::setw(6) << isvd.nrows() << " " << std::flush;
  std::cout << std::setw(6) << isvd.rank() << " " << std::flush;
  std::cout << std::scientific;
  std::cout << std::setw(12) << std::setprecision(6)
            << isvd.getRelativeError() << " " << std::flush;
  std::cout << std::defaultfloat;
  std::cout << std::endl;

  Tucker::Vector<scalar_t> *v =
      Tucker::MemoryManager::safe_new<Tucker::Vector<scalar_t>>(m);

  for (int j = n_init; j < n; ++j) {
    data_source.constructNextVector(v);
    isvd.updateFactorsWithNewSlice(v, tolerance);
    if (isvd.getRelativeError() >= tolerance) {
      std::cout << "failed: factor update error constraint" << std::endl;
      success = false;
      break;
    }
    if ((j + 1) % 100 == 0) {
      std::cout << std::setw(6) << isvd.nrows() << " " << std::flush;
      std::cout << std::setw(6) << isvd.rank() << " " << std::flush;
      std::cout << std::scientific;
      std::cout << std::setw(12) << std::setprecision(6)
                << isvd.getRelativeError() << " " << std::flush;
      std::cout << std::defaultfloat;
      std::cout << std::endl;
    }
  }

  Tucker::MemoryManager::safe_delete(A);
  Tucker::MemoryManager::safe_delete(v);
  Tucker::MemoryManager::safe_delete_array(s, n_init);
  Tucker::MemoryManager::safe_delete(U);
  Tucker::MemoryManager::safe_delete(G);

  return success;
}

int main() {
  bool test1 = testInitializeFactors();
  bool test2 = testUpdateFactors();

  if (test1 && test2) {
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
