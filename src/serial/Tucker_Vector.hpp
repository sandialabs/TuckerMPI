/*
 * Tucker_Vector.hpp
 *
 *  Created on: Mar 15, 2017
 *      Author: amklinv
 */

#ifndef SERIAL_TUCKER_VECTOR_HPP_
#define SERIAL_TUCKER_VECTOR_HPP_

#include "Tucker_Matrix.hpp"

namespace Tucker {

template<class scalar_t>
class Vector : public Matrix<scalar_t> {
public:
  /** \brief Constructor
   * \param[in] nrows Number of rows
   * \param[in] ncols Number of columns
   */
  Vector(const int nrows) :
    Matrix<scalar_t>::Matrix(nrows)
  {

  }

  /** \brief Element access
   *
   * \param[in] i Index of entry to be returned
   *
   * \exception std::out_of_range \a i is not in the range [0, nsz_)
   */
  scalar_t& operator[](const int i) {
    if(i < 0 || i >= nrows())
      throw std::out_of_range("invalid index");
    return this->data_[i];
  }

  /** \brief Const element access
   *
   * \param[in] i Index of entry to be returned
   *
   * \exception std::out_of_range \a i is not in the range [0, nsz_)
   */
  const scalar_t& operator[](const int i) const {
    if(i < 0 || i >= nrows())
      throw std::out_of_range("invalid index");
    return this->data_[i];
  }

  /// Returns the number of rows
  int nrows() const
  {
    return this->I_[0];
  }

  /// Returns the number of columns
  int ncols() const
  {
    return 1;
  }

private:
  /// @cond EXCLUDE
  // Disables the copy constructor
  Vector(const Vector<scalar_t>& m);
  /// @endcond
};

// Explicit instantiations to build static library for both single and double precision
template class Vector<float>;
template class Vector<double>;

} // end of namespace Tucker



#endif /* SERIAL_TUCKER_VECTOR_HPP_ */
