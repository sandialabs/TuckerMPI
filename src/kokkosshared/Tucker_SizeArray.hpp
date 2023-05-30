
#ifndef SIZEARRAY_HPP_
#define SIZEARRAY_HPP_

#include <iostream>
#include <vector>

namespace TuckerKokkos {

class SizeArray
{
public:
  SizeArray() = default;
  SizeArray(const int n) : nsz_(n), sz_(n){}

  int size() const { return nsz_; }
  int* data() { return sz_.data(); }
  const int* data() const { return sz_.data(); }

  int& operator[](const int i) {
    if(i < 0 || i >= nsz_){ throw std::out_of_range("invalid index"); }
    return sz_[i];
  }

  const int& operator[](const int i) const {
    if(i < 0 || i >= nsz_){ throw std::out_of_range("invalid index"); }
    return sz_[i];
  }

  size_t prod() const {
    return prod(0, nsz_-1);
  }

  size_t prod(const int low, const int high, const int defaultReturnVal = -1) const
  {
    if(low < 0 || high >= nsz_) {
      std::cerr << "ERROR: prod(" << low << "," << high
          << ") is invalid because indices must be in the range [0,"
          << nsz_ << ").  Returning " << defaultReturnVal << std::endl;
        return defaultReturnVal;
    }
    if(low > high) {
      return defaultReturnVal;
    }
    size_t result = 1;
    for(int j = low; j <= high; j++)
      result *= sz_[j];
    return result;
  }

  friend bool operator==(const SizeArray& sz1, const SizeArray& sz2){
    if(sz1.size() != sz2.size()){
      return false;
    }

    for(int i=0; i<sz1.size(); i++) {
      if(sz1[i] != sz2[i] ){
        return false;
      }
    }
    return true;
  }

  friend bool operator!=(const SizeArray& sz1, const SizeArray& sz2){
    return !(sz1 == sz2);
  }

  friend std::ostream& operator<<(std::ostream& os, const SizeArray& sz){
    for(int i=0; i<sz.size(); i++) {
      os << sz[i] << " ";
    }
    return os;
  }

private:
  int nsz_;
  std::vector<int> sz_;
};

} // end of namespace

#endif /* SIZEARRAY_HPP_ */
