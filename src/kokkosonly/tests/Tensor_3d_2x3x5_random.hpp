#include <gtest/gtest.h>
#include "Tucker_Tensor.hpp"
#include "Tucker_SizeArray.hpp"
#include <Kokkos_Core.hpp>

using scalar_t = double;
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using matrix = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space>;

class Tensor_3d_2x3x5_random : public ::testing::Test {    
    protected:

    void SetUp() override {
        std::array<int, 3> dims = {2,3,5};
        TuckerKokkos::SizeArray size(3);
        size[0] = dims.at(0);
        size[1] = dims.at(1);
        size[2] = dims.at(2);
        X = TuckerKokkos::Tensor<scalar_t, memory_space>(size);
        auto view1d_d = X.data();
        auto view1d_h = Kokkos::create_mirror(view1d_d);
        view1d_h(0) = 2;    view1d_h(1) = 3;    view1d_h(2) = 5;    view1d_h(3) = 7;
        view1d_h(4) = 11;   view1d_h(5) = 13;   view1d_h(6) = 17;   view1d_h(7) = 19;
        view1d_h(8) = 23;   view1d_h(9) = 29;   view1d_h(10) = 31;  view1d_h(11) = 37;
        view1d_h(12) = 41;  view1d_h(13) = 43;  view1d_h(14) = 47;  view1d_h(15) = 53;
        view1d_h(16) = 59;  view1d_h(17) = 61;  view1d_h(18) = 67;  view1d_h(19) = 71;
        view1d_h(20) = 73;  view1d_h(21) = 79;  view1d_h(22) = 83;  view1d_h(23) = 97;
        view1d_h(24) = 101; view1d_h(25) = 103; view1d_h(26) = 107; view1d_h(27) = 109;
        view1d_h(28) = 113; view1d_h(29) = 127;
        Kokkos::deep_copy(view1d_d, view1d_h);
    }
    
    TuckerKokkos::Tensor<scalar_t, memory_space> X;
};