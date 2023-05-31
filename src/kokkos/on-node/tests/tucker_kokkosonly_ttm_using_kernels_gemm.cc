#include <gtest/gtest.h>

#include "TuckerOnNode_Tensor.hpp"
#include "TuckerOnNode_ttm.hpp"

#include "Tensor_3d_2x3x5_random.hpp"

TEST_F(Tensor_3d_2x3x5_random, ttm_mat7x2) {

    // Matrix 7x2
    matrix mat7x2("mat7x2", 7, 2);
    auto view2d_h = Kokkos::create_mirror(mat7x2);
    // LayoutLeft, Column-major
    view2d_h(0, 0) = 131;   view2d_h(0, 1) = 167;
    view2d_h(1, 0) = 137;   view2d_h(1, 1) = 173;
    view2d_h(2, 0) = 139;   view2d_h(2, 1) = 179;
    view2d_h(3, 0) = 149;   view2d_h(3, 1) = 181;
    view2d_h(4, 0) = 151;   view2d_h(4, 1) = 191;
    view2d_h(5, 0) = 157;   view2d_h(5, 1) = 193;
    view2d_h(6, 0) = 163;   view2d_h(6, 1) = 197;
    Kokkos::deep_copy(mat7x2, view2d_h);

    // TTM
    TuckerOnNode::Tensor<scalar_t, memory_space> result = TuckerOnNode::ttm(X, 0, mat7x2, false);
    scalar_t* data = result.data().data();

    // Few true data
    std::array<scalar_t, 105> trueData;
    trueData[0] = 763; trueData[1] = 793; trueData[2] = 815; trueData[3] = 841;
    trueData[4] = 875; trueData[5] = 893; trueData[6] = 917; trueData[7] = 1824;
    trueData[8] = 1896; trueData[9] = 1948; trueData[10] = 2012; trueData[11] = 2092;
    trueData[12] = 2136; trueData[13] = 2194; trueData[14] = 3612; trueData[15] = 3756;
    trueData[16] = 3856; trueData[17] = 3992; trueData[18] = 4144; trueData[19] = 4236;
    trueData[20] = 4354; trueData[21] = 5400; trueData[22] = 5616; trueData[23] = 5764;
    trueData[24] = 5972; trueData[25] = 6196; trueData[26] = 6336; trueData[27] = 6514;
    trueData[28] = 7856; trueData[29] = 8168; trueData[30] = 8388; trueData[31] = 8676;
    trueData[32] = 9012; trueData[33] = 9208; trueData[34] = 9462; trueData[35] = 10240;
    trueData[36] = 10648; trueData[37] = 10932; trueData[38] = 11316; trueData[39] = 11748;
    trueData[40] = 12008; trueData[41] = 12342; trueData[42] = 12552; trueData[43] = 13056;
    trueData[44] = 13396; trueData[45] = 13892; trueData[46] = 14404; trueData[47] = 14736;
    trueData[48] = 15154; trueData[49] = 15008; trueData[50] = 15608; trueData[51] = 16020;
    trueData[52] = 16596; trueData[53] = 17220; trueData[54] = 17608; trueData[55] = 18102;
    trueData[56] = 17916; trueData[57] = 18636; trueData[58] = 19120; trueData[59] = 19832;
    trueData[60] = 20560; trueData[61] = 21036; trueData[62] = 21634; trueData[63] = 20634;
    trueData[64] = 21462; trueData[65] = 22022; trueData[66] = 22834; trueData[67] = 23678;
    trueData[68] = 24222; trueData[69] = 24908; trueData[70] = 22756; trueData[71] = 23668;
    trueData[72] = 24288; trueData[73] = 25176; trueData[74] = 26112; trueData[75] = 26708;
    trueData[76] = 27462; trueData[77] = 27072; trueData[78] = 28152; trueData[79] = 28900;
    trueData[80] = 29924; trueData[81] = 31060; trueData[82] = 31752; trueData[83] = 32638;
    trueData[84] = 30432; trueData[85] = 31656; trueData[86] = 32476; trueData[87] = 33692;
    trueData[88] = 34924; trueData[89] = 35736; trueData[90] = 36754; trueData[91] = 32220;
    trueData[92] = 33516; trueData[93] = 34384; trueData[94] = 35672; trueData[95] = 36976;
    trueData[96] = 37836; trueData[97] = 38914; trueData[98] = 36012; trueData[99] = 37452;
    trueData[100] = 38440; trueData[101] = 39824; trueData[102] = 41320; trueData[103] = 42252;
    trueData[104] = 43438;
    // Check
    for(int i=0; i<105; i++) {
        ASSERT_EQ(data[i], trueData[i]);
    }
}

TEST_F(Tensor_3d_2x3x5_random, ttm_mat2x7) {

    // Matrix 2x7
    matrix mat2x7("mat2x7", 2, 7);
    auto view2d_h = Kokkos::create_mirror(mat2x7);
    view2d_h(0,0)=131;  view2d_h(0,1)=139;  view2d_h(0,2)=151;  view2d_h(0,3)=163;  view2d_h(0,4)=173;  view2d_h(0,5)=181;  view2d_h(0,6)=193;
    view2d_h(1,0)=137;  view2d_h(1,1)=149;  view2d_h(1,2)=157;  view2d_h(1,3)=167;  view2d_h(1,4)=179;  view2d_h(1,5)=191;  view2d_h(1,6)=197;
    Kokkos::deep_copy(mat2x7, view2d_h);

    // TTM
    TuckerOnNode::Tensor<scalar_t, memory_space> result =
        TuckerOnNode::ttm(X, 0, mat2x7, true);
    scalar_t* data = result.data().data();

    // Few true data
    std::array<scalar_t, 105> trueData;
    trueData[0] = 673; trueData[1] = 725; trueData[2] = 773;
    trueData[3] = 827; trueData[4] = 883; trueData[5] = 935;
    trueData[6] = 977; trueData[7] = 1614; trueData[8] = 1738;
    trueData[9] = 1854; trueData[10] = 1984; trueData[11] = 2118;
    trueData[12] = 2242; trueData[13] = 2344; trueData[14] = 3222;
    trueData[15] = 3466; trueData[16] = 3702; trueData[17] = 3964;
    trueData[18] = 4230; trueData[19] = 4474; trueData[20] = 4684;
    trueData[21] = 4830; trueData[22] = 5194; trueData[23] = 5550;
    trueData[24] = 5944; trueData[25] = 6342; trueData[26] = 6706;
    trueData[27] = 7024; trueData[28] = 6986; trueData[29] = 7518;
    trueData[30] = 8026; trueData[31] = 8592; trueData[32] = 9170;
    trueData[33] = 9702; trueData[34] = 10152; trueData[35] = 9130;
    trueData[36] = 9822; trueData[37] = 10490; trueData[38] = 11232;
    trueData[39] = 11986; trueData[40] = 12678; trueData[41] = 13272;
    trueData[42] = 11262; trueData[43] = 12106; trueData[44] = 12942;
    trueData[45] = 13864; trueData[46] = 14790; trueData[47] = 15634;
    trueData[48] = 16384; trueData[49] = 13418; trueData[50] = 14430;
    trueData[51] = 15418; trueData[52] = 16512; trueData[53] = 17618;
    trueData[54] = 18630; trueData[55] = 19512; trueData[56] = 16086;
    trueData[57] = 17290; trueData[58] = 18486; trueData[59] = 19804;
    trueData[60] = 21126; trueData[61] = 22330; trueData[62] = 23404;
    trueData[63] = 18504; trueData[64] = 19892; trueData[65] = 21264;
    trueData[66] = 22778; trueData[67] = 24300; trueData[68] = 25688;
    trueData[69] = 26918; trueData[70] = 20386; trueData[71] = 21918;
    trueData[72] = 23426; trueData[73] = 25092; trueData[74] = 26770;
    trueData[75] = 28302; trueData[76] = 29652; trueData[77] = 24162;
    trueData[78] = 25990; trueData[79] = 27762; trueData[80] = 29728;
    trueData[81] = 31722; trueData[82] = 33550; trueData[83] = 35128;
    trueData[84] = 27342; trueData[85] = 29386; trueData[86] = 31422;
    trueData[87] = 33664; trueData[88] = 35910; trueData[89] = 37954;
    trueData[90] = 39784; trueData[91] = 28950; trueData[92] = 31114;
    trueData[93] = 33270; trueData[94] = 35644; trueData[95] = 38022;
    trueData[96] = 40186; trueData[97] = 42124; trueData[98] = 32202;
    trueData[99] = 34630; trueData[100] = 37002; trueData[101] = 39628;
    trueData[102] = 42282; trueData[103] = 44710; trueData[104] = 46828;

    // Check
    for(int i=0; i<105; i++) {
        ASSERT_EQ(data[i], trueData[i]);
    }
}

TEST_F(Tensor_3d_2x3x5_random, ttm_mat7x3) {

    // Matrix 7x3
    matrix mat7x3("mat7x3", 7, 3);
    auto view2d_h = Kokkos::create_mirror(mat7x3);
    view2d_h(0, 0) = 131;  view2d_h(0, 1) = 167;  view2d_h(0, 2) = 199;
    view2d_h(1, 0) = 137;  view2d_h(1, 1) = 173;  view2d_h(1, 2) = 211;
    view2d_h(2, 0) = 139;  view2d_h(2, 1) = 179;  view2d_h(2, 2) = 223;
    view2d_h(3, 0) = 149;  view2d_h(3, 1) = 181;  view2d_h(3, 2) = 227;
    view2d_h(4, 0) = 151;  view2d_h(4, 1) = 191;  view2d_h(4, 2) = 229;
    view2d_h(5, 0) = 157;  view2d_h(5, 1) = 193;  view2d_h(5, 2) = 233;
    view2d_h(6, 0) = 163;  view2d_h(6, 1) = 197;  view2d_h(6, 2) = 239;
    Kokkos::deep_copy(mat7x3, view2d_h);

    // TTM
    TuckerOnNode::Tensor<scalar_t, memory_space> result =
        TuckerOnNode::ttm(X, 1, mat7x3, false);
    scalar_t* data = result.data().data();

    // Few true data
    std::array<scalar_t, 105> trueData;
    trueData[0] = 3286; trueData[1] = 4149; trueData[2] = 3460;
    trueData[3] = 4365; trueData[4] = 3626; trueData[5] = 4569;
    trueData[6] = 3700; trueData[7] = 4665; trueData[8] = 3776;
    trueData[9] = 4767; trueData[10] = 3842; trueData[11] = 4851;
    trueData[12] = 3940; trueData[13] = 4975; trueData[14] = 12237;
    trueData[15] = 14695; trueData[16] = 12849; trueData[17] = 15427;
    trueData[18] = 13393; trueData[19] = 16083; trueData[20] = 13733;
    trueData[21] = 16479; trueData[22] = 14059; trueData[23] = 16881;
    trueData[24] = 14331; trueData[25] = 17201; trueData[26] = 14711;
    trueData[27] = 17653; trueData[28] = 24961; trueData[29] = 26623;
    trueData[30] = 26197; trueData[31] = 27931; trueData[32] = 27269;
    trueData[33] = 29067; trueData[34] = 28009; trueData[35] = 29847;
    trueData[36] = 28679; trueData[37] = 30585; trueData[38] = 29255;
    trueData[39] = 31193; trueData[40] = 30043; trueData[41] = 32029;
    trueData[42] = 37485; trueData[43] = 41797; trueData[44] = 39321;
    trueData[45] = 43861; trueData[46] = 40889; trueData[47] = 45641;
    trueData[48] = 42037; trueData[49] = 46897; trueData[50] = 43067;
    trueData[51] = 48023; trueData[52] = 43947; trueData[53] = 48995;
    trueData[54] = 45139; trueData[55] = 50319; trueData[56] = 53587;
    trueData[57] = 56969; trueData[58] = 56191; trueData[59] = 59765;
    trueData[60] = 58391; trueData[61] = 62149; trueData[62] = 60067;
    trueData[63] = 63905; trueData[64] = 61565; trueData[65] = 65455;
    trueData[66] = 62837; trueData[67] = 66799; trueData[68] = 64549;
    trueData[69] = 68615;

    // Check
    for(int i=0; i<70; i++) {
        ASSERT_EQ(data[i], trueData[i]);
    }
}
