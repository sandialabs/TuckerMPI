#include<cstdlib>
#include "TuckerMPI.hpp"
#include <cmath>
#include <map>
#include <iostream>
#include <iomanip>

template <class scalar_t>
bool checkEqual(const scalar_t* arr1, const scalar_t* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++) {
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind]))/std::abs(arr1[r+c*nrows]) > 100 * std::numeric_limits<scalar_t>::epsilon()) {
          std::cout << "epsilon: " << std::numeric_limits<scalar_t>::epsilon() << std::endl;
          std::cout << std::setprecision(9) << "mismatch :" << "arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

//three scenarios to test here:
//1. Each mode n processor fiber only has one processor in it. In this case NO redistribution is done
//and sequential LQ is called.
//2. Each mode n processor fiber has multiple processors AND n=N-1. In this case redistribution is done
//with each local unfolding being row major. Sequential LQ is not called.
//3. Each mode n processor fiber has multiple processors AND n!=N-1. In this case redistribution is done
//with each local unfolding being column major. parallel LQ is called.
int main(int argc, char* argv[])
{
// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t;
  std::string filename = "input_files/lq_data_single.mpi"; 
#else
  typedef double scalar_t;
  std::string filename = "input_files/lq_data.mpi";
#endif

  MPI_Init(&argc,&argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  scalar_t trueL0[144] = {-23849.5889482398,-17774.7242906376,-18094.6470371705,-17835.1318307058,-18293.7929012903,-18275.2636511234,-17906.6024545266,-18074.4790166212,-17959.1322487599,-17655.9258071020,-18290.2463412140,-18147.3568345061
                        ,0,15927.6956397282,6616.91547614796,6942.59317375532,6957.98058026128,7078.56703271951,6647.44937269167,6673.36984500899,6510.94710276394,7089.88941257394,6851.39159772014,6854.43275237639
                        ,0,0,14432.1068864436,4289.62076937678,4652.22642986779,5030.85707064290,4405.17700011329,4269.98075145095,4726.29303211693,4592.36949596314,4492.54413419240,4332.39004911653
                        ,0,0,0,13658.8874825149,2538.84486340496,2968.76970792052,2838.86599424121,2803.76714195008,2752.03029260961,3365.50742416397,3036.61298762652,2915.45297201958
                        ,0,0,0,0,-13263.3598885882,-2780.24338910834,-2345.29316353036,-2386.60983620948,-1931.57882600808,-3137.02699191715,-3172.08544024535,-2584.72237641521
                        ,0,0,0,0,0,12981.4661921772,1768.95218814928,2118.07043918983,1751.26493498321,2096.00559814417,1505.80614231973,1869.39505375413
                        ,0,0,0,0,0,0,12881.5703223868,1670.05541134559,1596.42904682517,1451.92041830562,1670.21611868951,1763.68927123662
                        ,0,0,0,0,0,0,0,-12744.6628599104,-1969.02408215288,-2300.90458921915,-1120.95929225171,-1867.84960020013
                        ,0,0,0,0,0,0,0,0,-12562.2257569451,-1146.42755000450,-1319.03145525071,-1337.21049548118
                        ,0,0,0,0,0,0,0,0,0,-12571.7699799737,-863.144860419557,-1322.33797945032
                        ,0,0,0,0,0,0,0,0,0,0,-12649.9259243531,-1526.02110230107
                        ,0,0,0,0,0,0,0,0,0,0,0,-12298.8349994680};
  scalar_t trueL1[144] = {-23984.8004786365,-17922.1545070963,-17615.4699046310,-17666.4769163879,-18264.9258804635,-18035.6846572606,-18509.4680022637,-18246.1674588372,-17758.0650870694,-18260.6927412264,-18167.1776835549,-18151.7261895835
                        ,0,15550.0120522065,7046.00512760060,6717.93126318658,6910.59125542114,6421.13173906777,6766.65196567129,6817.74935507676,6446.24129074201,6684.78654139556,6821.68754862194,6594.60509197659
                        ,0,0,-14453.4175121618,-3890.11639523449,-4035.06522750462,-4115.21188688365,-4335.51095828080,-4661.57130538406,-4209.21340262534,-4312.92953927133,-4166.11848139783,-3910.34184476888
                        ,0,0,0,-13755.8196534181,-3172.85674090367,-3602.73656488275,-3829.51638526944,-3324.29840084975,-3544.77881525183,-3293.45963708055,-3558.10741861874,-3159.99429603333
                        ,0,0,0,0,-13307.8894567530,-2858.06779694487,-2496.92835732930,-2326.91035778430,-2990.01061320349,-2434.27784853010,-2691.53384707340,-2342.93249547014
                        ,0,0,0,0,0,12954.6181329890,2479.95891782316,1896.01591301613,2251.38462517858,2109.58256671386,2222.30352532367,2191.56195523026
                        ,0,0,0,0,0,0,12866.0600021338,1941.11751970445,1958.47735265736,1681.46632302660,2047.30476054093,1706.14231095169
                        ,0,0,0,0,0,0,0,-12699.9402959805,-2092.99856919562,-1628.16078929978,-1526.20093099192,-1186.55505674856
                        ,0,0,0,0,0,0,0,0,12614.8609789388,1050.20021383114,1789.53241397171,1548.68144052383
                        ,0,0,0,0,0,0,0,0,0,12359.2083737049,1318.47973319212,1303.90074784872
                        ,0,0,0,0,0,0,0,0,0,0,-12283.4150324433,-1416.73874557502
                        ,0,0,0,0,0,0,0,0,0,0,0,-12685.1657741559};
  scalar_t trueL2[144] = {-23919.1890330755,-18172.9312979182,-17489.3878894276,-18240.1106240140,-18371.0018091487,-18056.8329638083,-17883.5587363975,-18534.7360392091,-18323.3179600674,-18285.4615762766,-18157.5008834719,-18054.0321163420
                        ,0,-15594.3435270981,-6628.76774922671,-6938.84424024239,-6925.39715190606,-6720.73139267506,-6705.70271454213,-6599.49962982223,-6266.87147118211,-6786.61694477214,-6680.07176587838,-6456.29599760679
                        ,0,0,-14230.3319139068,-3613.05270635871,-4330.36485193008,-4303.87858167391,-4127.00876416880,-4143.52848771550,-3663.10523139682,-3767.10273166744,-4334.12903053142,-4024.15692108709
                        ,0,0,0,13625.1593082232,3020.31515445465,2374.44517288196,2489.77444501950,3071.83185897708,3218.02484179389,3499.14972304482,2641.37000494223,3580.32801922099
                        ,0,0,0,0,-13356.6148193907,-2835.14117497171,-1997.65732531571,-2281.77661192980,-2518.27756380562,-2676.38169738660,-2980.63462220890,-3070.06246005802
                        ,0,0,0,0,0,-13035.5264434469,-2245.21690062202,-1914.04247592281,-1886.83582653882,-2318.77311210374,-2166.32909299685,-2150.26066091675
                        ,0,0,0,0,0,0,12907.2000131392,1579.98230846612,2042.76955769439,1214.06072281142,2301.97383855289,2039.53275434252
                        ,0,0,0,0,0,0,0,-12846.9484350577,-1398.15643742080,-1247.46958063450,-1905.53672110299,-1517.89580138261
                        ,0,0,0,0,0,0,0,0,-12726.2209970299,-1056.91240528160,-1190.08467712235,-1545.78155045950
                        ,0,0,0,0,0,0,0,0,0,-12758.8266016503,-887.499609476112,-1450.84600215865
                        ,0,0,0,0,0,0,0,0,0,0,12367.8576794228,2010.55610807143
                        ,0,0,0,0,0,0,0,0,0,0,0,-12490.4461152022};
  scalar_t trueL3[144] = {-24007.8582968161,-17882.9845916301,-17793.0873599271,-18311.1346528691,-17469.4133401988,-17728.1502472232,-17716.3810174774,-17983.9237078994,-18619.1042313542,-18545.0332343492,-17655.4634636532,-18033.3047057936
                        ,0,-16095.8485049879,-6880.70889028153,-7097.27257386136,-7281.96331967756,-7366.78363082625,-6252.15043579831,-7008.75889827275,-7157.92316789236,-6754.22077843093,-6280.82259161879,-7207.08906859890
                        ,0,0,-14307.2796634861,-4136.15162059204,-4049.91612113628,-4587.45385870540,-4259.13871822973,-4127.00353377372,-4697.92622018819,-4347.41908136038,-4675.37449047705,-4520.86949270615
                        ,0,0,0,13411.3148687325,3090.93319681329,3578.46452893151,3477.85379409751,2912.14180862004,2936.29914404115,3022.91000795858,3134.32783237319,3296.52837667573
                        ,0,0,0,0,13378.7124247722,2240.98009547508,2732.51001344516,1541.50627251131,2737.87494688981,2509.67219238583,2198.99029958295,2542.64279339756
                        ,0,0,0,0,0,12898.8917612941,1767.56453822725,1889.77419716685,1827.55166647415,1833.07269667899,1780.16524147501,1996.07931194956
                        ,0,0,0,0,0,0,-12846.8992588521,-1636.51063319894,-1645.12522048597,-1745.46563636387,-1790.08036465192,-1516.56757438216
                        ,0,0,0,0,0,0,0,12927.8030436730,1667.10463559258,1916.80679030354,1597.84321800364,1530.70725366705
                        ,0,0,0,0,0,0,0,0,12734.8345075763,1371.91229029643,1775.40013016993,1615.28142667633
                        ,0,0,0,0,0,0,0,0,0,12715.0348608256,1013.99059327380,1245.05799120537
                        ,0,0,0,0,0,0,0,0,0,0,12548.1593486025,1220.25087450754
                        ,0,0,0,0,0,0,0,0,0,0,0,12273.8705245290};

  int ndims = 4;
  Tucker::SizeArray* sz =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*sz)[0] = 12; (*sz)[1] = 12; (*sz)[2] = 12; (*sz)[3] = 12;
  std::map<int, scalar_t*> pointerMap;
  std::map<int, int> sizeMap;
  long d = (sizeof(scalar_t)*ndims);
  scalar_t processorGridLayouts_12[160] = {1,1,1,12
                                        ,1,1,12,1
                                        ,1,12,1,1
                                        ,12,1,1,1
                                        ,1,1,2,6
                                        ,1,1,6,2
                                        ,1,2,1,6
                                        ,1,2,6,1
                                        ,1,6,1,2
                                        ,1,6,2,1
                                        ,2,1,1,6
                                        ,2,1,6,1
                                        ,2,6,1,1
                                        ,6,1,1,2
                                        ,6,1,2,1
                                        ,6,2,1,1
                                        ,1,1,3,4
                                        ,1,1,4,3
                                        ,1,3,1,4
                                        ,1,3,4,1
                                        ,1,4,1,3
                                        ,1,4,3,1
                                        ,3,1,1,4
                                        ,3,1,4,1
                                        ,3,4,1,1
                                        ,4,1,1,3
                                        ,4,1,3,1
                                        ,4,3,1,1
                                        ,1,2,2,3
                                        ,1,2,3,2
                                        ,1,3,2,2
                                        ,2,1,2,3
                                        ,2,1,3,2
                                        ,2,2,1,3
                                        ,2,2,3,1
                                        ,2,3,1,2
                                        ,2,3,2,1
                                        ,3,1,2,2
                                        ,3,2,1,2
                                        ,3,2,2,1};
  scalar_t* pGridLayouts_12 = processorGridLayouts_12;
  pointerMap[12] = pGridLayouts_12; sizeMap[12] = sizeof(processorGridLayouts_12)/d;
  scalar_t processorGridLayouts_11[16] ={1,1,1,11
                                      ,1,1,11,1
                                      ,1,11,1,1
                                      ,11,1,1,1};
  scalar_t* pGridLayouts_11 = processorGridLayouts_11;
  pointerMap[11] = pGridLayouts_11; sizeMap[11] = sizeof(processorGridLayouts_11)/d;
  scalar_t processorGridLayouts_10[64] = {1,1,2,5
                                        ,1,1,5,2
                                        ,1,2,1,5
                                        ,1,2,5,1
                                        ,1,5,1,2
                                        ,1,5,2,1
                                        ,2,1,1,5
                                        ,2,1,5,1
                                        ,2,5,1,1
                                        ,5,1,1,2
                                        ,5,1,2,1
                                        ,5,2,1,1
                                        ,1,1,1,10
                                        ,1,1,10,1
                                        ,1,10,1,1
                                        ,10,1,1,1};
  scalar_t* pGridLayouts_10 = processorGridLayouts_10;
  pointerMap[10] = pGridLayouts_10; sizeMap[10] = sizeof(processorGridLayouts_10)/d;
  scalar_t processorGridLayouts_9[40] = {1,1,3,3
                                      ,1,3,1,3
                                      ,1,3,3,1
                                      ,3,1,1,3
                                      ,3,1,3,1
                                      ,3,3,1,1
                                      ,1,1,1,9
                                      ,1,1,9,1
                                      ,1,9,1,1
                                      ,9,1,1,1};
  scalar_t* pGridLayouts_9 = processorGridLayouts_9;
  pointerMap[9] = pGridLayouts_9; sizeMap[9] = sizeof(processorGridLayouts_9)/d;
  scalar_t processorGridLayouts_8[80] = {1,1,1,8
                                      ,1,1,8,1
                                      ,1,8,1,1
                                      ,8,1,1,1
                                      ,1,2,2,2
                                      ,2,1,2,2
                                      ,2,2,1,2
                                      ,2,2,2,1
                                      ,1,1,2,4
                                      ,1,1,4,2
                                      ,1,2,1,4
                                      ,1,2,4,1
                                      ,1,4,1,2
                                      ,1,4,2,1
                                      ,2,1,1,4
                                      ,2,1,4,1
                                      ,2,4,1,1
                                      ,4,1,1,2
                                      ,4,1,2,1
                                      ,4,2,1,1};                                      
  scalar_t* pGridLayouts_8 = processorGridLayouts_8;
  pointerMap[8] = pGridLayouts_8; sizeMap[8] = sizeof(processorGridLayouts_8)/d;
  scalar_t processorGridLayouts_7[16] = {1,1,1,7
                                      ,1,1,7,1
                                      ,1,7,1,1
                                      ,7,1,1,1};
  scalar_t* pGridLayouts_7 = processorGridLayouts_7;
  pointerMap[7] = pGridLayouts_7; sizeMap[7] = sizeof(processorGridLayouts_7)/d;
  scalar_t processorGridLayouts_6[64] = {1,1,1,6
                                      ,1,1,6,1
                                      ,1,6,1,1
                                      ,6,1,1,1
                                      ,1,1,2,3
                                      ,1,1,3,2
                                      ,1,2,1,3
                                      ,1,2,3,1
                                      ,1,3,1,2
                                      ,1,3,2,1
                                      ,2,1,1,3
                                      ,2,1,3,1
                                      ,2,3,1,1
                                      ,3,1,1,2
                                      ,3,1,2,1
                                      ,3,2,1,1};
  scalar_t* pGridLayouts_6 = processorGridLayouts_6;
  pointerMap[6] = pGridLayouts_6; sizeMap[6] = sizeof(processorGridLayouts_6)/d;
  scalar_t processorGridLayouts_5[16] = {1,1,1,5
                                      ,1,1,5,1
                                      ,1,5,1,1
                                      ,5,1,1,1};
  scalar_t* pGridLayouts_5 = processorGridLayouts_5;
  pointerMap[5] = pGridLayouts_5; sizeMap[5] = sizeof(processorGridLayouts_5)/d;
  scalar_t processorGridLayouts_4[40] = {1,1,1,4
                                      ,1,1,4,1
                                      ,1,4,1,1
                                      ,4,1,1,1
                                      ,1,1,2,2
                                      ,1,2,1,2
                                      ,1,2,2,1
                                      ,2,1,1,2
                                      ,2,1,2,1
                                      ,2,2,1,1};                                    
  scalar_t* pGridLayouts_4 = processorGridLayouts_4;
  pointerMap[4] = pGridLayouts_4; sizeMap[4] = sizeof(processorGridLayouts_4)/d;
  scalar_t processorGridLayouts_3[16] = {1,1,1,3
                                      ,1,1,3,1
                                      ,1,3,1,1
                                      ,3,1,1,1};
  scalar_t* pGridLayouts_3 = processorGridLayouts_3;
  pointerMap[3] = pGridLayouts_3; sizeMap[3] = sizeof(processorGridLayouts_3)/d;
  scalar_t processorGridLayouts_2[16] = {1,1,1,2
                                      ,1,1,2,1
                                      ,1,2,1,1
                                      ,2,1,1,1};
  scalar_t* pGridLayouts_2 = processorGridLayouts_2;
  pointerMap[2] = pGridLayouts_2; sizeMap[2] = sizeof(processorGridLayouts_2)/d;
  scalar_t processorGridLayouts_1[4] = {1,1,1,1};
  scalar_t* pGridLayouts_1 = processorGridLayouts_1;
  pointerMap[1] = pGridLayouts_1; sizeMap[1] = sizeof(processorGridLayouts_1)/d;

  int nPossibleProcGrid = sizeMap[np];
  scalar_t* processorGridLayouts = pointerMap[np];

  int root = 0;
  int LSize = 12; // length of the side of the square that L is in.
  int compareResultBuff;
  for(int t=0; t<nPossibleProcGrid; t++){
    Tucker::SizeArray* nprocsPerDim = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
    (*nprocsPerDim)[0] = *(processorGridLayouts+t*4);
    (*nprocsPerDim)[1] = *(processorGridLayouts+t*4+1); 
    (*nprocsPerDim)[2] = *(processorGridLayouts+t*4+2); 
    (*nprocsPerDim)[3] = *(processorGridLayouts+t*4+3);
    TuckerMPI::Distribution* dist =
      Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*sz,*nprocsPerDim);
    Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(nprocsPerDim);
    TuckerMPI::Tensor<scalar_t>* tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);
    TuckerMPI::importTensorBinary(filename.c_str(),tensor);

    Tucker::Matrix<scalar_t>* L0 = TuckerMPI::LQ<scalar_t>(tensor, 0, false);
    compareResultBuff = (int)checkEqual(L0->data(), trueL0, LSize, LSize);
    if(compareResultBuff != 1){
      Tucker::MemoryManager::safe_delete(tensor);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    Tucker::MemoryManager::safe_delete(L0);
  
    Tucker::Matrix<scalar_t>* L1 = TuckerMPI::LQ<scalar_t>(tensor, 1, false);
    compareResultBuff = checkEqual(L1->data(), trueL1, LSize, LSize);
    if(compareResultBuff != 1){
      Tucker::MemoryManager::safe_delete(tensor);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    Tucker::MemoryManager::safe_delete(L1);
    
    Tucker::Matrix<scalar_t>* L2 = TuckerMPI::LQ<scalar_t>(tensor, 2, false);
    compareResultBuff = checkEqual(L2->data(), trueL2, LSize, LSize);
    if(compareResultBuff != 1){
      Tucker::MemoryManager::safe_delete(tensor);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    Tucker::MemoryManager::safe_delete(L2);
    MPI_Barrier(MPI_COMM_WORLD);

    Tucker::Matrix<scalar_t>* L3 = TuckerMPI::LQ<scalar_t>(tensor, 3, false);
    compareResultBuff = checkEqual(L3->data(), trueL3, LSize, LSize);
    if(compareResultBuff != 1){
      Tucker::MemoryManager::safe_delete(tensor);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    Tucker::MemoryManager::safe_delete(L3);

    Tucker::MemoryManager::safe_delete(tensor);
  }
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(sz);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
