#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

// Imported from TuckerMPI_slice_test.cpp
TEST(tuckerkokkosmpi, compute_slice_metrics_allmode)
{
  using scalar_t = double;

  scalar_t trueData[11][4][5];
  trueData[0][0][0] = -0.493284685681523;
  trueData[0][0][1] = 0.497003271606648;
  trueData[0][0][2] = -10.553150952476088;
  trueData[0][0][3] = -0.027410781694743;
  trueData[0][0][4] = 0.279860988681208;
  trueData[1][0][0] = -0.498848942870893;
  trueData[1][0][1] = 0.499080394761361;
  trueData[1][0][2] = -1.110888518497833;
  trueData[1][0][3] = -0.002885424723371;
  trueData[1][0][4] = 0.280312118084699;
  trueData[2][0][0] = -0.499477624643055;
  trueData[2][0][1] = 0.499491620097705;
  trueData[2][0][2] = 1.862939627376536;
  trueData[2][0][3] = 0.004838804226952;
  trueData[2][0][4] = 0.293629663290977;
  trueData[0][1][0] = -0.498848942870893;
  trueData[0][1][1] = 0.497560349512189;
  trueData[0][1][2] = -6.082733668613957;
  trueData[0][1][3] = -0.026332180383610;
  trueData[0][1][4] = 0.291721762321716;
  trueData[1][1][0] = -0.483017061662739;
  trueData[1][1][1] = 0.493704624120852;
  trueData[1][1][2] = 1.141409922911623;
  trueData[1][1][3] = 0.004941168497453;
  trueData[1][1][4] = 0.287170306225749;
  trueData[2][1][0] = -0.499477624643055;
  trueData[2][1][1] = 0.499491620097705;
  trueData[2][1][2] = -4.454191596296774;
  trueData[2][1][3] = -0.019282214702583;
  trueData[2][1][4] = 0.282838892110639;
  trueData[3][1][0] = -0.486716799532747;
  trueData[3][1][1] = 0.496156111296869;
  trueData[3][1][2] = 4.968202870959377;
  trueData[3][1][3] = 0.021507371735755;
  trueData[3][1][4] = 0.293074683975052;
  trueData[4][1][0] = -0.493284685681523;
  trueData[4][1][1] = 0.499080394761361;
  trueData[4][1][2] = -5.373787372557653;
  trueData[4][1][3] = -0.023263148798951;
  trueData[4][1][4] = 0.266355438174839;
  trueData[0][2][0] = -0.486716799532747;
  trueData[0][2][1] = 0.496134716626885;
  trueData[0][2][2] = -1.779496142152801;
  trueData[0][2][3] = -0.010784825103956;
  trueData[0][2][4] = 0.301403543856597;
  trueData[1][2][0] = -0.495365775865933;
  trueData[1][2][1] = 0.495389727655092;
  trueData[1][2][2] = -5.690788931387834;
  trueData[1][2][3] = -0.034489629887199;
  trueData[1][2][4] = 0.284317210268765;
  trueData[2][2][0] = -0.498848942870893;
  trueData[2][2][1] = 0.499080394761361;
  trueData[2][2][2] = -5.820784195477449;
  trueData[2][2][3] = -0.035277479972591;
  trueData[2][2][4] = 0.279638687405347;
  trueData[3][2][0] = -0.490197747736938;
  trueData[3][2][1] = 0.497003271606648;
  trueData[3][2][2] = 4.030595375660884;
  trueData[3][2][3] = 0.024427850761581;
  trueData[3][2][4] = 0.282034988253036;
  trueData[4][2][0] = -0.484596562348445;
  trueData[4][2][1] = 0.487934734952495;
  trueData[4][2][2] = -4.107269380982341;
  trueData[4][2][3] = -0.024892541702923;
  trueData[4][2][4] = 0.288884641357105;
  trueData[5][2][0] = -0.499477624643055;
  trueData[5][2][1] = 0.499491620097705;
  trueData[5][2][2] = 3.088416977971128;
  trueData[5][2][3] = 0.018717678654370;
  trueData[5][2][4] = 0.270159821340926;
  trueData[6][2][0] = -0.488097930498759;
  trueData[6][2][1] = 0.497560349512189;
  trueData[6][2][2] = 0.478226452771030;
  trueData[6][2][3] = 0.002898342138006;
  trueData[6][2][4] = 0.281262397035613;
  trueData[0][3][0] = -0.488097930498759;
  trueData[0][3][1] = 0.470592781760616;
  trueData[0][3][2] = 2.261310449772215;
  trueData[0][3][3] = 0.021536289997831;
  trueData[0][3][4] = 0.294591170268916;
  trueData[1][3][0] = -0.495365775865933;
  trueData[1][3][1] = 0.496134716626885;
  trueData[1][3][2] = -1.423818336371447;
  trueData[1][3][3] = -0.013560174632109;
  trueData[1][3][4] = 0.285521396645195;
  trueData[2][3][0] = -0.471325847535894;
  trueData[2][3][1] = 0.487982003161633;
  trueData[2][3][2] = -0.765891404236863;
  trueData[2][3][3] = -0.007294203849875;
  trueData[2][3][4] = 0.284221121591849;
  trueData[3][3][0] = -0.484512874363981;
  trueData[3][3][1] = 0.499080394761361;
  trueData[3][3][2] = -2.535681063213515;
  trueData[3][3][3] = -0.024149343459176;
  trueData[3][3][4] = 0.278787446345454;
  trueData[4][3][0] = -0.498848942870893;
  trueData[4][3][1] = 0.493704624120852;
  trueData[4][3][2] = -0.447282634936700;
  trueData[4][3][3] = -0.004259834618445;
  trueData[4][3][4] = 0.255815687289101;
  trueData[5][3][0] = -0.499477624643055;
  trueData[5][3][1] = 0.489950205708831;
  trueData[5][3][2] = 4.961601156036384;
  trueData[5][3][3] = 0.047253344343204;
  trueData[5][3][4] = 0.275914342724120;
  trueData[6][3][0] = -0.483017061662739;
  trueData[6][3][1] = 0.499491620097705;
  trueData[6][3][2] = -3.032569852634673;
  trueData[6][3][3] = -0.028881617644140;
  trueData[6][3][4] = 0.295717185129862;
  trueData[7][3][0] = -0.489663381656604;
  trueData[7][3][1] = 0.495389727655092;
  trueData[7][3][2] = -6.694887636266307;
  trueData[7][3][3] = -0.063760834631108;
  trueData[7][3][4] = 0.286915110073222;
  trueData[8][3][0] = -0.492179706430665;
  trueData[8][3][1] = 0.487934734952495;
  trueData[8][3][2] = -4.783191325007987;
  trueData[8][3][3] = -0.045554203095314;
  trueData[8][3][4] = 0.270941924784583;
  trueData[9][3][0] = -0.479464225341815;
  trueData[9][3][1] = 0.497003271606648;
  trueData[9][3][2] = 0.624140090371664;
  trueData[9][3][3] = 0.005944191336873;
  trueData[9][3][4] = 0.284749850716116;
  trueData[10][3][0] = -0.490197747736938;
  trueData[10][3][1] = 0.497560349512189;
  trueData[10][3][2] = 2.035170712889842;
  trueData[10][3][3] = 0.019382578217999;
  trueData[10][3][4] = 0.301217280929192;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);


  std::vector<std::vector<int>> procGrids = {
     {3,1,1,1}, {1,3,1,1}, {1,1,3,1} };

  for (auto & procGrid : procGrids)
  {
    std::vector<int> dataTensorDim = {3,5,7,11};
    TuckerMpi::Tensor<scalar_t> tensor(dataTensorDim, procGrid);
    TuckerMpi::read_tensor_binary(rank, tensor, "./tensor_data_files/3x5x7x11.bin");

    const std::array<Tucker::Metric,5> metricIDs{Tucker::Metric::MIN,
                Tucker::Metric::MAX, Tucker::Metric::SUM, Tucker::Metric::MEAN, 
                Tucker::Metric::VARIANCE};

    for(int mode=0; mode<tensor.rank(); ++mode) 
    {
      auto metrics = TuckerMpi::compute_slice_metrics(rank, tensor, mode, metricIDs);
      auto metrics_h = Tucker::create_mirror(metrics);
      Tucker::deep_copy(metrics_h, metrics);

      auto map = tensor.getDistribution().getMap(mode, false);
      for(int j=0; j<tensor.localExtent(mode); j++) 
      {
        const int globalJ = map->getGlobalIndex(j);

        for (int i=0; i<(int)metricIDs.size(); ++i){
          auto view = metrics_h.get(metricIDs[i]);
          scalar_t computed = (i<=3) ? view(j) : std::sqrt(view(j));
          scalar_t diff = std::abs( computed - trueData[globalJ][mode][i]);
          ASSERT_TRUE(diff < 100 * std::numeric_limits<scalar_t>::epsilon() );

          // if (rank==0){
          //   std::cout << mode 
          //    << " " << i 
          //    << " " << j 
          //    << " "
          //    << std::setprecision(13) << view(j) 
          //    << " " << trueData[globalJ][mode][i] 
          //    << " " << diff 
          //    << '\n';
          // }
        }
      }
    }

  }// loop over procGrid

}
