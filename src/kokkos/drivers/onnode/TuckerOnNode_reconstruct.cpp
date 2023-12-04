#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>

#include "ParameterFileParserUtils.hpp"
#include "CmdLineParse.hpp"
#include "TuckerOnNode.hpp"

int main(int argc, char* argv[])
{
  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using exec_space = typename memory_space::execution_space;

  int ret = -1;
  Kokkos::initialize(argc, argv);
  try {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    //
    // Get the name of the input file
    //
    const auto paramfn = Tucker::parse_cmdline_or(
      argc, (const char**)argv, "--parameter-file", "paramfile.txt");

    //
    // Parse parameter file
    //
    const auto fileAsStrings   = Tucker::read_file_as_strings(paramfn);
    bool boolPrintOptions       = Tucker::string_parse<bool>(fileAsStrings, "Print options", false);
    bool boolOptimizeFlops      = Tucker::string_parse<bool>(fileAsStrings, "Optimize flops", true);
    std::vector<int> subs_begin = Tucker::parse_multivalued_field<int>(fileAsStrings, "Beginning subscripts");
    std::vector<int> subs_end   = Tucker::parse_multivalued_field<int>(fileAsStrings, "Ending subscripts");
    std::vector<int> rec_order  = Tucker::parse_multivalued_field<int>(fileAsStrings, "Reconstruction order");
    std::string sthosvd_dir     = Tucker::string_parse<std::string>(fileAsStrings, "STHOSVD directory", "compressed");
    std::string sthosvd_fn      = Tucker::string_parse<std::string>(fileAsStrings, "STHOSVD file prefix", "sthosvd");
    std::string out_fns_file    = Tucker::string_parse<std::string>(fileAsStrings, "Output file list", "rec.txt");
    std::vector<std::string> outputFilenames = Tucker::read_file_as_strings(out_fns_file);

    /////////////////////////////////////////////////
    // Assert that none of the size arrays are empty //
    /////////////////////////////////////////////////
    if (subs_begin.size() == 0) {
      std::cerr << "Error: Beginning subscripts is a required parameter\n";
      std::abort();
    }

    if (subs_end.size() == 0) {
      std::cerr << "Error: Ending subscripts is a required parameter\n";
      std::abort();
    }

    auto print_vec = [](const auto& vec)
    {
      std::copy(vec.begin(), vec.end(),
                std::ostream_iterator<int>(std::cout, " "));
    };

    if (boolPrintOptions) {
      std::cout << "Start of subscripts to be recomputed\n";
      std::cout << "- Beginning subscripts = ";
      print_vec(subs_begin);
      std::cout << std::endl << std::endl;

      std::cout << "End of subscripts to be recomputed\n";
      std::cout << "- Ending subscripts = ";
      print_vec(subs_end);
      std::cout << std::endl << std::endl;

      std::cout << "Directory location of ST-HOSVD output files\n";
      std::cout << "- STHOSVD directory = " << sthosvd_dir << std::endl << std::endl;

      std::cout << "Base name of ST-HOSVD output files\n";
      std::cout << "- STHOSVD file prefix = " << sthosvd_fn << std::endl << std::endl;

      std::cout << "File containing a list of filenames to output the reconstructed data into\n";
      std::cout << "- Output file list = " << out_fns_file << std::endl << std::endl;

      if (rec_order.size() > 0) {
        std::cout << "Mode order for reconstruction\n";
        std::cout << "NOTE: if left unspecified, the memory-optimal one will be automatically selected\n";
        std::cout << "- Reconstruction order = ";
        print_vec(rec_order);
        std::cout << std::endl << std::endl;
      }
      else {
        std::cout << "If true, choose the reconstruction ordering that requires the minimum number of flops\n";
        std::cout << "Otherwise, choose the one that will require the least memory\n";
        std::cout << "- Optimize flops = " << (boolOptimizeFlops ? "true" : "false") << std::endl << std::endl;
      }

      std::cout << "If true, print the parameters\n";
      std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

      std::cout << std::endl;
    }

    ///////////////////////
    // Check array sizes //
    ///////////////////////
    const int nd = subs_begin.size();

    if (nd != subs_end.size()) {
      std::cerr << "Error: The size of the subs_end array (" << subs_end.size();
      std::cerr << ") must be equal to the size of the subs_begin array ("
                << nd << ")" << std::endl;
      std::abort();
    }

    if (rec_order.size() != 0 && nd != rec_order.size()) {
      std::cerr << "Error: The size of the rec_order array (" << rec_order.size();
      std::cerr << ") must be equal to the size of the subs_begin array ("
                << nd << ")" << std::endl;
      std::abort();
    }

    ////////////////////////////////////////////////////////
    // Make sure the subs begin and end arrays make sense //
    ////////////////////////////////////////////////////////
    for(int i=0; i<nd; i++) {
      if(subs_begin[i] < 0) {
        std::cerr << "Error: subs_begin[" << i << "] = "
                  << subs_begin[i] << " < 0\n";
        std::abort();
      }

      if (subs_begin[i] > subs_end[i]) {
        std::cerr << "Error: subs_begin[" << i << "] = "
          << subs_begin[i] << " > subs_end[" << i << "] = "
          << subs_end[i] << std::endl;
        std::abort();
      }
    }

    ////////////////////////////////////
    // Read the core size from a file //
    ////////////////////////////////////
    std::vector<int> coreSize(nd);
    std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn + "_ranks.txt";
    std::ifstream ifs(dimFilename);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open core size file: " << dimFilename
                << std::endl;
      std::abort();
    }
    for (int mode=0; mode<nd; mode++) {
      ifs >> coreSize[mode];
    }
    ifs.close();

    //////////////////////////////////////
    // Read the global size from a file //
    //////////////////////////////////////
    std::vector<int> I_dims(nd);
    std::string sizeFilename = sthosvd_dir + "/" + sthosvd_fn + "_size.txt";
    ifs.open(sizeFilename);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open global size file: " << sizeFilename
                << std::endl;
      std::abort();
    }
    for (int mode=0; mode<nd; mode++) {
      ifs >> I_dims[mode];
    }
    ifs.close();

    //////////////////////////////////////////////
    // Make sure the core size data makes sense //
    //////////////////////////////////////////////
    for (int i=0; i<nd; i++) {
      if (coreSize[i] <= 0) {
        std::cerr << "coreSize[" << i << "] = " << coreSize[i]
                  << " <= 0\n";
        std::abort();
      }

      if (coreSize[i] > I_dims[i]) {
        std::cerr << "coreSize[" << i << "] = " << coreSize[i]
                  << " > I_dims[" << I_dims[i] << std::endl;
        std::abort();
      }
    }

    ////////////////////////////////////////////////////////////
    // Create the optimal reconstruction order if unspecified //
    ////////////////////////////////////////////////////////////
    if (rec_order.size() == 0) {
      Tucker::Timer orderTimer;
      orderTimer.start();

      // Compute the size of the final tensor
      std::vector<int> rec_size(nd);
      for (int i=0; i<nd; i++) {
        rec_size[i] = 1 + subs_end[i] - subs_begin[i];
      }


      // Create the SizeArray
      rec_order.resize(nd);
      std::vector<int> temp_order(nd);
      for (int i=0; i<nd; i++) {
        rec_order[i] = i;
        temp_order[i] = i;
      }

      auto prod = [](const auto& vec)
      {
        return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int>());
      };

      size_t min_flops = static_cast<size_t>(-1);
      size_t min_mem = static_cast<size_t>(-1);
      std::vector<int> current_dims(nd);
      do {
        // Initialize current dimensions
        for (int i=0; i<nd; i++) {
          current_dims[i] = coreSize[i];
        }

        if (boolOptimizeFlops) {
          // Compute the number of flops
          size_t flops = 0;
          for (int i=0; i<nd; i++) {
            flops += rec_size[temp_order[i]] * prod(current_dims);
            current_dims[temp_order[i]] = rec_size[temp_order[i]];
          }

          if(min_flops == static_cast<size_t>(-1) || flops < min_flops) {
            min_flops = flops;
            for(int i=0; i<nd; i++) {
              rec_order[i] = temp_order[i];
            }
          }
        }
        else {
          // Compute the memory footprint
          size_t mem = std::inner_product(
            rec_size.begin(), rec_size.end(), current_dims.begin(), 0);
          size_t max_mem = mem;
          for (int i=0; i<nd; i++) {
            mem += prod(current_dims);
            current_dims[temp_order[i]] = rec_size[temp_order[i]];
            mem += prod(current_dims);
            mem -= coreSize[temp_order[i]]*rec_size[temp_order[i]];
            max_mem = std::max(mem,max_mem);
          }

          if(min_mem == static_cast<size_t>(-1) || max_mem < min_mem) {
            min_mem = max_mem;
            for (int i=0; i<nd; i++) {
              rec_order[i] = temp_order[i];
            }
          }
        }
      } while( std::next_permutation(temp_order.begin(),temp_order.end()) );

      std::cout << "Reconstruction order: ";
      print_vec(rec_order);
      std::cout << std::endl;

      orderTimer.stop();
      std::cout << "Computing the optimal reconstruction order: " << orderTimer.duration() << " s\n";
    }

    //////////////////////////////////////////////////////////
    // Make sure the reconstruction order array makes sense //
    //////////////////////////////////////////////////////////
    for (int i=0; i<nd; i++) {
      if (rec_order[i] < 0) {
        std::cerr << "Error: rec_order[" << i << "] = "
                  << rec_order[i] << " < 0\n";
        std::abort();
      }

      if (rec_order[i] >= nd) {
        std::cerr << "Error: rec_order[" << i << "] = "
                  << rec_order[i] << " >= nd = " << nd << std::endl;
        std::abort();
      }

      for(int j=i+1; j<nd; j++) {
        if (rec_order[i] == rec_order[j]) {
          std::cerr << "Error: rec_order[" << i << "] == rec_order["
                    << j << "] = " << rec_order[i] << std::endl;
          std::abort();
        }
      }
    }

    ///////////////////////////
    // Read core tensor data //
    ///////////////////////////
    Tucker::Timer readTimer;
    readTimer.start();
    std::string coreFilename = sthosvd_dir + "/" + sthosvd_fn + "_core.mpi";
    using tensor_type = TuckerOnNode::Tensor<scalar_t, memory_space>;
    tensor_type G(coreSize);
    TuckerOnNode::read_tensor_binary(G, coreFilename);
    size_t nnz = G.size();
    std::cout << "Core tensor size: ";
    for (int i=0; i<G.rank(); ++i)
      std::cout << G.extent(i) << " ";
    std::cout << ", or ";
    Tucker::print_bytes_to_stream(std::cout, nnz*sizeof(double));

    //////////////////////////
    // Read factor matrices //
    //////////////////////////
    using factor_1d_view_type =
      Kokkos::View<scalar_t*,Kokkos::LayoutLeft,memory_space>;
    using factor_2d_view_type =
      Kokkos::View<scalar_t**,Kokkos::LayoutLeft,memory_space>;
    std::vector<factor_1d_view_type> factors(nd);
    for (int mode=0; mode<nd; mode++)
    {
      std::ostringstream ss;
      ss << sthosvd_dir << "/" << sthosvd_fn << "_mat_" << mode << ".mpi";
      factors[mode] =
        factor_1d_view_type("fac", I_dims[mode]*coreSize[mode]);
      auto f_h = Kokkos::create_mirror_view(factors[mode]);
      Tucker::fill_rank1_view_from_binary_file(f_h, ss.str());
      Kokkos::deep_copy(factors[mode], f_h);
    }
    readTimer.stop();
    std::cout << "Time spent reading: " << readTimer.duration() << "s\n";

    ////////////////////////////////////////////////////
    // Reconstruct the requested pieces of the tensor //
    ////////////////////////////////////////////////////
    Tucker::Timer reconstructTimer;
    reconstructTimer.start();
    for (int i=0; i<nd; i++)
    {
      int mode = rec_order[i];
      // Grab the requested rows of the factor matrix
      int start_subs = subs_begin[mode];
      int end_subs = subs_end[mode]+1; // subs_end is supposed to be inclusive
      factor_2d_view_type fac(factors[mode].data(), I_dims[mode], coreSize[mode]);
      auto factMat =
        Kokkos::subview(fac, std::make_pair(start_subs, end_subs), Kokkos::ALL);
      G = TuckerOnNode::ttm(G, mode, factMat, false);

      size_t nnz = G.size();
      std::cout << "Tensor size after reconstruction iteration "
                << i << ": ";
      for (int j=0; j<G.rank(); ++j)
        std::cout << G.extent(j) << " ";
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, nnz*sizeof(double));
    }
    reconstructTimer.stop();
    std::cout << "Time spent reconstructing: " << reconstructTimer.duration() << "s\n";

    ///////////////////////////////////////////////////////
    // Scale and shift if necessary                      //
    // This step only happens if the scaling file exists //
    ///////////////////////////////////////////////////////
    Tucker::Timer scaleTimer;
    scaleTimer.start();
    std::string scaleFilename = sthosvd_dir + "/" + sthosvd_fn + "_scale.txt";
    ifs.open(scaleFilename);

    if (ifs.is_open()) {
      int scale_mode;
      ifs >> scale_mode;
      std::cout << "Scaling mode " << scale_mode << std::endl;
      int scale_size = 1 + subs_end[scale_mode] - subs_begin[scale_mode];
      std::vector<double> scales(scale_size);
      std::vector<double> shifts(scale_size);
      for (int i=0; i<I_dims[scale_mode]; i++) {
        double scale, shift;
        ifs >> scale >> shift;
        if (i >= subs_begin[scale_mode] && i <= subs_end[scale_mode]) {
          scales[i-subs_begin[scale_mode]] = 1./scale;
          shifts[i-subs_begin[scale_mode]] = -shift/scale;
        }
      }
      ifs.close();

      Kokkos::View<double*,Kokkos::HostSpace> scales_h(
        scales.data(), scale_size);
      Kokkos::View<double*,Kokkos::HostSpace> shifts_h(
        shifts.data(), scale_size);
      auto scales_d =
        Kokkos::create_mirror_view_and_copy(exec_space(), scales_h);
      auto shifts_d =
        Kokkos::create_mirror_view_and_copy(exec_space(), shifts_h);
      TuckerOnNode::transform_slices(G, scale_mode, scales_d, shifts_d);
    }
    else {
      std::cout << "Failed to open scaling and shifting file: " << scaleFilename
                << "\nAssuming no scaling and shifting was performed\n";
    }
    scaleTimer.stop();
    std::cout << "Time spent shifting and scaling: " << scaleTimer.duration() << "s\n";

    ////////////////////////////////////////////
    // Write the reconstructed tensor to disk //
    ////////////////////////////////////////////
    Tucker::Timer writeTimer;
    writeTimer.start();
    TuckerOnNode::write_tensor_binary(G, outputFilenames);
    writeTimer.stop();
    std::cout << "Time spent writing: " << writeTimer.duration() << "s\n";

    ret = 0;
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (std::string& e) {
    std::cerr << e << std::endl;
  }
  catch (...) {
    std::cerr << "Caught unknown exception!" << std::endl;
  }
  Kokkos::finalize();

  return ret;
}