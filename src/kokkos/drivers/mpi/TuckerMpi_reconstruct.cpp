#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <functional>

#include "ParameterFileParserUtils.hpp"
#include "CmdLineParse.hpp"
#include "TuckerMpi.hpp"

int main(int argc, char* argv[])
{
  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using exec_space = typename memory_space::execution_space;

  int ret = -1;
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  try {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    int mpiRank, nprocs;
    const MPI_Comm& comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &mpiRank);
    MPI_Comm_size(comm, &nprocs);

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
    std::vector<int> proc_grid_dims = Tucker::parse_multivalued_field<int>(fileAsStrings, "Grid dims");
    std::vector<int> modeOrder      = Tucker::parse_multivalued_field<int>(fileAsStrings, "Decompose mode order");
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
    if(proc_grid_dims.size() == 0) {
      if (mpiRank == 0)
        std::cerr << "Error: Grid dims is a required parameter\n";
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    if (subs_begin.size() == 0) {
      std::cerr << "Error: Beginning subscripts is a required parameter\n";
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    if (subs_end.size() == 0) {
      std::cerr << "Error: Ending subscripts is a required parameter\n";
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    auto print_vec = [](const auto& vec)
    {
      std::copy(vec.begin(), vec.end(),
                std::ostream_iterator<int>(std::cout, " "));
    };

    if (mpiRank == 0 && boolPrintOptions) {
      std::cout << "Global dimensions of the processor grid\n";
      std::cout << "- Grid dims = ";
      print_vec(proc_grid_dims);
      std::cout << std::endl << std::endl;
 
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
    int nd = proc_grid_dims.size();

    auto prod = [](const auto& vec)
    {
      return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int>());
    };

    // Does |grid| == nprocs?
    if (prod(proc_grid_dims) != nprocs) {
      if (mpiRank==0) {
        std::cerr << "Processor grid dimensions do not multiply to nprocs" << std::endl;
        std::cout << "Processor grid dimensions: ";
        print_vec(proc_grid_dims);
        std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (nd != subs_end.size()) {
      if (mpiRank==0) {
        std::cerr << "Error: The size of the subs_end array (" << subs_end.size();
        std::cerr << ") must be equal to the size of the subs_begin array ("
                  << nd << ")" << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rec_order.size() != 0 && nd != rec_order.size()) {
      if (mpiRank==0) {
        std::cerr << "Error: The size of the rec_order array (" << rec_order.size();
        std::cerr << ") must be equal to the size of the subs_begin array ("
                  << nd << ")" << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ////////////////////////////////////////////////////////
    // Make sure the subs begin and end arrays make sense //
    ////////////////////////////////////////////////////////
    for(int i=0; i<nd; i++) {
      if(subs_begin[i] < 0) {
        if (mpiRank==0) {
          std::cerr << "Error: subs_begin[" << i << "] = "
                    << subs_begin[i] << " < 0\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      if (subs_begin[i] > subs_end[i]) {
        if (mpiRank==0) {
          std::cerr << "Error: subs_begin[" << i << "] = "
                    << subs_begin[i] << " > subs_end[" << i << "] = "
                    << subs_end[i] << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }

    ////////////////////////////////////
    // Read the core size from a file //
    ////////////////////////////////////
    std::vector<int> coreSize(nd);
    std::ifstream ifs;
    if (mpiRank == 0) {
      std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn + "_ranks.txt";
      ifs.open(dimFilename);
      if (!ifs.is_open()) {
        std::cerr << "Failed to open core size file: " << dimFilename
                  << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      for (int mode=0; mode<nd; mode++) {
        ifs >> coreSize[mode];
      }
      ifs.close();
    }
    MPI_Bcast(coreSize.data(),nd,MPI_INT,0,MPI_COMM_WORLD);

    //////////////////////////////////////
    // Read the global size from a file //
    //////////////////////////////////////
    std::vector<int> I_dims(nd);
    if (mpiRank == 0) {
      std::string sizeFilename = sthosvd_dir + "/" + sthosvd_fn + "_size.txt";
      ifs.open(sizeFilename);
      if (!ifs.is_open()) {
        std::cerr << "Failed to open global size file: " << sizeFilename
                  << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      for (int mode=0; mode<nd; mode++) {
        ifs >> I_dims[mode];
      }
      ifs.close();
    }
    MPI_Bcast(I_dims.data(),nd,MPI_INT,0,MPI_COMM_WORLD);

    //////////////////////////////////////////////
    // Make sure the core size data makes sense //
    //////////////////////////////////////////////
    for (int i=0; i<nd; i++) {
      if (coreSize[i] <= 0) {
        if (mpiRank==0) {
          std::cerr << "coreSize[" << i << "] = " << coreSize[i]
                    << " <= 0\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      if (coreSize[i] > I_dims[i]) {
        if (mpiRank==0) {
          std::cerr << "coreSize[" << i << "] = " << coreSize[i]
                    << " > I_dims[" << I_dims[i] << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD,1);
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

      if (mpiRank == 0) {
        std::cout << "Reconstruction order: ";
        print_vec(rec_order);
        std::cout << std::endl;

        orderTimer.stop();
        std::cout << "Computing the optimal reconstruction order: " << orderTimer.duration() << " s\n";
      }
    }

    //////////////////////////////////////////////////////////
    // Make sure the reconstruction order array makes sense //
    //////////////////////////////////////////////////////////
    for (int i=0; i<nd; i++) {
      if (rec_order[i] < 0) {
        if (mpiRank == 0) {
          std::cerr << "Error: rec_order[" << i << "] = "
                    << rec_order[i] << " < 0\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      if (rec_order[i] >= nd) {
        if (mpiRank == 0) {
          std::cerr << "Error: rec_order[" << i << "] = "
                    << rec_order[i] << " >= nd = " << nd << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      for(int j=i+1; j<nd; j++) {
        if (rec_order[i] == rec_order[j]) {
          if (mpiRank == 0) {
            std::cerr << "Error: rec_order[" << i << "] == rec_order["
                      << j << "] = " << rec_order[i] << std::endl;
          }
          MPI_Barrier(MPI_COMM_WORLD);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
    }

    /////////////////////////////////////////////
    // Set up distribution object for the core //
    /////////////////////////////////////////////
    TuckerMpi::Distribution dist(coreSize, proc_grid_dims);

    ///////////////////////////
    // Read core tensor data //
    ///////////////////////////
    Tucker::Timer readTimer;
    MPI_Barrier(MPI_COMM_WORLD);
    readTimer.start();
    std::string coreFilename = sthosvd_dir + "/" + sthosvd_fn + "_core.mpi";
    using tensor_type = TuckerMpi::Tensor<scalar_t, memory_space>;
    tensor_type G(dist);
    TuckerMpi::read_tensor_binary(mpiRank, G, coreFilename);
    if (mpiRank == 0) {
      size_t local_nnz = G.localSize();
      size_t global_nnz = G.globalSize();
      std::cout << "Local core tensor size: ";
      for (int i=0; i<G.rank(); ++i)
        std::cout << G.localExtent(i) << " ";
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(double));
      std::cout << "Global core tensor size: ";
      for (int i=0; i<G.rank(); ++i)
        std::cout << G.globalExtent(i) << " ";
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(double));
    }

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
    MPI_Barrier(MPI_COMM_WORLD);
    readTimer.stop();
    if (mpiRank == 0) {
      std::cout << "Time spent reading: " << readTimer.duration() << "s\n";
    }

    ////////////////////////////////////////////////////
    // Reconstruct the requested pieces of the tensor //
    ////////////////////////////////////////////////////
    Tucker::Timer reconstructTimer;
    MPI_Barrier(MPI_COMM_WORLD);
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
      G = TuckerMpi::ttm(G, mode, factMat, false);

      if (mpiRank == 0) {
        size_t local_nnz = G.localSize();
        size_t global_nnz = G.globalSize();
        std::cout << "Local tensor size after reconstruction iteration "
                  << i << ": ";
        for (int j=0; j<G.rank(); ++j)
          std::cout << G.localExtent(j) << " ";
        std::cout << ", or ";
        Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(double));
        std::cout << "Local tensor size after reconstruction iteration "
                  << i << ": ";
        for (int j=0; j<G.rank(); ++j)
          std::cout << G.globalExtent(j) << " ";
        std::cout << ", or ";
        Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(double));
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    reconstructTimer.stop();
    if (mpiRank == 0)
      std::cout << "Time spent reconstructing: " << reconstructTimer.duration() << "s\n";

    ///////////////////////////////////////////////////////
    // Scale and shift if necessary                      //
    // This step only happens if the scaling file exists //
    ///////////////////////////////////////////////////////
    Tucker::Timer scaleTimer;
    MPI_Barrier(MPI_COMM_WORLD);
    scaleTimer.start();
    int scale_mode;
    if (mpiRank == 0) {
      std::string scaleFilename = sthosvd_dir + "/" + sthosvd_fn + "_scale.txt";
      ifs.open(scaleFilename);

      if (ifs.is_open()) {
        int scale_mode;
        ifs >> scale_mode;
      }
      else {
        std::cout << "Failed to open scaling and shifting file: " << scaleFilename
                  << "\nAssuming no scaling and shifting was performed\n";
        scale_mode = nd;
      }
    }
    MPI_Bcast(&scale_mode,1,MPI_INT,0,MPI_COMM_WORLD);

    if (scale_mode < nd) {
      int scale_size = 1 + subs_end[scale_mode] - subs_begin[scale_mode];
      std::vector<double> scales(scale_size);
      std::vector<double> shifts(scale_size);
      if (mpiRank == 0) {
        std::cout << "Scaling mode " << scale_mode << std::endl;
        for (int i=0; i<I_dims[scale_mode]; i++) {
          double scale, shift;
          ifs >> scale >> shift;
          if (i >= subs_begin[scale_mode] && i <= subs_end[scale_mode]) {
            scales[i-subs_begin[scale_mode]] = 1./scale;
            shifts[i-subs_begin[scale_mode]] = -shift/scale;
          }
        }
        ifs.close();
      }
      TuckerMpi::MPI_Bcast_(scales.data(),scale_size,0,MPI_COMM_WORLD);
      TuckerMpi::MPI_Bcast_(shifts.data(),scale_size,0,MPI_COMM_WORLD);

      int row_begin = dist.getMap(scale_mode)->getGlobalIndex(0);
      if(row_begin >= 0) {
        Kokkos::View<double*,Kokkos::HostSpace> scales_h(
          scales.data()+row_begin, scale_size-row_begin);
        Kokkos::View<double*,Kokkos::HostSpace> shifts_h(
          shifts.data()+row_begin, scale_size-row_begin);
        auto scales_d =
          Kokkos::create_mirror_view_and_copy(exec_space(), scales_h);
        auto shifts_d =
          Kokkos::create_mirror_view_and_copy(exec_space(), shifts_h);
        TuckerOnNode::transform_slices(G.localTensor(), scale_mode, scales_d, shifts_d);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    scaleTimer.stop();
    if (mpiRank == 0)
      std::cout << "Time spent shifting and scaling: " << scaleTimer.duration() << "s\n";

    ////////////////////////////////////////////
    // Write the reconstructed tensor to disk //
    ////////////////////////////////////////////
    Tucker::Timer writeTimer;
    MPI_Barrier(MPI_COMM_WORLD);
    writeTimer.start();
    TuckerMpi::write_tensor_binary(mpiRank, G, outputFilenames);
    MPI_Barrier(MPI_COMM_WORLD);
    writeTimer.stop();
    if (mpiRank == 0)
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
  MPI_Finalize();

  return ret;
}
