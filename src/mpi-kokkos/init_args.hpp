#ifndef TUCKER_MPIKOKKOS_HELP_HPP
#define TUCKER_MPIKOKKOS_HELP_HPP

template<class ScalarType>
struct InputArgs
{
  bool boolAuto                         ;
  bool boolSTHOSVD                      ;
  bool boolWriteSTHOSVD                 ;
  bool boolPrintOptions                 ;
  bool boolWritePreprocessed            ;
  bool boolUseOldGram                   ;
  bool boolUseLQ                        ;
  bool boolPrintSV                      ;
  bool boolReconstruct                  ;
  bool useButterflyTSQR                 ;
  
  ScalarType tol                        ;
  ScalarType stdThresh                  ;

  Tucker::SizeArray* I_dims             ;
  Tucker::SizeArray* R_dims             ;
  Tucker::SizeArray* proc_grid_dims     ;
  Tucker::SizeArray* modeOrder          ;

  std::string scaling_type              ;
  std::string sthosvd_dir               ;
  std::string sthosvd_fn                ;
  std::string sv_dir                    ;
  std::string sv_fn                     ;
  std::string in_fns_file               ;
  std::string pre_fns_file              ;
  std::string reconstruct_report_file   ;
  std::string stats_file                ;
  std::string timing_file               ;
  int nd                                ;
  int scale_mode                        ;
};

/**
 * Print options
 */
template<class ScalarType>
void print_args(const InputArgs<ScalarType> & args)
{
  if (args.boolPrintOptions) {
    std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
    std::cout << "- Global dims = " << *args.I_dims << std::endl << std::endl;
    
    std::cout << "The global dimensions of the processor grid\n";
    std::cout << "- Grid dims = " << *args.proc_grid_dims << std::endl << std::endl;
    
    std::cout << "Mode order for decomposition\n";
    std::cout << "- Decompose mode order = " << *args.modeOrder << std::endl << std::endl;
    
    std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
    std::cout << "- Automatic rank determination = " << (args.boolAuto ? "true" : "false") << std::endl << std::endl;

    std::cout << "Used for automatic rank determination; the desired error rate\n";
    std::cout << "- SV Threshold = " << args.tol << std::endl << std::endl;

    if(!args.boolAuto) {
      std::cout << "Global dimensions of the desired core tensor\n";
      std::cout << "Not used if \"Automatic rank determination\" is enabled\n";
      std::cout << "- Ranks = " << *args.R_dims << std::endl << std::endl;
    }

    std::cout << "List of filenames of raw data to be read\n";
    std::cout << "- Input file list = " << args.in_fns_file << std::endl << std::endl;

    std::cout << "How to scale the tensor\n";
    std::cout << "- Scaling type = " << args.scaling_type << std::endl << std::endl;

    std::cout << "Which mode's hyperslices will be scaled\n";
    std::cout << "- Scale mode = " << args.scale_mode << std::endl << std::endl;

    std::cout << "Threshold for standard deviation before we simply set it to 1\n";
    std::cout << "Used in StandardCentering scaling\n";
    std::cout << "- STD Threshold = " << args.stdThresh << std::endl << std::endl;

    std::cout << "If true, perform ST-HOSVD\n";
    std::cout << "- Perform STHOSVD = " << (args.boolSTHOSVD ? "true" : "false") << std::endl << std::endl;

    std::cout << "If true, use the old Gram algorithm; otherwise use the new one\n";
    std::cout << "- Use old Gram = " << (args.boolUseOldGram ? "true" : "false") << std::endl << std::endl;

    std::cout << "Location of a report of the reconstruction errors \n";
    std::cout << "- Reconstruction report file = " << args.reconstruct_report_file << std::endl << std::endl;

    std::cout << "Location of statistics file containing min, max, mean, and std of each hyperslice\n";
    std::cout << "- Stats file = " << args.stats_file << std::endl << std::endl;

    std::cout << "If true, write the preprocessed data to a file\n";
    std::cout << "- Write preprocessed data = " << (args.boolWritePreprocessed ? "true" : "false") << std::endl << std::endl;

    std::cout << "File containing a list of filenames to output the scaled data into\n";
    std::cout << "- Preprocessed output file list = " << args.pre_fns_file << std::endl << std::endl;

    std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors\n";
    std::cout << "- Write STHOSVD result = " << (args.boolWriteSTHOSVD ? "true" : "false") << std::endl << std::endl;

    std::cout << "Directory location of ST-HOSVD output files\n";
    if(args.boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
    std::cout << "- STHOSVD directory = " << args.sthosvd_dir << std::endl << std::endl;

    std::cout << "Base name of ST-HOSVD output files\n";
    std::cout << "- STHOSVD file prefix = " << args.sthosvd_fn << std::endl << std::endl;

    std::cout << "Directory to place singular value files into\n";
    if(args.boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
    std::cout << "- SV directory = " << args.sv_dir << std::endl << std::endl;

    std::cout << "Base name for writing the singular value files\n";
    std::cout << "- SV file prefix = " << args.sv_fn << std::endl << std::endl;

    std::cout << "Name of the CSV file holding the timing results\n";
    std::cout << "- Timing file = " << args.timing_file << std::endl << std::endl;

    std::cout << "If true, reconstruct an approximation of the original tensor after ST-HOSVD\n";
    if(args.boolReconstruct) std::cout << "WARNING: This may require a great deal of memory\n";
    std::cout << "- Reconstruct tensor = " << (args.boolReconstruct ? "true" : "false") << std::endl << std::endl;

    std::cout << "If true, print the parameters\n";
    std::cout << "- Print options = " << (args.boolPrintOptions ? "true" : "false") << std::endl << std::endl;
  } 
}

template<class ScalarType>
InputArgs<ScalarType> parse_input_file(const std::vector<std::string> & fileAsString)
{
  InputArgs<ScalarType> args;
  
  args.boolAuto                     = Tucker::stringParse<bool>(fileAsString, "Automatic rank determination", false);
  args.boolSTHOSVD                  = Tucker::stringParse<bool>(fileAsString, "Perform STHOSVD", false);
  args.boolWriteSTHOSVD             = Tucker::stringParse<bool>(fileAsString, "Write core tensor and factor matrices", false);
  args.boolPrintOptions             = Tucker::stringParse<bool>(fileAsString, "Print options", false);
  args.boolWritePreprocessed        = Tucker::stringParse<bool>(fileAsString, "Write preprocessed data", false);
  args.boolUseOldGram               = Tucker::stringParse<bool>(fileAsString, "Use old Gram", false);
  args.boolUseLQ                    = Tucker::stringParse<bool>(fileAsString, "Compute SVD via LQ", false);
  args.boolPrintSV                  = Tucker::stringParse<bool>(fileAsString, "Print factor matrices", false);
  args.boolReconstruct              = Tucker::stringParse<bool>(fileAsString, "Reconstruct tensor", false);
  args.useButterflyTSQR             = Tucker::stringParse<bool>(fileAsString, "Use butterfly TSQR", false);
  
  args.tol                          = Tucker::stringParse<ScalarType>(fileAsString, "SV Threshold", 1e-6);
  args.stdThresh                    = Tucker::stringParse<ScalarType>(fileAsString, "STD Threshold", 1e-9);

  args.I_dims                       = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  if (!args.boolAuto) {
    args.R_dims                     = Tucker::stringParseSizeArray(fileAsString, "Ranks");
  } else {
    args.R_dims                     = 0;
  }
  args.proc_grid_dims               = Tucker::stringParseSizeArray(fileAsString, "Grid dims");
  args.modeOrder                    = Tucker::stringParseSizeArray(fileAsString, "Decompose mode order");

  args.scaling_type                 = Tucker::stringParse<std::string>(fileAsString, "Scaling type", "None");
  args.sthosvd_dir                  = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  args.sthosvd_fn                   = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  args.sv_dir                       = Tucker::stringParse<std::string>(fileAsString, "SV directory", ".");
  args.sv_fn                        = Tucker::stringParse<std::string>(fileAsString, "SV file prefix", "sv");
  args.in_fns_file                  = Tucker::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");
  args.pre_fns_file                 = Tucker::stringParse<std::string>(fileAsString, "Preprocessed output file list", "pre.txt");
  args.reconstruct_report_file      = Tucker::stringParse<std::string>(fileAsString, "Reconstruction report file", "reconstruction.txt");
  args.stats_file                   = Tucker::stringParse<std::string>(fileAsString, "Stats file", "stats.txt");
  args.timing_file                  = Tucker::stringParse<std::string>(fileAsString, "Timing file", "runtime.csv");

  args.nd = args.I_dims->size();
  args.scale_mode                   = Tucker::stringParse<int>(fileAsString, "Scale mode", args.nd-1);
  
  return args;
}

/**
 * Assert that we either have automatic rank determination
 * or the user has supplied their own ranks
 */
template<class ScalarType>
int check_args(InputArgs<ScalarType> & args)
{

  if(!args.boolAuto && !args.R_dims) {
    std::cerr << "ERROR: Please either enable Automatic rank determination, "
              << "or provide the desired core tensor size via the Ranks parameter\n";
    return EXIT_FAILURE;
  }

  if(args.tol >= 1) {
    std::cerr << "ERROR: The reconstruction error tolerance should be smaller than 1. \n";
    return EXIT_FAILURE;
  }

  if(!args.modeOrder){
    args.modeOrder = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(args.nd);
    for(int i=0; i<args.nd; i++){
      args.modeOrder->data()[i] = i;
      // std::cout <<"modeOrder[" <<i<<"]: " << args.modeOrder->data()[i] << " ";
    }
    // std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}

template<class ScalarType>
void chech_array_sizes(const InputArgs<ScalarType> args, const int rank, const int nprocs)
{
  // Does |grid| == nprocs?
  if ((int)args.proc_grid_dims->prod() != nprocs){
    if (rank==0) {
      std::cerr << "Processor grid dimensions do not multiply to nprocs" << std::endl;
      std::cout << "Processor grid dimensions: " << *args.proc_grid_dims << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (args.nd != args.proc_grid_dims->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of global dimension array (" << args.nd;
      std::cerr << ") must be equal to the size of the processor grid ("
          << args.proc_grid_dims->size() << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!args.boolAuto && args.R_dims->size() != 0 && args.R_dims->size() != args.nd) {
    if (rank == 0) {
      std::cerr << "Error: The size of the ranks array (" << args.R_dims->size();
      std::cerr << ") must be 0 or equal to the size of the processor grid (" << args.nd << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

#endif // End of TUCKER_MPIKOKKOS_HELP_HPP