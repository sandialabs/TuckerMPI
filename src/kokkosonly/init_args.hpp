#ifndef TUCKER_MPIKOKKOS_HELP_HPP
#define TUCKER_MPIKOKKOS_HELP_HPP

template<class ScalarType>
struct InputArgs
{
  bool boolAuto;
  bool boolSTHOSVD;
  bool boolWriteSTHOSVD;
  bool boolPrintOptions;
  bool boolWritePreprocessed;
  bool boolUseLQ;

  ScalarType tol;
  ScalarType stdThresh;

  std::string scaling_type;
  std::string sthosvd_dir;
  std::string sthosvd_fn;
  std::string sv_dir;
  std::string sv_fn;
  std::string in_fns_file;
  std::string pre_fns_file;
  std::string stats_file;
  int nd;
  int scale_mode;
};

template<class ScalarType>
void print_args(const InputArgs<ScalarType> & args)
{
  if (args.boolPrintOptions) {
    std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
    std::cout << "- Automatic rank determination = " << (args.boolAuto ? "true" : "false") << std::endl << std::endl;

    std::cout << "Used for automatic rank determination; the desired error rate\n";
    std::cout << "- SV Threshold = " << args.tol << std::endl << std::endl;

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

    std::cout << "Location of statistics file containing min, max, mean, and std of each hyperslice\n";
    std::cout << "- Stats file = " << args.stats_file << std::endl << std::endl;

    std::cout << "If true, write the preprocessed data to a file\n";
    std::cout << "- Write preprocessed data = " << (args.boolWritePreprocessed ? "true" : "false") << std::endl << std::endl;

    std::cout << "File containing a list of filenames to output the scaled data into\n";
    std::cout << "- Preprocessed output file list = " << args.pre_fns_file << std::endl << std::endl;

    std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors)\n";
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

    std::cout << "If true, print the parameters\n";
    std::cout << "- Print options = " << (args.boolPrintOptions ? "true" : "false") << std::endl << std::endl;

    std::cout << std::endl;
  }
}

template<class ScalarType>
InputArgs<ScalarType> parse_input_file(const std::vector<std::string> & fileAsString)
{
  InputArgs<ScalarType> args;

  args.boolAuto                     = TuckerKokkos::stringParse<bool>(fileAsString, "Automatic rank determination", false);
  args.boolSTHOSVD                  = TuckerKokkos::stringParse<bool>(fileAsString, "Perform STHOSVD", false);
  args.boolWriteSTHOSVD             = TuckerKokkos::stringParse<bool>(fileAsString, "Write core tensor and factor matrices", false);
  args.boolPrintOptions             = TuckerKokkos::stringParse<bool>(fileAsString, "Print options", false);
  args.boolWritePreprocessed        = TuckerKokkos::stringParse<bool>(fileAsString, "Write preprocessed data", false);
  args.boolUseLQ                    = TuckerKokkos::stringParse<bool>(fileAsString, "Compute SVD via LQ", false);

  args.tol                          = TuckerKokkos::stringParse<ScalarType>(fileAsString, "SV Threshold", 1e-6);
  args.stdThresh                    = TuckerKokkos::stringParse<ScalarType>(fileAsString, "STD Threshold", 1e-9);

  args.scaling_type                 = TuckerKokkos::stringParse<std::string>(fileAsString, "Scaling type", "None");
  args.sthosvd_dir                  = TuckerKokkos::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  args.sthosvd_fn                   = TuckerKokkos::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  args.sv_dir                       = TuckerKokkos::stringParse<std::string>(fileAsString, "SV directory", ".");
  args.sv_fn                        = TuckerKokkos::stringParse<std::string>(fileAsString, "SV file prefix", "sv");
  args.in_fns_file                  = TuckerKokkos::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");
  args.pre_fns_file                 = TuckerKokkos::stringParse<std::string>(fileAsString, "Preprocessed output file list", "pre.txt");
  args.stats_file                   = TuckerKokkos::stringParse<std::string>(fileAsString, "Stats file", "stats.txt");

  args.scale_mode                   = TuckerKokkos::stringParse<int>(fileAsString, "Scale mode", args.nd-1);

  return args;
}

/**
 * Assert that we either have automatic rank determination
 * or the user has supplied their own ranks
 */
template<class ScalarType>
int check_args(InputArgs<ScalarType> & args)
{
  return EXIT_SUCCESS;
}

template<class ScalarType>
void chech_array_sizes(const InputArgs<ScalarType> args)
{
  // if (!args.boolAuto && args.R_dims->size() != 0 && args.R_dims->size() != args.nd) {
  //   std::cerr << "Error: The size of the ranks array (" << R_dims->size();
  //   std::cerr << ") must be 0 or equal to the size of the global dimensions (" << nd << ")" << std::endl;
  //   return EXIT_FAILURE;
  // }
}

#endif // End of TUCKER_MPIKOKKOS_HELP_HPP
