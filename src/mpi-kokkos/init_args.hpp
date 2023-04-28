
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

  ScalarType tol                          ;
  ScalarType stdThresh                    ;

  Tucker::SizeArray* I_dims             ;
  // Tucker::SizeArray* R_dims = 0;
  // if(!boolAuto)  R_dims                 ;
  Tucker::SizeArray* proc_grid_dims     ;
  // Tucker::SizeArray* modeOrder          ;

  // std::string scaling_type              ;
  // std::string sthosvd_dir               ;
  // std::string sthosvd_fn                ;
  // std::string sv_dir                    ;
  // std::string sv_fn                     ;
  // std::string in_fns_file               ;
  // std::string pre_fns_file              ;
  // std::string reconstruct_report_file   ;
  // std::string stats_file                ;
  // std::string timing_file               ;
  // int nd = I_dims->size();
  // int scale_mode                        ;
};

template<class ScalarType>
void print_args(const InputArgs<ScalarType> & args)
{
#if 0  
  //
  // Print options
  //
  if (rank == 0 && boolPrintOptions) {
    std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
    std::cout << "- Global dims = " << *I_dims << std::endl << std::endl;

    std::cout << "The global dimensions of the processor grid\n";
    std::cout << "- Grid dims = " << *proc_grid_dims << std::endl << std::endl;

    std::cout << "Mode order for decomposition\n";
    std::cout << "- Decompose mode order " << *modeOrder << std::endl << std::endl;

    std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
    std::cout << "- Automatic rank determination = " << (boolAuto ? "true" : "false") << std::endl << std::endl;

    std::cout << "Used for automatic rank determination; the desired error rate\n";
    std::cout << "- SV Threshold = " << tol << std::endl << std::endl;

    if(!boolAuto) {
      std::cout << "Global dimensions of the desired core tensor\n";
      std::cout << "Not used if \"Automatic rank determination\" is enabled\n";
      std::cout << "- Ranks = " << *R_dims << std::endl << std::endl;
    }

    std::cout << "List of filenames of raw data to be read\n";
    std::cout << "- Input file list = " << in_fns_file << std::endl << std::endl;

    std::cout << "How to scale the tensor\n";
    std::cout << "- Scaling type = " << scaling_type << std::endl << std::endl;

    std::cout << "Which mode's hyperslices will be scaled\n";
    std::cout << "- Scale mode = " << scale_mode << std::endl << std::endl;

    std::cout << "Threshold for standard deviation before we simply set it to 1\n";
    std::cout << "Used in StandardCentering scaling\n";
    std::cout << "- STD Threshold = " << stdThresh << std::endl << std::endl;

    std::cout << "If true, perform ST-HOSVD\n";
    std::cout << "- Perform STHOSVD = " << (boolSTHOSVD ? "true" : "false") << std::endl << std::endl;

    std::cout << "If true, use the old Gram algorithm; otherwise use the new one\n";
    std::cout << "- Use old Gram = " << (boolUseOldGram ? "true" : "false") << std::endl << std::endl;

    std::cout << "Location of a report of the reconstruction errors \n";
    std::cout << "- Reconstruction report file = " << reconstruct_report_file << std::endl << std::endl;

    std::cout << "Location of statistics file containing min, max, mean, and std of each hyperslice\n";
    std::cout << "- Stats file = " << stats_file << std::endl << std::endl;

    std::cout << "If true, write the preprocessed data to a file\n";
    std::cout << "- Write preprocessed data = " << (boolWritePreprocessed ? "true" : "false") << std::endl << std::endl;

    std::cout << "File containing a list of filenames to output the scaled data into\n";
    std::cout << "- Preprocessed output file list = " << pre_fns_file << std::endl << std::endl;

    std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors\n";
    std::cout << "- Write STHOSVD result = " << (boolWriteSTHOSVD ? "true" : "false") << std::endl << std::endl;

    std::cout << "Directory location of ST-HOSVD output files\n";
    if(boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
    std::cout << "- STHOSVD directory = " << sthosvd_dir << std::endl << std::endl;

    std::cout << "Base name of ST-HOSVD output files\n";
    std::cout << "- STHOSVD file prefix = " << sthosvd_fn << std::endl << std::endl;

    std::cout << "Directory to place singular value files into\n";
    if(boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
    std::cout << "- SV directory = " << sv_dir << std::endl << std::endl;

    std::cout << "Base name for writing the singular value files\n";
    std::cout << "- SV file prefix = " << sv_fn << std::endl << std::endl;

    std::cout << "Name of the CSV file holding the timing results\n";
    std::cout << "- Timing file = " << timing_file << std::endl << std::endl;

    std::cout << "If true, reconstruct an approximation of the original tensor after ST-HOSVD\n";
    if(boolReconstruct) std::cout << "WARNING: This may require a great deal of memory\n";
    std::cout << "- Reconstruct tensor = " << (boolReconstruct ? "true" : "false") << std::endl << std::endl;

    std::cout << "If true, print the parameters\n";
    std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

    std::cout << std::endl;
  }
#endif  
}

template<class ScalarType>
InputArgs<ScalarType> parse_input_file(const std::vector<std::string> & fileAsString)
{
  InputArgs<ScalarType> args;
  
  // bool boolAuto                         = Tucker::stringParse<bool>(fileAsString, "Automatic rank determination", false);
  // bool boolSTHOSVD                      = Tucker::stringParse<bool>(fileAsString, "Perform STHOSVD", false);
  // bool boolWriteSTHOSVD                 = Tucker::stringParse<bool>(fileAsString, "Write core tensor and factor matrices", false);
  // bool boolPrintOptions                 = Tucker::stringParse<bool>(fileAsString, "Print options", false);
  // bool boolWritePreprocessed            = Tucker::stringParse<bool>(fileAsString, "Write preprocessed data", false);
  // bool boolUseOldGram                   = Tucker::stringParse<bool>(fileAsString, "Use old Gram", false);
  // bool boolUseLQ                        = Tucker::stringParse<bool>(fileAsString, "Compute SVD via LQ", false);
  // bool boolPrintSV                      = Tucker::stringParse<bool>(fileAsString, "Print factor matrices", false);
  // bool boolReconstruct                  = Tucker::stringParse<bool>(fileAsString, "Reconstruct tensor", false);
  // bool useButterflyTSQR                 = Tucker::stringParse<bool>(fileAsString, "Use butterfly TSQR", false);

  // ScalarType tol                          = Tucker::stringParse<ScalarType>(fileAsString, "SV Threshold", 1e-6);
  // ScalarType stdThresh                    = Tucker::stringParse<ScalarType>(fileAsString, "STD Threshold", 1e-9);

  // Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  // Tucker::SizeArray* R_dims = 0;
  // if(!boolAuto)  R_dims                 = Tucker::stringParseSizeArray(fileAsString, "Ranks");
  // Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");
  // Tucker::SizeArray* modeOrder          = Tucker::stringParseSizeArray(fileAsString, "Decompose mode order");

  // std::string scaling_type              = Tucker::stringParse<std::string>(fileAsString, "Scaling type", "None");
  // std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  // std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  // std::string sv_dir                    = Tucker::stringParse<std::string>(fileAsString, "SV directory", ".");
  // std::string sv_fn                     = Tucker::stringParse<std::string>(fileAsString, "SV file prefix", "sv");
  // std::string in_fns_file               = Tucker::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");
  // std::string pre_fns_file              = Tucker::stringParse<std::string>(fileAsString, "Preprocessed output file list", "pre.txt");
  // std::string reconstruct_report_file   = Tucker::stringParse<std::string>(fileAsString, "Reconstruction report file", "reconstruction.txt");
  // std::string stats_file                = Tucker::stringParse<std::string>(fileAsString, "Stats file", "stats.txt");
  // std::string timing_file               = Tucker::stringParse<std::string>(fileAsString, "Timing file", "runtime.csv");

  // int nd = I_dims->size();
  // int scale_mode                        = Tucker::stringParse<int>(fileAsString, "Scale mode", nd-1);
  
  return args;
}

void check_args()
{
#if 0  
  // Assert that we either have automatic rank determination or the user
  // has supplied their own ranks
  //
  if(!boolAuto && !R_dims) {
    std::cerr << "ERROR: Please either enable Automatic rank determination, "
              << "or provide the desired core tensor size via the Ranks parameter\n";
    return EXIT_FAILURE;
  }

  if(tol >= 1) {
    std::cerr << "ERROR: The reconstruction error tolerance should be smaller than 1. \n";
    return EXIT_FAILURE;
  }

  if(!modeOrder){
    modeOrder = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(nd);
    for(int i=0; i<nd; i++){
      modeOrder->data()[i] = i;
      std::cout <<"modeOrder[" <<i<<"]: " << modeOrder->data()[i];
    }
    std::cout << std::endl;
  }
#endif
}


#endif

