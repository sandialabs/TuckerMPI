#ifndef MPIKOKKOS_TUCKER_PARAM_FILE_PARSER_HPP_
#define MPIKOKKOS_TUCKER_PARAM_FILE_PARSER_HPP_

#include "ParameterFileParserUtils.hpp"
#include <fstream>
#include <iomanip>
#include <vector>
#include <optional>
#include <algorithm>

namespace TuckerMpiDistributed{

template<class ScalarType>
struct InputParameters
{
  int nd;
  bool boolAutoRankDetermination;
  bool boolSTHOSVD;
  bool boolWriteSTHOSVD;
  bool boolPrintOptions;
  bool boolWriteTensorAfterPreprocessing;
  bool boolPrintSV;
  bool boolReconstruct;
  ScalarType tol;
  ScalarType stdThresh;

  std::vector<int> proc_grid_dims;
  std::vector<int> modeOrder;

  std::string scaling_type;
  std::string sthosvd_dir;
  std::string sthosvd_fn;
  std::string sv_dir;
  std::string sv_fn;

  // is the file which inside contains list of files to read from,
  // see raw.txt inside the driver tests
  std::string in_fns_file;
  // rawDataFilenames: contains each row read from in_fns_file
  std::vector<std::string> rawDataFilenames;

  // is the file which inside contains list of files to *write* to,
  // see pre.txt inside the driver tests
  std::string preproc_fns_file;
  // rawDataFilenames: contains each row read from in_fns_file
  std::vector<std::string> preprocDataFilenames;


  std::string stats_file;
  std::string timing_file;
  int scale_mode;

private:
  std::vector<int> dataTensorDims_;
  std::optional<std::vector<int>> coreTensorDims_;

public:
  InputParameters(const std::string & paramFile)
  {
    const auto fileAsStrings = Tucker::read_file_as_strings(paramFile);
    parse(fileAsStrings);
  }

  auto const & dimensionsOfDataTensor() const { return dataTensorDims_; }
  auto const & dimensionsOfCoreTensor() const { return coreTensorDims_; }

  void describe() const
  {
    if (boolPrintOptions) {
      std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
      std::cout << "- Global dims = ";
      std::for_each(dataTensorDims_.cbegin(), dataTensorDims_.cend(),
		    [=](int v){ std::cout << v << " "; });
      std::cout << std::endl;

      std::cout << "The global dimensions of the processor grid\n";
      std::cout << "- Grid dims = ";
      std::for_each(proc_grid_dims.cbegin(), proc_grid_dims.cend(),
		    [=](int v){ std::cout << v << " "; });
      std::cout << std::endl;

      std::cout << "Mode order for decomposition\n";
      std::cout << "- Decompose mode order ";
      std::for_each(modeOrder.cbegin(), modeOrder.cend(),
		    [=](int v){ std::cout << v << " "; });
      std::cout << std::endl;

      std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
      std::cout << "- Automatic rank determination = "
		<< (boolAutoRankDetermination ? "true" : "false") <<  std::endl;

      std::cout << "Used for automatic rank determination; the desired error rate\n";
      std::cout << "- SV Threshold = " << tol <<  std::endl;

      if(!boolAutoRankDetermination) {
        std::cout << "Global dimensions of the desired core tensor\n";
        std::cout << "Not used if \"Automatic rank determination\" is enabled\n";
        std::cout << "- Ranks = ";
        const auto & vec = coreTensorDims_.value();
        std::for_each(vec.cbegin(), vec.cend(),
		      [=](int v){ std::cout << v << " "; });
	std::cout << std::endl;
      }

      std::cout << "List of filenames of raw data to be read\n";
      std::cout << "- Input file list = " << in_fns_file <<  std::endl;

      std::cout << "How to scale the tensor\n";
      std::cout << "- Scaling type = " << scaling_type <<  std::endl;

      std::cout << "Which mode's hyperslices will be scaled\n";
      std::cout << "- Scale mode = " << scale_mode <<  std::endl;

      std::cout << "Threshold for standard deviation before we simply set it to 1\n";
      std::cout << "Used in StandardCentering scaling\n";
      std::cout << "- STD Threshold = " << stdThresh <<  std::endl;

      std::cout << "If true, perform ST-HOSVD\n";
      std::cout << "- Perform STHOSVD = " << (boolSTHOSVD ? "true" : "false") <<  std::endl;

      std::cout << "Location of statistics file containing min, max, mean, and std of each hyperslice\n";
      std::cout << "- Stats file = " << stats_file <<  std::endl;

      std::cout << "If true, write the preprocessed data to a file\n";
      std::cout << "- Write preprocessed data = " << (boolWriteTensorAfterPreprocessing ? "true" : "false") <<  std::endl;

      std::cout << "File containing a list of filenames to output the scaled data into\n";
      std::cout << "- Preprocessed output file list = " << preproc_fns_file <<  std::endl;

      std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors)\n";
      std::cout << "- Write STHOSVD result = " << (boolWriteSTHOSVD ? "true" : "false") <<  std::endl;

      std::cout << "Directory location of ST-HOSVD output files\n";
      if(boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
      std::cout << "- STHOSVD directory = " << sthosvd_dir <<  std::endl;

      std::cout << "Base name of ST-HOSVD output files\n";
      std::cout << "- STHOSVD file prefix = " << sthosvd_fn <<  std::endl;

      std::cout << "Directory to place singular value files into\n";
      if(boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
      std::cout << "- SV directory = " << sv_dir <<  std::endl;

      std::cout << "Base name for writing the singular value files\n";
      std::cout << "- SV file prefix = " << sv_fn <<  std::endl;

      std::cout << "Name of the CSV file holding the timing results\n";
      std::cout << "- Timing file = " << timing_file <<  std::endl;

      std::cout << "If true, reconstruct an approximation of the original tensor after ST-HOSVD\n";
      if(boolReconstruct) std::cout << "WARNING: This may require a great deal of memory\n";
      std::cout << "- Reconstruct tensor = " << (boolReconstruct ? "true" : "false") <<  std::endl;

      std::cout << "If true, print the parameters\n";
      std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl;

      std::cout << std::endl;
    }
  }

private:
  void parse(const std::vector<std::string>& fileAsStrings)
  {
    using namespace Tucker;
    dataTensorDims_ = parse_multivalued_field<int>(fileAsStrings, "Global dims");
    nd = dataTensorDims_.size();

    boolAutoRankDetermination = string_parse<bool>(fileAsStrings, "Automatic rank determination", false);
    if (!boolAutoRankDetermination) {
      coreTensorDims_ = parse_multivalued_field<int>(fileAsStrings, "Ranks");
      std::cout << "Global dimensions of the core tensor is fixed:\n";
      const auto & vec = coreTensorDims_.value();
      std::cout << "- Ranks = ";
      std::for_each(vec.cbegin(), vec.cend(), [=](int v){ std::cout << v << " "; });
    }

    boolSTHOSVD             = string_parse<bool>(fileAsStrings, "Perform STHOSVD", false);
    boolWriteSTHOSVD        = string_parse<bool>(fileAsStrings, "Write core tensor and factor matrices", false);
    boolPrintOptions        = string_parse<bool>(fileAsStrings, "Print options", false);
    boolWriteTensorAfterPreprocessing = string_parse<bool>(fileAsStrings, "Write preprocessed data", false);
    boolPrintSV             = string_parse<bool>(fileAsStrings, "Print factor matrices", false);
    boolReconstruct         = string_parse<bool>(fileAsStrings, "Reconstruct tensor", false);
    tol                     = string_parse<ScalarType>(fileAsStrings, "SV Threshold", 1e-6);
    stdThresh               = string_parse<ScalarType>(fileAsStrings, "STD Threshold", 1e-9);
    proc_grid_dims          = parse_multivalued_field<int>(fileAsStrings, "Grid dims");
    modeOrder               = parse_multivalued_field<int>(fileAsStrings, "Decompose mode order");
    scaling_type            = string_parse<std::string>(fileAsStrings, "Scaling type", "None");
    sthosvd_dir             = string_parse<std::string>(fileAsStrings, "STHOSVD directory", "compressed");
    sthosvd_fn              = string_parse<std::string>(fileAsStrings, "STHOSVD file prefix", "sthosvd");
    sv_dir                  = string_parse<std::string>(fileAsStrings, "SV directory", ".");
    sv_fn                   = string_parse<std::string>(fileAsStrings, "SV file prefix", "sv");

    in_fns_file             = string_parse<std::string>(fileAsStrings, "Input file list", "raw.txt");
    rawDataFilenames        = read_file_as_strings(in_fns_file);

    preproc_fns_file        = string_parse<std::string>(fileAsStrings, "Preprocessed output file list", "pre.txt");
    preprocDataFilenames    = read_file_as_strings(preproc_fns_file);

    stats_file              = string_parse<std::string>(fileAsStrings, "Stats file", "stats.txt");
    timing_file             = string_parse<std::string>(fileAsStrings, "Timing file", "runtime.csv");
    scale_mode              = string_parse<int>(fileAsStrings, "Scale mode", nd-1);
  }

  int check_args(){
    if(!boolAutoRankDetermination && !coreTensorDims_) {
      std::cerr << "ERROR: Please either enable Automatic rank determination, "
                << "or provide the desired core tensor size via the Ranks parameter\n";
      return EXIT_FAILURE;
    }

    if(tol >= 1) {
      std::cerr << "ERROR: The reconstruction error tolerance should be smaller than 1. \n";
      return EXIT_FAILURE;
    }

    if(!modeOrder.data()){
      modeOrder = std::vector<int>(nd);
      for(int i=0; i<nd; i++){
        modeOrder.data()[i] = i;
        std::cout <<"modeOrder[" <<i<<"]: " << modeOrder.data()[i];
      }
      std::cout << std::endl;
    }

    return EXIT_SUCCESS;
  }
};

}
#endif
