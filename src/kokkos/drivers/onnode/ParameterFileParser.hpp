#ifndef TUCKER_KOKKOSONLY_PARAM_FILE_PARSER_HPP_
#define TUCKER_KOKKOSONLY_PARAM_FILE_PARSER_HPP_

#include "ParameterFileParserUtils.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <optional>

namespace TuckerOnNode{

template<class ScalarType>
struct InputParameters
{
  int nd;
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
  int scale_mode;

private:
  std::vector<int> dataTensorDims_;
  std::optional<std::vector<int>> coreTensorDims_;

public:
  InputParameters(const std::string & paramFile)
  {
    const auto fileAsStrings = Tucker::read_file_as_strings(paramFile);
    std::cout << fileAsStrings.size() << " " << fileAsStrings[0] << std::endl;
    parse(fileAsStrings);
  }

  auto const & dimensionsOfDataTensor() const { return dataTensorDims_; }
  auto const & dimensionsOfCoreTensor() const { return coreTensorDims_; }

  void describe() const
  {
    if (boolPrintOptions) {
      std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
      std::cout << "- Automatic rank determination = " << (boolAuto ? "true" : "false") << std::endl << std::endl;

      std::cout << "Used for automatic rank determination; the desired error rate\n";
      std::cout << "- SV Threshold = " << tol << std::endl << std::endl;

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

      std::cout << "Location of statistics file containing min, max, mean, and std of each hyperslice\n";
      std::cout << "- Stats file = " << stats_file << std::endl << std::endl;

      std::cout << "If true, write the preprocessed data to a file\n";
      std::cout << "- Write preprocessed data = " << (boolWritePreprocessed ? "true" : "false") << std::endl << std::endl;

      std::cout << "File containing a list of filenames to output the scaled data into\n";
      std::cout << "- Preprocessed output file list = " << pre_fns_file << std::endl << std::endl;

      std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors)\n";
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

      std::cout << "If true, print the parameters\n";
      std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

      std::cout << std::endl;
    }
  }

private:
  void parse(const std::vector<std::string>& fileAsStrings)
  {
    using namespace Tucker;
    dataTensorDims_ = parse_multivalued_field<int>(fileAsStrings, "Global dims");
    nd = dataTensorDims_.size();

    boolAuto = string_parse<bool>(fileAsStrings, "Automatic rank determination", false);
    if (!boolAuto) {
      coreTensorDims_ = parse_multivalued_field<int>(fileAsStrings, "Ranks");
      std::cout << "Global dimensions of the core tensor is fixed:\n";

      const auto & vec = coreTensorDims_.value();
      std::cout << "- Ranks = ";
      std::for_each(vec.cbegin(), vec.cend(), [=](int v){ std::cout << v << " "; });
    }

    boolSTHOSVD           = string_parse<bool>(fileAsStrings, "Perform STHOSVD", false);
    boolWriteSTHOSVD      = string_parse<bool>(fileAsStrings, "Write core tensor and factor matrices", false);
    boolPrintOptions      = string_parse<bool>(fileAsStrings, "Print options", false);
    boolWritePreprocessed = string_parse<bool>(fileAsStrings, "Write preprocessed data", false);
    boolUseLQ             = string_parse<bool>(fileAsStrings, "Compute SVD via LQ", false);
    tol                   = string_parse<ScalarType>(fileAsStrings, "SV Threshold", 1e-6);
    stdThresh             = string_parse<ScalarType>(fileAsStrings, "STD Threshold", 1e-9);
    scaling_type          = string_parse<std::string>(fileAsStrings, "Scaling type", "None");
    sthosvd_dir           = string_parse<std::string>(fileAsStrings, "STHOSVD directory", "compressed");
    sthosvd_fn            = string_parse<std::string>(fileAsStrings, "STHOSVD file prefix", "sthosvd");
    sv_dir                = string_parse<std::string>(fileAsStrings, "SV directory", ".");
    sv_fn                 = string_parse<std::string>(fileAsStrings, "SV file prefix", "sv");
    in_fns_file           = string_parse<std::string>(fileAsStrings, "Input file list", "raw.txt");
    pre_fns_file          = string_parse<std::string>(fileAsStrings, "Preprocessed output file list", "pre.txt");
    stats_file            = string_parse<std::string>(fileAsStrings, "Stats file", "stats.txt");
    scale_mode            = string_parse<int>(fileAsStrings, "Scale mode", nd-1);
  }

  void check_args(){
    std::cout << "Argument checking: MISSING IMPL " << std::endl;
    std::abort();
  }
};

} // end namespace TuckerOnNode
#endif
