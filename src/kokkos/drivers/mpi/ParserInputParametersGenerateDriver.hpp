#ifndef MPIKOKKOS_TUCKER_PARAM_FILE_PARSER_GENERATE_HPP_
#define MPIKOKKOS_TUCKER_PARAM_FILE_PARSER_GENERATE_HPP_

#include "ParameterFileParserUtils.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>

template<class ScalarType>
struct InputParametersGenerateDriver
{
  bool boolPrintOptions;
  std::vector<int> procGridDims_;
  std::vector<int> dataTensorDims_;
  std::vector<int> coreTensorDims_;
  unsigned int seed_;
  ScalarType epsilon_;

  // file which contains list of files to write into
  std::string out_fns_file;
  // outFilenames: contains each row read from out_fns_file
  std::vector<std::string> outDataFilenames;

public:
  InputParametersGenerateDriver(const std::string & paramFile, int mpiRank)
  {
    const auto fileAsStrings = Tucker::read_file_as_strings(paramFile);
    parse(fileAsStrings);
    if (mpiRank==0){
      check_args();
    }
  }

  void describe() const
  {
    if (boolPrintOptions) {
      std::cout << "File containing a list of filenames to output the constructed data into\n";
      std::cout << "- Output file list = " << out_fns_file << std::endl << std::endl;

      std::cout << "The global dimensions of the processor grid\n";
      std::cout << "- Grid dims = ";
      std::for_each(procGridDims_.cbegin(), procGridDims_.cend(), [=](int v){ std::cout << v << " "; });
      std::cout << std::endl;

      std::cout << "Global dimensions of the original tensor\n";
      std::cout << "- Global dims = ";
      std::for_each(dataTensorDims_.cbegin(), dataTensorDims_.cend(), [](auto s){ std::cout << s << " "; } );
      std::cout << std::endl;

      std::cout << "Global dimensions of the desired core tensor\n";
      std::cout << "- Ranks = ";
      std::for_each(coreTensorDims_.cbegin(), coreTensorDims_.cend(), [](auto s){ std::cout << s << " "; } );
      std::cout << std::endl;

      std::cout << "If true, print the parameters\n";
      std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

      std::cout << std::endl;
    }
  }

private:
  void parse(const std::vector<std::string>& fileAsStrings)
  {
    using namespace Tucker;
    boolPrintOptions = string_parse<bool>(fileAsStrings, "Print options", false);
    procGridDims_   = parse_multivalued_field<int>(fileAsStrings, "Grid dims");
    dataTensorDims_ = parse_multivalued_field<int>(fileAsStrings, "Global dims");
    coreTensorDims_ = parse_multivalued_field<int>(fileAsStrings, "Ranks");

    const auto defaultSeed = std::chrono::system_clock::now().time_since_epoch().count();
    seed_  = string_parse<unsigned int>(fileAsStrings, "RNG seed", defaultSeed);

    epsilon_ = string_parse<ScalarType>(fileAsStrings, "Noise", 1e-8);

    out_fns_file = string_parse<std::string>(fileAsStrings, "Output file list", "rec.txt");
    outDataFilenames = read_file_as_strings(out_fns_file);
  }

  void check_args(){
    std::cout << "Arguments checking: Starting" << std::endl;

    auto positive = [](auto v) -> bool{ return v > 0; };
    bool b1 = std::all_of(dataTensorDims_.cbegin(), dataTensorDims_.cend(), positive);
    if (!b1){
      std::cerr << "ERROR: Please enter strictly positive dimensions for the data tensor\n";
      std::abort();
    }

    bool b2 = std::all_of(coreTensorDims_.cbegin(), coreTensorDims_.cend(), positive);
    if (!b2){
      std::cerr << "ERROR: Please enter strictly positive dimensions for the core tensor\n";
      std::abort();
    }

    std::cout << "Arguments checking: Done" << std::endl;
  }
};

#endif
