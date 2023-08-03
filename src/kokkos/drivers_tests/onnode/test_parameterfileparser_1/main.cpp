#include "CmdLineParse.hpp"
#include "ParameterFileParser.hpp"

int main(int argc, char* argv[])
{
  #ifdef PARAM1
  std::string filename = "paramfile_1.txt";
  #else
  std::string filename = "paramfile_2.txt";
  #endif
  const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						"--parameter-file", filename);
  const TuckerOnNode::InputParameters<double> inputs(paramfn);

  const auto globdims = inputs.dimensionsOfDataTensor();
  const std::vector<int> goldGlobDims = {3,5,7,11,1,1};
  if (globdims != goldGlobDims){
    std::puts("FAILED");
    return 0;
  }

  const auto ranks_optional = inputs.dimensionsOfCoreTensor();
  #ifdef PARAM1
  if (ranks_optional){
    std::puts("FAILED");
    return 0;
  }
  if (!inputs.boolAutoRankDetermination){
    std::puts("FAILED");
    return 0;
  }
  #else
  if (!ranks_optional){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.boolAutoRankDetermination){
    std::puts("FAILED");
    return 0;
  }
  const std::vector<int> goldRanks = {2,2,2,2,1,1};
  if (ranks_optional.value() != goldRanks){
    std::puts("FAILED");
    return 0;
  }
  #endif

  if ( (inputs.tol - 0.123456) > 1e-10 ){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.in_fns_file.compare("myraw.txt") != 0){
    std::puts("FAILED");
    return 0;
  }
  if (!inputs.boolSTHOSVD){
    std::puts("FAILED");
    return 0;
  }
  if(!inputs.boolWriteResultsOfSTHOSVD){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.scaling_type.compare("Somestring") != 0){
    std::puts("FAILED");
    return 0;
  }
  if ( (inputs.scale_mode - 123) > 1e-10 ){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.sthosvd_dir.compare("mycompressed") != 0){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.sthosvd_fn.compare("sthosvd_myprefix") != 0){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.sv_dir.compare("./somedir") != 0){
    std::puts("FAILED");
    return 0;
  }
  if (inputs.sv_fn.compare("mysvprefix") != 0){
    std::puts("FAILED");
    return 0;
  }
  if (!inputs.boolPrintOptions){
    std::puts("FAILED");
    return 0;
  }
  std::puts("PASSED");
  return 0;
}