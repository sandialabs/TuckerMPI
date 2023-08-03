#include "CmdLineParse.hpp"
#include "ParameterFileParser.hpp"

int main(int argc, char* argv[])
{
  const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						"--parameter-file", "paramfile.txt");
  const TuckerOnNode::InputParameters<double> inputs(paramfn);

  const auto globdims = inputs.dimensionsOfDataTensor();
  const std::vector<int> goldGlobDims = {3,5,7,11,1,1};
  if (globdims != goldGlobDims){
    std::puts("FAILED");
    return 0;
  }
  if (!inputs.boolAutoRankDetermination){
    std::puts("FAILED");
    return 0;
  }
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