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

  std::puts("PASSED");
  return 0;
}
