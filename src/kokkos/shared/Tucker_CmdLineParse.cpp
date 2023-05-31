
#include "Tucker_CmdLineParse.hpp"

namespace Tucker{

std::string parse_cmdline_or(const int argc,
			     const char* argv[],
			     const std::string& cl_arg,
			     const std::string& default_value)
{
  int arg=1;
  std::string tmp;
  while (arg < argc) {
    if (cl_arg == std::string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc){ return ""; }
      // convert to string
      tmp = std::string(argv[arg]);
      // return tkr_real if everything is OK
      return tmp;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;
}

}