#ifndef CMD_LINE_PARSER_HPP_
#define CMD_LINE_PARSER_HPP_

#include <string>

std::string parse_cmdline_or(const int argc,
			     const char* argv[],
			     const std::string& cl_arg,
			     const std::string& default_value);

#endif
