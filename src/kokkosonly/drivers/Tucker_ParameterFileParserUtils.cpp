#include "Tucker_ParameterFileParserUtils.hpp"
#include <sstream>
#include <limits>
#include <cassert>

namespace TuckerKokkos{

SizeArray parse_size_array(const std::vector<std::string>& fileAsStrings,
			 const std::string& keyword)
{
  std::vector<int> tmp;
  int value;
  for (auto line : fileAsStrings) {
    // If the keyword is in the string then use that value
    if (line.find(keyword) != std::string::npos) {
      // Find the equal sign
      std::size_t equalPos = line.find("=");
      std::stringstream valueStream(line.substr(equalPos+1));

      // The value should be one "word", extract it from the string
      while(valueStream >> value) {
	tmp.push_back(value);
      }
    }
  }

  assert(tmp.size() <= std::numeric_limits<int>::max());
  if(tmp.size() == 0){ return {}; }
  SizeArray arr((int)tmp.size());
  for (int i = 0; i < (int)tmp.size(); i++) {
    arr[i] = tmp[i];
  }

  return arr; // Returns empty array if nothing is ever pushed onto tmp vector
}

std::vector<std::string> read_file_as_strings(const std::string& fileToRead)
{
  std::string line;
  std::vector<std::string> fileLines;
  std::ifstream myFile(fileToRead);

  if (!myFile) {
    std::cerr << "Error opening parameter file: "
	      << fileToRead << std::endl;
  }

  while (std::getline(myFile, line)) {
    // Lines starting with # are comments. The second part tests to see if the line is blank
    if (!(line.find_first_of("#") == 0) && line.find_first_not_of(" \t") != std::string::npos) {
      fileLines.push_back(line);
    }
  }

  return fileLines;
}

}
