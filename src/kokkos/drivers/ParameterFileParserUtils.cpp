
#include "ParameterFileParserUtils.hpp"
#include <sstream>
#include <limits>
#include <cassert>

namespace Tucker{

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

}// end namespace Tucker
