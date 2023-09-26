#ifndef TUCKER_KOKKOSONLY_PARAM_FILE_PARSER_UTILS_HPP_
#define TUCKER_KOKKOSONLY_PARAM_FILE_PARSER_UTILS_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

namespace Tucker{

/** \brief Parses a single option
 *
 * \param[in] lines Vector of strings; each string represents a single option defined by the user
 * \param[in] keyword Option to be parsed
 * \param[in] default_value Default value of the option
 */
template<typename T>
T string_parse(const std::vector<std::string>& lines,
	      const std::string& keyword,
	      const T& default_value)
{
  T value = default_value;
  for (auto line : lines) {
    // If the keyword is in the string then use that value
    if (line.find(keyword) != std::string::npos) {
      // Find the equal sign
      std::size_t equalPos = line.find("=");
      // Extract the string after that equal sign
      std::string valueSubstring = line.substr(equalPos+1);

      // This is explicitly for bool arguments:
      // In both, the second clause makes sure that filenames with "true" or "false" in them
      // are not replaced by 0 or 1
      if (valueSubstring.find("true") != std::string::npos &&
          valueSubstring.find_first_not_of("true \t") == std::string::npos) {

        valueSubstring = "1";
      } else if (valueSubstring.find("false") != std::string::npos &&
          valueSubstring.find_first_not_of("true \t") == std::string::npos) {

        valueSubstring = "0";
      }

      std::stringstream valueStream(valueSubstring);
      // The value should be one "word", extract it from the string
      valueStream >> value;
      break;
    }
  }

  return value;
}

std::vector<std::string> read_file_as_strings(const std::string& fileToRead);

template<class T>
std::vector<T> parse_multivalued_field(const std::vector<std::string>& fileAsStrings, const std::string& keyword)
{
  std::vector<T> result;
  T value;
  for (auto line : fileAsStrings) {
    // If the keyword is in the string then use that value
    if (line.find(keyword) != std::string::npos) {
      // Find the equal sign
      std::size_t equalPos = line.find("=");
      std::stringstream valueStream(line.substr(equalPos+1));

      // The value should be one "word", extract it from the string
      while(valueStream >> value) { result.push_back(value); }
    }
  }

  return result;
}

}
#endif
