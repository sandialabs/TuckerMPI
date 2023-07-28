
#define TUCKER_HEADER_TEST_STRINGIZE_IMPL(x) #x
#define TUCKER_HEADER_TEST_STRINGIZE(x) TUCKER_HEADER_TEST_STRINGIZE_IMPL(x)

#define TUCKER_HEADER_TO_TEST				\
  TUCKER_HEADER_TEST_STRINGIZE(TUCKER_HEADER_TEST_NAME)

// include header twice to see if the include guards are set correctly
#include TUCKER_HEADER_TO_TEST
#include TUCKER_HEADER_TO_TEST

int main() { return 0; }
