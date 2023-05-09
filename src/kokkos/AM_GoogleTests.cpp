// #include "Tucker_SizeArray.hpp"

#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(HelloTest, BasicAssertions2) {
  // Expect two strings not to be equal.
  EXPECT_EQ(4, 4);
  // Expect equality.
  EXPECT_EQ(2, 3);
}

/*
namespace my {
namespace project {
namespace {

// The fixture
class SizeArrayTest : public ::testing::Test {
	protected:
		void SetUp() override { }

		void TearDown() override { }

		// Class members declared here can be used by all tests in the test suite
		Tucker::SizeArray sa_(4);
};

// Tests that the Foo::Bar() method does Abc.
TEST_F(SizeArrayTest, MethodSize) {
    Tucker::SizeArray sa(5);
    EXPECT_EQ(sa.size(), 5);
    EXPECT_EQ(sa_.size(), 4);
}

}  // namespace
}  // namespace project
}  // namespace my


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
*/