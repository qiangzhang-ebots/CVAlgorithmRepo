#include <iostream>

#include "ArithmeticAlgorithmSample/ArithmeticAlgorithm.h"

int main(int argc, char** argv) {
  ArithmeticAlgorithm algo;
  int x = 10, y = 20;
  std::cout << x << "+" << y << "=" << algo.Add(x, y) << std::endl;
  return 0;
}
