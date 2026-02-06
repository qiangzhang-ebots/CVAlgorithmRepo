#include<iostream>
#include "kArithmeticAlgorithmSample/ArithmeticAlgorithm.h" // 这里的路径取决于 kArithmeticAlgorithmSample/include/ 下的结构

int main(int argc, char ** argv)
{
  ArithmeticAlgorithm algo;
  int x=10,y=20;
  std::cout<<x<<"+"<<y<<"="<<algo.Add(x,y)<<std::endl;
  return 0;
}