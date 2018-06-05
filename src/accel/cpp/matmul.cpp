#include <Eigen/Dense>
#include "matmul.hpp"

using Mat4x4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

Mat4x4 gemm4x4(Mat4x4 A, Mat4x4 B) {
  return A * B;
}
