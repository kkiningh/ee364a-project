#ifndef MATMUL_HPP_
#define MATMUL_HPP_

#include <Eigen/Dense>

using Mat4x4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

Mat4x4 gemm4x4(Mat4x4 A, Mat4x4 B);

#endif
