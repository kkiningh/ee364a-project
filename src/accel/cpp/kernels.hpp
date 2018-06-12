#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
	#include "OpenCL/cl2.hpp"
#else
	#include "CL/cl2.hpp"
#endif

#include <Eigen/Dense>

#include <string>
#include <cstring>

#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <iterator>

cl::Platform getPlatform() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (!platforms.size()) {
    std::cout
      << "No platforms avalible"
      << std::endl;
    throw std::exception();
  }
  return platforms[0]; // Use first platform
}

cl::Device getDevice(cl::Platform platform, cl_device_type type = CL_DEVICE_TYPE_GPU) {
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (!devices.size()) {
    std::cout
      << "No devices available"
      << std::endl;
    throw std::exception();
  }
  return devices[0]; // Use first device
}

std::string queryDeviceName(cl::Device& device) {
  return device.getInfo<CL_DEVICE_NAME>();
}

cl::Program::Sources readKernelSource(const char* filename) {
  std::ifstream stream {filename};
  if (!stream.is_open()) {
    std::cout << "Cannot open file: " << filename << std::endl;
    throw std::exception();
  }

  std::string source {
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>()
  };

  return {source};
}

cl::Program createProgram(cl::Context context, cl::Device device, const char* filename) {
  // Create the program
  auto sources = readKernelSource(filename);
  auto program = cl::Program{context, sources};
  if (program.build({device}) != CL_SUCCESS) {
    std::cout
      << "Error building for device : "
      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
      << std::endl;
    throw std::exception();
  }

  return program;
}

class KernelRunner {
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> kernel;

public:
  KernelRunner(size_t count)
    : platform(getPlatform())
    , device(getDevice(platform))
    , context({device})
    , queue({context, device})
    , program(createProgram(context, device, "src/accel/kernels/kernel.cl"))
    , kernel(program, "run")
  { }

  Eigen::VectorXf multBlockDiag(
      Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, 2>> D,
      Eigen::Ref<const Eigen::VectorXf> x
  ) {
    auto bufferD = cl::Buffer{context, CL_MEM_READ_ONLY, D.size() * sizeof(float)};
    cl::copy(queue, D.data(), D.data() + D.size(), bufferD);

    return x;
  }

  Eigen::VectorXf run(
      Eigen::Ref<const Eigen::MatrixXf> L, // L matrix
      Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, 2>> D,  // D matrix
      const int K // iterations
  ) {
    // Output size
    auto count = L.size();

    // Allocate the buffers
    auto bufferL = cl::Buffer{context, CL_MEM_READ_ONLY, L.size() * sizeof(float)};
    auto bufferD = cl::Buffer{context, CL_MEM_READ_ONLY, D.size() * sizeof(float)};

    auto bufferW = cl::Buffer{context, CL_MEM_READ_WRITE, count * sizeof(float)};
    auto bufferY = cl::Buffer{context, CL_MEM_READ_WRITE, count * sizeof(float)};

    // Copy data from host to device
    cl::copy(queue, L.data(), L.data() + L.size(), bufferL);
    cl::copy(queue, D.data(), D.data() + D.size(), bufferD);

    // Run the kernel
    auto global = cl::NDRange(count);
    kernel(cl::EnqueueArgs(queue, global), bufferL, bufferD, bufferW, bufferY, K);

    // Copy result from device to host
    auto Wout = Eigen::VectorXf{count};
    cl::copy(queue, bufferW, Wout.data(), Wout.data() + Wout.size());

    return Wout;
  }
};

#endif
