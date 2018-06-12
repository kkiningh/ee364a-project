Stantord EE364b Project (Spring 2018)
--
Kevin Kiningham

# Dependencies

These directions have only been tested on Ubuntu 16.04 with SDK version 18.0.

Download and install the [Intel FPGA SDK For OpenCL](https://www.altera.com/products/design-software/embedded-software-developers/opencl/overview.html).

Install [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
```bash
sudo apt install libeigen3-dev
```

Install OpenCL headers (note that you may have to install additional drivers depending on your GPU).
```bash
sudo apt install opencl-headers clinfo
```

Install [CMake](https://cmake.org/)
```bash
sudo apt get install cmake
```

Install [Pipenv](https://docs.pipenv.org/)
```bash
pip install pipenv --user
```

Install all python dependencies by running pipenv inside of the project directory
```bash
pipenv install
```

# Building

This project is distributed as a C++ python extension which can be built using the provided setup.py.
```bash
pipenv run python setup.py build
```

# Code examples

```python
import numpy as np
from accel import accel

L = np.load('KKT_Factorized_L.npz')
D = np.load('KKT_Factorized_D.npz')
runner = accel.Kernelrunner()

# Run the solver with a random initial trajectory for 35 iterations
x, w = runner.run(L, D, 35)
```
