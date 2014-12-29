nvidia-cdl
==========

NVIDIA CUDA device list

Gathers device information and compiles a report with enumerated devices in an order consistent with CUDA API. Original `nvidia-smi` does not provide them in any particular order and this can be a problem on machines with multiple GPUs.

Provides the following information:
- Driver version
- GPU name
- GPU current temperature
- GPU UUID
- GPU PCI bus ID
- GPU memory usage

We also support data on actively running processes with command line arguments "--gpu=[number of CUDA device on which the process runs]" hoping that these processes run on the specified GPU.

To the staff of NVIDIA, thank you so very much for all of your help and expert assistance regarding this issue. 

Code is distributed under EPL and Public Domain, except `nvml.h` header which is the property of NVIDIA Corporation.

Building
--------

Install CUDA Toolkit 5.5 or later and run `make`.

The code is compiled statically to simplify distribution over a large number of machines.
