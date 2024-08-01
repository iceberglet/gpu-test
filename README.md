# jcuda-samples

This repository contains samples for the JCuda libraries.

**Note:** Some of the samples require third-party libraries, JCuda
libraries that are not part of the [`jcuda-main`](https://github.com/jcuda/jcuda-main) 
package (for example, [`JCudaVec`](https://github.com/jcuda/jcuda-vec) or 
[`JCudnn`](https://github.com/jcuda/jcudnn)), or utility libraries
that are not available in Maven Central. In order to compile these
samples, additional setup steps may be necessary. The main goal
of this repository is to collect and maintain the samples in a 
form that allows them to serve as a collection of snippets that
can easily be copied and pasted into own projects to get started.


1. install Nvidia CUDA toolkit https://developer.nvidia.com/cuda-toolkit-archive (12.0.0 for latest jcuda)
2. fix "cl.exe" path : C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64
   - requires gcc compiler and toolchain on linux
3. Since CUDA 11.0, macOS is not a supported environment for CUDA.


Info Links:
1. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=ptx
- nvcc compiles [device code] cu (in C++) into ptx files (CUDA instruction set architecture)
- Nvidia driver loads ptx via JIT compiler into [compute cache]
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
- 
2. ss 