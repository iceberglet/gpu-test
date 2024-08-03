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
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

Profiling:
- nvvp/nvprof doesn't support CC8.0+ (4080SUPER is 8.9...)
- nvvp -vm [java_bin_path]  (e.g. ~/.jdks/temurin-1.8.0_422/bin)
  C:\Users\drago\.jdks\openjdk-22\bin\java.exe 
-XX:+UseZGC "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2024.1\lib\idea_rt.jar=2646:C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2024.1\bin"-Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8 -classpath D:\Dev\jcuda-samples\target\classes;C:\Users\drago\.m2\repository\org\apache\commons\commons-math3\3.6.1\commons-math3-3.6.1.jar;C:\Users\drago\.m2\repository\org\jcuda\jcuda\12.0.0\jcuda-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcuda-natives\12.0.0\jcuda-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcublas\12.0.0\jcublas-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcublas-natives\12.0.0\jcublas-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcufft\12.0.0\jcufft-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcufft-natives\12.0.0\jcufft-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcurand\12.0.0\jcurand-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcurand-natives\12.0.0\jcurand-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcusparse\12.0.0\jcusparse-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcusparse-natives\12.0.0\jcusparse-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcusolver\12.0.0\jcusolver-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcusolver-natives\12.0.0\jcusolver-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcudnn\12.0.0\jcudnn-12.0.0.jar;C:\Users\drago\.m2\repository\org\jcuda\jcudnn-natives\12.0.0\jcudnn-natives-12.0.0-windows-x86_64.jar;C:\Users\drago\.m2\repository\org\jcuda\jcuda-vec\0.0.2\jcuda-vec-0.0.2.jar;C:\Users\drago\.m2\repository\de\javagl\matrixmarketreader\0.0.1-SNAPSHOT\matrixmarketreader-0.0.1-SNAPSHOT.jar;C:\Users\drago\.m2\repository\org\jcuda\jcuda-matrix-utils\0.0.1-SNAPSHOT\jcuda-matrix-utils-0.0.1-SNAPSHOT.jar;C:\Users\drago\.m2\repository\com\github\wendykierp\JTransforms\3.1\JTransforms-3.1-with-dependencies.jar;C:\Users\drago\.m2\repository\pl\edu\icm\JLargeArrays\1.5\JLargeArrays-1.5.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt-main\2.3.2\gluegen-rt-main-2.3.2.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-android-aarch64.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-android-armv6.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-linux-amd64.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-linux-armv6.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-linux-armv6hf.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-linux-i586.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-macosx-universal.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-solaris-amd64.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-solaris-i586.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-windows-amd64.jar;C:\Users\drago\.m2\repository\org\jogamp\gluegen\gluegen-rt\2.3.2\gluegen-rt-2.3.2-natives-windows-i586.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all-main\2.3.2\jogl-all-main-2.3.2.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-android-aarch64.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-android-armv6.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-linux-amd64.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-linux-armv6.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-linux-armv6hf.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-linux-i586.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-macosx-universal.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-solaris-amd64.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-solaris-i586.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-windows-amd64.jar;C:\Users\drago\.m2\repository\org\jogamp\jogl\jogl-all\2.3.2\jogl-all-2.3.2-natives-windows-i586.jar;C:\Users\drago\.m2\repository\org\lwjgl\lwjgl\3.3.4\lwjgl-3.3.4.jar;C:\Users\drago\.m2\repository\org\lwjgl\lwjgl-cuda\3.3.4\lwjgl-cuda-3.3.4.jar;C:\Users\drago\.m2\repository\org\lwjgl\lwjgl-opengl\3.3.4\lwjgl-opengl-3.3.4.jar;C:\Users\drago\.m2\repository\org\lwjgl\lwjgl\3.3.4\lwjgl-3.3.4-natives-windows.jar;C:\Users\drago\.m2\repository\org\lwjgl\lwjgl-opengl\3.3.4\lwjgl-opengl-3.3.4-natives-windows.jar benchmark.BenchMarkingTest

- NSight:
- https://developer.nvidia.com/nsight-systems/get-started#platforms





Glossary & Concepts
       MultiProcessor
           |
Grid > Block/CTA > Warp(32)
- Grid
- Block, ThreadBlock, Cooperative Thread Array (CTA) runs concurrently on the same SM, with fast shared memory and barriers
  - Independent from one another
- Warps: every 32 threads on a CTA

- SM/Multiprocessor - Streaming multiprocessor - executes multiple CTA at the same time
- Cuda core
- Wave: total number of CTAs which can run concurrently
- Theoretical Max Warps per SM -> 48 means an SM can at most process 48 * 32 =~1500 threads concurrently

- CTA Occupancy: No. of CTA per SM, limited by the number of threads (block size), shared memory and hardware barriers
- Register Pressure: 
    - regsPerBlock=65536 is the maximum number of 32-bit registers available to a thread block; this number is 
    - shared by all thread blocks simultaneously resident on a multiprocessor;
    - (i.e. if each thread takes 20 doubles -> 20 * 64bits)


Principles:
1. The number of blocks in a grid should be larger than the number of multiprocessors (80) so that all multiprocessors have at least one block to execute.
   - we want the contrary! must not take all of them, else there will be stuff waiting
2. Furthermore, there should be multiple active blocks per multiprocessor so that blocks that aren’t waiting for a __syncthreads() can keep the hardware busy
3. A lower occupancy kernel will have more registers available per thread than a higher occupancy kernel, which may result in less register spilling to local memory

Threads per block should be a multiple of warp size to avoid wasting computation on under-populated warps and to facilitate coalescing.
A minimum of 64 threads per block should be used, and only if there are multiple concurrent blocks per multiprocessor.
Between 128 and 256 threads per block is a good initial range for experimentation with different block sizes.
Use several smaller thread blocks rather than one large thread block per multiprocessor if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads().



By default, the nvcc compiler generates IEEE-compliant code, but it also provides options to generate code that somewhat less accurate but faster:
-ftz=true (denormalized numbers are flushed to zero)
-prec-div=false (less precise division)
-prec-sqrt=false (less precise square root)
Another, more aggressive, option is -use_fast_math, which coerces every functionName() call to the equivalent __functionName() call. This makes the code run faster at the cost of diminished precision and accuracy. See Math Libraries.


--IMPORTANT!!!!--
Flow control instructions (if, switch, do, for, while) can significantly affect the instruction throughput by causing threads of 
the same warp to diverge; that is, to follow different execution paths. If this happens, the different execution paths must be executed separately; this increases the total number of instructions executed for this warp.
To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps.