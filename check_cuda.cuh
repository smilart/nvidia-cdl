/*
 * Copyright (c) 2014 Smilart and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Alexander Komarov alexander07k@gmail.com - implementation.
 */

#ifndef CHECK_CUDA_CUH_
#define CHECK_CUDA_CUH_

#include "check.h"

#ifndef CPU
#endif

#ifdef checkCudaCall
#error checkCudaCall macro defined not only in check_cuda.cuh
#endif

#ifdef checkCuCall
#error checkCuCall macro defined not only in check_cuda.cuh
#endif

#define checkCudaCall(function, ...) \
{ \
	cudaError_t result = function; \
	if (result != cudaSuccess) { \
        std::string descriptionOfError = std::string("Description of error - ")  + std::string(cudaGetErrorString(result)) + std::string("."); \
       throwString(__FILE__, QUOTE_VALUE(__LINE__), #function, descriptionOfError , ##__VA_ARGS__); \
	} \
}

#define checkKernelRun(kernelNameAndGridDim, ...) \
{ \
	kernelNameAndGridDim, ##__VA_ARGS__; \
	cudaError_t result = cudaGetLastError(); \
	if (result != cudaSuccess) { \
		throwString(__FILE__, QUOTE_VALUE(__LINE__), extractKernelName(#kernelNameAndGridDim) + "<<<>>> start failure", cudaGetErrorString(result)); \
	} \
	result = cudaThreadSynchronize(); \
	if (result != cudaSuccess) { \
		throwString(__FILE__, QUOTE_VALUE(__LINE__), extractKernelName(#kernelNameAndGridDim) + "<<<>>> run failure", cudaGetErrorString(result)); \
	} \
}

#endif // CHECK_CUDA_CUH_