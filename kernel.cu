#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

struct merge_sort_ctx_t
{
	uint32_t* d_src;
	uint32_t* d_dest;
	uint32_t* tmp;
	size_t elem_count;
	size_t size_bytes;
	size_t elems_per_chunk;
	size_t unmerged_chunks;
	size_t threads_per_block;
	size_t total_thread_blocks;
};

static int uint32_comp(const void* a, const void* b)
{
	if (*((uint32_t*)a) < *((uint32_t*)b)) return -1;
	else if (*((uint32_t*)a) == *((uint32_t*)b)) return 0;
	else return 1;
}

__device__ void print_array_device(uint32_t* arr, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		printf("%*u", 5, arr[i]);
	}
	printf("\n");
}

__host__ void print_array_host(uint32_t* arr, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		printf("%*u ", 10, arr[i]);
	}
	printf("\n");
}

__global__ void device_merge(merge_sort_ctx_t* ctx)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t* d_src = ctx->d_src;
	uint32_t* d_dest = ctx->d_dest;

	if (ctx->unmerged_chunks > thread_id)
	{
		size_t left = thread_id * ctx->elems_per_chunk;
		size_t right = left + ctx->elems_per_chunk;

		const size_t mid = left + ((right - left) / 2);
		size_t left_head = left;
		size_t right_head = mid;
		size_t write_head = left;

		// Keep going until either (left or right) side is exhausted.
		while (left_head < mid && right_head < right)
		{
			if (d_src[left_head] <= d_src[right_head])
			{
				d_dest[write_head] = d_src[left_head];
				++write_head;
				++left_head;
			}
			else
			{
				d_dest[write_head] = d_src[right_head];
				++write_head;
				++right_head;
			}
		}

		// Exhaust the left side if not exhausted already.
		while (left_head < mid)
		{
			d_dest[write_head] = d_src[left_head];
			++write_head;
			++left_head;
		}

		// Do the same to the right side.
		while (right_head < right)
		{
			d_dest[write_head] = d_src[right_head];
			++write_head;
			++right_head;
		}
	}
}

__host__ void host_merge(uint32_t* dest, const uint32_t* src, const size_t left, const size_t right)
{
	const size_t mid = left + ((right - left) / 2);
	size_t left_head = left;
	size_t right_head = mid;
	size_t write_head = left;

	for (; write_head < right; ++write_head)
	{
		if (left_head < mid && (right_head >= right || src[left_head] <= src[right_head]))
		{
			dest[write_head] = src[left_head];
			++left_head;
		}
		else
		{
			dest[write_head] = src[right_head];
			++right_head;
		}
	}
}

void check_gpu_err()
{
	auto err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << endl;
	}
}

int main() try
{
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;

	//const uint32_t N_ELEMENTS = 64;
	//const uint32_t N_ELEMENTS = 1048576;
	//const uint32_t N_ELEMENTS = 2097152;
	//const uint32_t N_ELEMENTS = 4194304;
	const uint32_t N_ELEMENTS = 268435456;
	//const uint32_t MAX_THREADS_PER_BLOCK = 16;
	const uint32_t MAX_THREADS_PER_BLOCK = 8; // Experimental
	const size_t HOST_MAX_NATIVE_THREADS = 8;

	cout << "Sorting an array of "
		<< N_ELEMENTS
		<< " 32-bit random integers"
		<< endl;

	// Initial context on the host.
	merge_sort_ctx_t* d_ctx;
	cudaMallocManaged(&d_ctx, sizeof(merge_sort_ctx_t));
	check_gpu_err();

	d_ctx->elem_count = N_ELEMENTS;
	d_ctx->size_bytes = d_ctx->elem_count * sizeof(uint32_t);
	d_ctx->unmerged_chunks = d_ctx->elem_count / 2;
	d_ctx->elems_per_chunk = 2;
	d_ctx->threads_per_block = MAX_THREADS_PER_BLOCK;
	d_ctx->total_thread_blocks = d_ctx->unmerged_chunks / d_ctx->threads_per_block;
	if (d_ctx->unmerged_chunks % d_ctx->threads_per_block != 0) ++d_ctx->total_thread_blocks;
	// Allocate buffers in GPU memory.
	cudaMalloc(&d_ctx->d_src, d_ctx->size_bytes);
	check_gpu_err();
	cudaMalloc(&d_ctx->d_dest, d_ctx->size_bytes);
	check_gpu_err();

	// Create the array to be sorted in host memory.
	uint32_t* h_src = new uint32_t[d_ctx->size_bytes];
	for (size_t i = 0; i < d_ctx->elem_count; ++i)
	{
		h_src[i] = (uint32_t)rand();
	}

	// Destination buffer.
	uint32_t* h_dest = new uint32_t[d_ctx->size_bytes];
	memset(h_dest, 0, d_ctx->size_bytes);

	// Temporary results buffer.
	uint32_t* h_tmpsrc = new uint32_t[d_ctx->size_bytes];
	memset(h_tmpsrc, 0, d_ctx->size_bytes);

	/////////////////////////////////////////////////////////////////////////////
	// CUDA_SORT
	/////////////////////////////////////////////////////////////////////////////

	begin = std::chrono::steady_clock::now();

	// Copy the array to be sorted into GPU memory.
	cudaMemcpy(d_ctx->d_src, h_src, d_ctx->size_bytes, cudaMemcpyHostToDevice);
	check_gpu_err();

	while (d_ctx->unmerged_chunks >= HOST_MAX_NATIVE_THREADS * 32 /* Found this value to be good */)
	{
		// Figure out the total number of thread blocks needed for the computation.
		d_ctx->total_thread_blocks = d_ctx->unmerged_chunks / d_ctx->threads_per_block;
		if (d_ctx->unmerged_chunks % d_ctx->threads_per_block != 0) ++d_ctx->total_thread_blocks;

		// Do a merge run.
		device_merge<<<d_ctx->total_thread_blocks, d_ctx->threads_per_block>>>(d_ctx);
		cudaDeviceSynchronize();
		check_gpu_err();

		d_ctx->elems_per_chunk *= 2;
		d_ctx->unmerged_chunks /= 2;

		// The temporary results in d_src aren't needed, so d_dest can be used as the source buffer
		// for the next run and the data in d_src overwritten.
		swap(d_ctx->d_src, d_ctx->d_dest);
	}

	// Unswap.
	swap(d_ctx->d_src, d_ctx->d_dest);

	auto gpu_end = std::chrono::steady_clock::now();

	// Copy the GPU results into host memory.
	cudaMemcpy(h_tmpsrc, d_ctx->d_dest, d_ctx->size_bytes, cudaMemcpyDeviceToHost);
	check_gpu_err();

	// Copy the context into host memory (for better performance).
	merge_sort_ctx_t h_ctx;
	cudaMemcpy(&h_ctx, d_ctx, sizeof(merge_sort_ctx_t), cudaMemcpyDeviceToHost);
	check_gpu_err();

	vector<thread*> threads;

	// Do the last runs on the CPU.
	while (h_ctx.unmerged_chunks != 0)
	{
		const size_t elems_per_thread = h_ctx.elem_count / h_ctx.unmerged_chunks;
		const size_t threads_per_batch = std::min(h_ctx.unmerged_chunks, HOST_MAX_NATIVE_THREADS);
		const size_t thread_batches = std::max((size_t)1ULL, h_ctx.unmerged_chunks / threads_per_batch);

		for (size_t j = 0; j < thread_batches; ++j)
		{
			for (size_t i = 0; i < threads_per_batch; ++i)
			{
				size_t left = (j * elems_per_thread * threads_per_batch) + (i * elems_per_thread);
				size_t right = left + elems_per_thread;
				threads.push_back(new thread(host_merge, h_dest, h_tmpsrc, left, right));
			}
			for (const auto& thread : threads)
			{
				thread->join();
				delete thread;
			}
			threads.clear();
		}

		h_ctx.elems_per_chunk *= 2;
		h_ctx.unmerged_chunks /= 2;

		swap(h_tmpsrc, h_dest);
	}

	// Unswap.
	swap(h_tmpsrc, h_dest);

	auto cpu_end = std::chrono::steady_clock::now();

	cout << setw(11)
		<< right
		<< "- cuda_sort"
		<< setw(6)
		<< right
		<< std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - begin).count()
		<< " ms"
		<< " (GPU: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - begin).count()
		<< " ms, CPU: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - gpu_end).count()
		<< " ms)"
		<< endl;

	for (size_t i = 0; i < h_ctx.elem_count; ++i)
	{
		if (i == 0) continue;

		if (h_dest[i] < h_dest[i - 1])
		{

			cout << "ERROR IN OUTPUT (first erroneous index = " << i << ", value = " << h_dest[i] << ")!" << endl;

			break;
		}
	}


	/////////////////////////////////////////////////////////////////////////////
	// QSORT
	/////////////////////////////////////////////////////////////////////////////


	uint32_t* src2 = new uint32_t[h_ctx.size_bytes];
	memcpy(src2, h_src, h_ctx.size_bytes);

	begin = std::chrono::steady_clock::now();
	qsort(src2, h_ctx.elem_count, sizeof(uint32_t), uint32_comp);
	end = std::chrono::steady_clock::now();
	cout << setw(11)
		<< right
		<< "- qsort    "
		<< setw(6)
		<< right
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms"
		<< endl;
	delete[] src2;


	/////////////////////////////////////////////////////////////////////////////
	// STD::SORT
	/////////////////////////////////////////////////////////////////////////////


	std::vector<uint32_t> src3;
	for (size_t i = 0; i < h_ctx.elem_count; ++i)
		src3.push_back(0);
	memcpy(src3.data(), h_src, h_ctx.size_bytes);

	begin = std::chrono::steady_clock::now();
	std::sort(src3.begin(), src3.end());
	end = std::chrono::steady_clock::now();
	cout << setw(11)
		<< right
		<< "- std::sort"
		<< setw(6)
		<< right
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms"
		<< endl;

	cudaFree(d_ctx->d_src);
	check_gpu_err();
	cudaFree(d_ctx->d_dest);
	check_gpu_err();
	cudaFree(d_ctx);
	check_gpu_err();
	delete[] h_src;
	delete[] h_dest;

	cout << endl;
	cout << "Press Enter to exit..." << endl;
	std::string s;
	std::getline(cin, s);

	return 0;
}
catch (std::bad_alloc& e)
{
	cerr << "Memory allocation failed. Aborting execution." << endl;
}
catch (std::exception& e)
{
	cerr << "An unexpected exception occurred. Aborting execution." << endl;
}
