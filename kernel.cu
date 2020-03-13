#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
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
	size_t run_size;
	size_t running_threads;
	size_t threads_per_block;
	size_t total_blocks;
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

__global__ void print_array_global(uint32_t* arr, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		printf("%*u", 5, arr[i]);
	}
	printf("\n");
}

void __host__ print_array_host(uint32_t* arr, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		printf("%*u ", 10, arr[i]);
	}
	printf("\n");
}

__device__ void merge_in_thread(uint32_t* dest, const uint32_t* src, const size_t left, const size_t right)
{
	const size_t mid = left + ((right - left) / 2);
	size_t left_head = left;
	size_t right_head = mid;
	size_t write_head = left;

	for (; write_head < right; ++write_head)
	{
		if (left_head < mid && (src[left_head] < src[right_head] || right_head >= right))
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

	/*
	while (left_head < mid && right_head < right)
	{
		if (src[left_head] <= src[right_head])
		{
			dest[write_head] = src[left_head];
			++write_head;
			++left_head;
		}
		else
		{
			dest[write_head] = src[right_head];
			++write_head;
			++right_head;
		}
	}

	while (left_head < mid)
	{
		dest[write_head] = src[left_head];
		++write_head;
		++left_head;
	}

	while (right_head < right)
	{
		dest[write_head] = src[right_head];
		++write_head;
		++right_head;
	}
	*/
}

__global__ void device_merge(merge_sort_ctx_t* ctx)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (ctx->running_threads > thread_id)
	{
		size_t left = thread_id * (ctx->run_size * 2);
		size_t right = left + (ctx->run_size * 2);
		const size_t mid = left + ((right - left) / 2);
		size_t left_head = left;
		size_t right_head = mid;
		size_t write_head = left;

		for (; write_head < right; ++write_head)
		{
			if (left_head < mid && (right_head >= right || ctx->d_src[left_head] <= ctx->d_src[right_head]))
			{
				ctx->d_dest[write_head] = ctx->d_src[left_head];
				++left_head;
			}
			else
			{
				ctx->d_dest[write_head] = ctx->d_src[right_head];
				++right_head;
			}
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

__global__ void device_merge_optimized(merge_sort_ctx_t* ctx)
{
	extern __shared__ uint8_t shared[];

	const int block_first_thread_id = blockIdx.x * blockDim.x;
	const int thread_id = block_first_thread_id + threadIdx.x;

	//	merge_sort_ctx_t* const shared_ctx = (merge_sort_ctx_t*)shared;
	merge_sort_ctx_t* const shared_ctx = ctx;

	if (threadIdx.x == 0)
	{
		memcpy(shared_ctx, ctx, sizeof(merge_sort_ctx_t));
	}

	__syncthreads();

	const size_t shared_chunk_size = shared_ctx->run_size * 2 * shared_ctx->running_threads % blockDim.x;
	const size_t shared_chunk_size_bytes = shared_chunk_size * sizeof(uint32_t);
	const size_t chunk_begin_offset = block_first_thread_id * shared_chunk_size;

	uint32_t* const global_src = &shared_ctx->d_src[chunk_begin_offset];
	uint32_t* const global_dest = &shared_ctx->d_dest[chunk_begin_offset];

	uint32_t* const shared_src1 = (uint32_t*)(((merge_sort_ctx_t*)shared) + 1);
	uint32_t* const shared_dest1 = shared_src1 + shared_chunk_size;

	uint32_t* const shared_src = global_src;
	uint32_t* const shared_dest = global_dest;

	if (threadIdx.x == 0)
	{
		/*
		printf("%llu\n", sizeof(merge_sort_ctx_t));
		printf("%llu %llu %llu %llu\n", &shared, shared_ctx, shared_src, shared_dest);
		printf("shared_chunk_size = %llu\n", shared_chunk_size);
		printf("shared_chunk_size_bytes = %llu\n", shared_chunk_size_bytes);
		printf("chunk_begin_offset = %llu\n", chunk_begin_offset);
		printf("d_src = %p, d_dest = %p, global_src = %p, global_dest = %p\n", ctx->d_src, ctx->d_dest, global_src, global_dest);
		if (shared_ctx->d_src != global_src) printf("%p != %p\n", shared_ctx->d_src, global_src);
		if (shared_ctx->d_dest != global_dest) printf("%p != %p\n", shared_ctx->d_dest, global_dest);
		*/
		memcpy(shared_src, global_src, shared_chunk_size_bytes);
		memcpy(shared_dest, global_dest, shared_chunk_size_bytes);
	}

	__syncthreads();

	if (thread_id < shared_ctx->running_threads)
	{
		size_t left = thread_id * (shared_ctx->run_size * 2);
		/*
		size_t right = left + (shared_ctx->run_size * 2);
		const size_t mid = left + ((right - left) / 2);
		size_t left_head = left;
		size_t right_head = mid;
		size_t write_head = left;
		*/

		const size_t rel_left = left - (block_first_thread_id * shared_ctx->run_size * 2);
		const size_t rel_right = rel_left + (shared_ctx->run_size * 2);
		const size_t rel_mid = rel_left + ((rel_right - rel_left) / 2);
		size_t rel_left_head = rel_left;
		size_t rel_right_head = rel_mid;
		size_t rel_write_head = rel_left;

		for (; rel_write_head < rel_right; ++rel_write_head)
		{
			if (rel_left_head < rel_mid && (rel_right_head >= rel_right || shared_src[rel_left_head] <= shared_src[rel_right_head]))
			{
				shared_dest[rel_write_head] = shared_src[rel_left_head];
				++rel_left_head;
			}
			else
			{
				shared_dest[rel_write_head] = shared_src[rel_right_head];
				++rel_right_head;
			}
		}

		if (threadIdx.x == 0)
		{
			memcpy(&shared_ctx->d_dest[chunk_begin_offset], shared_dest, shared_chunk_size_bytes);
		}

		__syncthreads();

		/*
		for (; write_head < right; ++write_head)
		{
			if (left_head < mid && (shared_ctx->d_src[left_head] < shared_ctx->d_src[right_head] || right_head >= right))
			{
				shared_ctx->d_dest[write_head] = shared_ctx->d_src[left_head];
				++left_head;
			}
			else
			{
				shared_ctx->d_dest[write_head] = shared_ctx->d_src[right_head];
				++right_head;
			}
		}
		*/
	}
}

int main()
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

	// Initial context on the host.
	merge_sort_ctx_t* d_ctx;
	cudaMallocManaged(&d_ctx, sizeof(merge_sort_ctx_t));
	d_ctx->elem_count = N_ELEMENTS;
	d_ctx->size_bytes = d_ctx->elem_count * sizeof(uint32_t);
	d_ctx->running_threads = d_ctx->elem_count / 2;
	d_ctx->run_size = 1;
	d_ctx->threads_per_block = MAX_THREADS_PER_BLOCK;
	d_ctx->total_blocks = d_ctx->running_threads / d_ctx->threads_per_block;
	if (d_ctx->running_threads % d_ctx->threads_per_block != 0) ++d_ctx->total_blocks;
	// Allocate buffers in GPU memory.
	cudaMalloc(&d_ctx->d_src, d_ctx->size_bytes);
	cudaMalloc(&d_ctx->d_dest, d_ctx->size_bytes);

	// Create the array to be sorted in host memory.
	uint32_t* h_src = new uint32_t[d_ctx->size_bytes];
	for (size_t i = 0; i < d_ctx->elem_count; ++i)
	{
		h_src[i] = d_ctx->elem_count - 1 - i;
		assert(h_src[i] == d_ctx->elem_count - 1 - i);
	}
	assert(h_src[0] == d_ctx->elem_count - 1);
	assert(h_src[d_ctx->elem_count - 1] == 0);

	// Destination buffer.
	uint32_t* h_dest = new uint32_t[d_ctx->size_bytes];
	memset(h_dest, 0, d_ctx->size_bytes);

	// Temporary results buffer.
	uint32_t* h_tmpsrc = new uint32_t[d_ctx->size_bytes];
	memset(h_tmpsrc, 0, d_ctx->size_bytes);

	begin = std::chrono::steady_clock::now();

	// Copy the array to be sorted into the GPU buffer.
	cudaMemcpy(d_ctx->d_src, h_src, d_ctx->size_bytes, cudaMemcpyHostToDevice);

	while (d_ctx->running_threads >= HOST_MAX_NATIVE_THREADS * 32 /* Found this value to be good */)
	{
		// Figure out the total number of thread blocks needed for the computation.
		d_ctx->total_blocks = d_ctx->running_threads / d_ctx->threads_per_block;
		if (d_ctx->running_threads % d_ctx->threads_per_block != 0) ++d_ctx->total_blocks;

		// Do a merge run.
		device_merge<< <d_ctx->total_blocks, d_ctx->threads_per_block>> > (d_ctx);
		cudaDeviceSynchronize();
		
		// Update the sort status.
		d_ctx->run_size *= 2;
		d_ctx->running_threads /= 2;

		swap(d_ctx->d_src, d_ctx->d_dest);
	}

	// Unswap.
	swap(d_ctx->d_src, d_ctx->d_dest);

	end = std::chrono::steady_clock::now();
	cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;

	// Copy the GPU results into host memory.
	cudaMemcpy(h_tmpsrc, d_ctx->d_dest, d_ctx->size_bytes, cudaMemcpyDeviceToHost);

	// Copy the context into host memory (for better performance).
	merge_sort_ctx_t h_ctx;
	cudaMemcpy(&h_ctx, d_ctx, sizeof(merge_sort_ctx_t), cudaMemcpyDeviceToHost);

	vector<thread*> threads;

	// Do the last runs on the CPU.
	while (h_ctx.running_threads != 0)
	{
		const size_t elems_per_thread = h_ctx.elem_count / h_ctx.running_threads;
		const size_t threads_per_batch = std::min(h_ctx.running_threads, HOST_MAX_NATIVE_THREADS);
		const size_t thread_batches = std::max((size_t)1ULL, h_ctx.running_threads / threads_per_batch);

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

		h_ctx.run_size *= 2;
		h_ctx.running_threads /= 2;

		swap(h_tmpsrc, h_dest);
	}

	// Unswap.
	swap(h_tmpsrc, h_dest);

	end = std::chrono::steady_clock::now();

	cout << "Sorted "
		<< h_ctx.elem_count
		<< " 32-bit integers in"
		<< endl;

	cout << "- "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms using cuda_sort"
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

	// QSORT

	uint32_t* src2 = new uint32_t[h_ctx.size_bytes];
	memcpy(src2, h_src, h_ctx.size_bytes);

	begin = std::chrono::steady_clock::now();
	qsort(src2, h_ctx.elem_count, sizeof(uint32_t), uint32_comp);
	end = std::chrono::steady_clock::now();
	cout << "- "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms using qsort"
		<< endl;
	delete[] src2;

	// STD::SORT

	std::vector<uint32_t> src3;
	for (size_t i = 0; i < h_ctx.elem_count; ++i)
		src3.push_back(0);
	memcpy(src3.data(), h_src, h_ctx.size_bytes);

	begin = std::chrono::steady_clock::now();
	std::sort(src3.begin(), src3.end());
	end = std::chrono::steady_clock::now();
	cout << "- "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms using std::sort"
		<< endl;

	cudaFree(d_ctx->d_src);
	cudaFree(d_ctx->d_dest);
	cudaFree(d_ctx);
	delete[] h_src;
	delete[] h_dest;
//	delete[] h_ctx;

	return 0;
}
