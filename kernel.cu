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

// The status of the sort.
template<typename T>
struct merge_sort_ctx_t
{
	T* d_src;
	T* d_dest;
	T* tmp;
	size_t elem_count;
	size_t size_bytes;
	size_t elems_per_chunk;
	size_t unmerged_chunks;
	size_t threads_per_block;
	size_t total_thread_blocks;
};

// Command line arguments.
struct merge_sort_cfg_t
{
	size_t max_cpu_threads;
	size_t block_size;
	size_t array_len;
};

// Just a stupid function to overwrite everything in the data cache.
void overwrite_dcache()
{
	// Bytes
	static const size_t LARGER_THAN_L1 = 10000000;
	// -> uint64s
	static const size_t UINT64_COUNT = LARGER_THAN_L1 / 8;
	static uint64_t arr[UINT64_COUNT];
	// Rand will not be optimized out.
	for (size_t i = 0; i < UINT64_COUNT; ++i) arr[i] = rand();
}

// Comparison function for qsort.
template<typename T>
__forceinline static inline int int_comp(const void* a, const void* b)
{
	if (*((T*)a) < *((T*)b)) return -1;
	else if (*((T*)a) == *((T*)b)) return 0;
	else return 1;
}

// Debug function that prints an array in GPU code.
template<typename T>
__device__ void print_array_device(T* arr, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		printf("%*u", 5, arr[i]);
	}
	printf("\n");
}

// Debug function that prints an array in CPU code.
template<typename T>
__host__ void print_array_host(T* arr, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		printf("%*u ", 10, arr[i]);
	}
	printf("\n");
}

// Performs parallel merging on the GPU.
template<typename T>
__global__ void device_merge(merge_sort_ctx_t<T>* ctx)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	T* d_src = ctx->d_src;
	T* d_dest = ctx->d_dest;

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

// Performs a sequential merge on the CPU.
template<typename T>
__host__ void host_merge(T* dest, const T* src, const size_t left, const size_t right)
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

// Convenience wrapper for cudaGetLastError.
void check_gpu_err()
{
	auto err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << endl;
	}
}

// If a command line argument is "d", the default value is used. Check if so.
bool default_arg(const char* arg)
{
	return strcmp(arg, "d") == 0;
}

// Sorts the array in src and writes the result in dest.
// The run configuration is provided in cfg.
// Returns a pointer to dest.
template<typename T>
T* cuda_sort(T* h_dest, T* h_src, merge_sort_cfg_t cfg)
{
	// Initial context on the host.
	merge_sort_ctx_t<T>* d_ctx;
	cudaMallocManaged(&d_ctx, sizeof(merge_sort_ctx_t<T>));
	check_gpu_err();

	d_ctx->elem_count = cfg.array_len;
	d_ctx->size_bytes = d_ctx->elem_count * sizeof(T);
	d_ctx->unmerged_chunks = d_ctx->elem_count / 2;
	d_ctx->elems_per_chunk = 2;
	d_ctx->threads_per_block = cfg.block_size;
	d_ctx->total_thread_blocks = d_ctx->unmerged_chunks / d_ctx->threads_per_block;
	if (d_ctx->unmerged_chunks % d_ctx->threads_per_block != 0) ++d_ctx->total_thread_blocks;
	// Allocate buffers in GPU memory.
	cudaMalloc(&d_ctx->d_src, d_ctx->size_bytes);
	check_gpu_err();
	cudaMalloc(&d_ctx->d_dest, d_ctx->size_bytes);
	check_gpu_err();

	// Temporary results buffer.
	T* h_tmpsrc = new T[d_ctx->size_bytes];

	// Copy the array to be sorted into GPU memory.
	cudaMemcpy(d_ctx->d_src, h_src, d_ctx->size_bytes, cudaMemcpyHostToDevice);
	check_gpu_err();

	while (d_ctx->unmerged_chunks >= cfg.max_cpu_threads * 32 /* Found this value to be good */)
	{
		// Figure out the total number of thread blocks needed for the computation.
		d_ctx->total_thread_blocks = d_ctx->unmerged_chunks / d_ctx->threads_per_block;
		if (d_ctx->unmerged_chunks % d_ctx->threads_per_block != 0) ++d_ctx->total_thread_blocks;

		// Do a merge run.
		device_merge<T> << <d_ctx->total_thread_blocks, d_ctx->threads_per_block >> > (d_ctx);
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

	// Copy the GPU results into host memory.
	cudaMemcpy(h_tmpsrc, d_ctx->d_dest, d_ctx->size_bytes, cudaMemcpyDeviceToHost);
	check_gpu_err();

	// Copy the context into host memory (for better performance).
	merge_sort_ctx_t<T> h_ctx;
	cudaMemcpy(&h_ctx, d_ctx, sizeof(merge_sort_ctx_t<T>), cudaMemcpyDeviceToHost);
	check_gpu_err();

	vector<thread*> threads;

	// Do the last runs on the CPU.
	while (h_ctx.unmerged_chunks != 0)
	{
		const size_t elems_per_thread = h_ctx.elem_count / h_ctx.unmerged_chunks;
		const size_t threads_per_batch = std::min(h_ctx.unmerged_chunks, cfg.max_cpu_threads);
		const size_t thread_batches = std::max((size_t)1ULL, h_ctx.unmerged_chunks / threads_per_batch);

		for (size_t j = 0; j < thread_batches; ++j)
		{
			for (size_t i = 0; i < threads_per_batch; ++i)
			{
				size_t left = (j * elems_per_thread * threads_per_batch) + (i * elems_per_thread);
				size_t right = left + elems_per_thread;
				threads.push_back(new thread(host_merge<T>, h_dest, h_tmpsrc, left, right));
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

	for (size_t i = 0; i < h_ctx.elem_count; ++i)
	{
		if (i == 0) continue;

		if (h_dest[i] < h_dest[i - 1])
		{
			cout << "ERROR IN OUTPUT (first erroneous index = " << i << ", value = " << h_dest[i] << ")!" << endl;
			break;
		}
	}

	cudaFree(d_ctx->d_src);
	check_gpu_err();
	cudaFree(d_ctx->d_dest);
	check_gpu_err();
	cudaFree(d_ctx);
	check_gpu_err();

	return h_dest;
}

template<typename T>
void run_benchmark(merge_sort_cfg_t cfg)
{
	srand(time(nullptr));

	// Source buffer.
	T* cuda_src = new T[cfg.array_len];
	for (size_t i = 0; i < cfg.array_len; ++i)
		cuda_src[i] = (T)rand();

	// Destination buffer.
	T* cuda_dest = new T[cfg.array_len];
	memset(cuda_dest, 0, cfg.array_len * sizeof(T));

	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;

	// cuda_sort

	overwrite_dcache();

	begin = std::chrono::steady_clock::now();
	cuda_sort<T>(cuda_dest, cuda_src, cfg);
	end = std::chrono::steady_clock::now();

	cout << setw(11)
		<< left
		<< "cuda_sort"
		<< setw(6)
		<< right
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms"
		<< endl;

	// qsort

	overwrite_dcache();

	memset(cuda_dest, 0, sizeof(T) * cfg.array_len);
	
	T* qsort_src = new T[cfg.array_len];
	memcpy(qsort_src, cuda_src, sizeof(T) * cfg.array_len);

	begin = std::chrono::steady_clock::now();
	qsort(qsort_src, cfg.array_len, sizeof(T), int_comp<T>);
	end = std::chrono::steady_clock::now();
	cout << setw(11)
		<< left
		<< "qsort    "
		<< setw(6)
		<< right
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms"
		<< endl;

	// std::sort

	overwrite_dcache();

	std::vector<T> stdsort_src;
	for (size_t i = 0; i < cfg.array_len; ++i)
		stdsort_src.push_back(0);
	memcpy(stdsort_src.data(), cuda_src, sizeof(T) * cfg.array_len);

	begin = std::chrono::steady_clock::now();
	std::sort(stdsort_src.begin(), stdsort_src.end());
	end = std::chrono::steady_clock::now();
	cout << setw(11)
		<< left
		<< "std::sort"
		<< setw(6)
		<< right
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< " ms"
		<< endl;

	delete[] cuda_src;
	delete[] cuda_dest;
}

int main(int argc, char* argv[]) try
{
	merge_sort_cfg_t cfg;

	if (argc < 4)
	{
		cerr << "Invalid arguments" << endl;
		cerr << "Usage: cuda_sort.exe <number of CPU threads> <GPU thread block size> <array size as power of two>" << endl;
		cerr << "'d' = use default value" << endl;
		return -1;
	}
	else
	{
		if (default_arg(argv[1])) cfg.max_cpu_threads = 8;
		else cfg.max_cpu_threads = stoi(argv[1]);

		if (default_arg(argv[2])) cfg.block_size = 8;
		else cfg.block_size = stoi(argv[2]);

		if (default_arg(argv[3])) cfg.array_len = (size_t)pow(2, 24);
		else cfg.array_len = (size_t)pow(2, stoi(argv[3]));
	}

	cout << endl;
	cout << "DEBUG: Max CPU threads: " << cfg.max_cpu_threads << endl;
	cout << "DEBUG: GPU thread block size: " << cfg.block_size << endl;
	cout << "DEBUG: Array length: " << cfg.array_len << endl;
	cout << endl;

	cout << "8-BIT INTEGER SORT" << endl;
	run_benchmark<uint8_t>(cfg);
	cout << endl;

	cout << "16-BIT INTEGER SORT" << endl;
	run_benchmark<uint16_t>(cfg);
	cout << endl;

	cout << "32-BIT INTEGER SORT" << endl;
	run_benchmark<uint32_t>(cfg);
	cout << endl;

	cout << "64-BIT INTEGER SORT" << endl;
	run_benchmark<uint64_t>(cfg);
	cout << endl;

	cout << "32-BIT FLOATING POINT SORT" << endl;
	run_benchmark<float>(cfg);
	cout << endl;

	cout << "64-BIT FLOATING POINT SORT" << endl;
	run_benchmark<double>(cfg);
	cout << endl;

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
