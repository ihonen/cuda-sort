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

// A struct containing the current status of the sort.
template<typename T>
struct merge_sort_ctx_t
{
	// The source buffer in GPU memory.
	T* d_src;
	// The destination buffer in GPU memory.
	T* d_dest;
	// The total number of elements in the array.
	size_t elem_count;
	// Self-explanatory.
	size_t size_bytes;
	// How many elements there are in a chunk.
	// A chunk is the data region a single thread is working on.
	// In merge, two sorted subarrays are merged into
	// a bigger subarray. The chunk size is the size of the two
	// subarrays (or the size of the bigger subarray).
	size_t elems_per_chunk;
	// How many chunks there currently are, i.e. how many threads
	// are running at the moment (one thread operates on one
	// chunk).
	size_t unmerged_chunks;
	// How many GPU threads there are in a thread block.
	size_t threads_per_block;
	// How many GPU thread blocks there are in total.
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
// Probably unnecessary, but ensures that the benchmarks are fair.
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

// Comparison function for ascending qsort.
template<typename T>
__forceinline static inline int int_comp(const void* a, const void* b)
{
	if (*((T*)a) < *((T*)b)) return -1;
	else if (*((T*)a) == *((T*)b)) return 0;
	else return 1;
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
// Ctx contains the current status of the sort,
// so each thread can figure out where the limits of
// the region of data it is handling are.
template<typename T>
__global__ void device_merge(merge_sort_ctx_t<T>* ctx)
{
	// "Global" "thread id".
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	// Store the data locations in local memory.
	T* d_src = ctx->d_src;
	T* d_dest = ctx->d_dest;

	// Perform a normal merge as per the definition of merge sort.
	// Compute the limits of the thread's data region based on thread id.
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

// Performs a sequential merge on the CPU
// according to the definition of merge sort.
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

// Sorts the array in src.
// The run configuration is provided in cfg.
// Returns a pointer to src.
template<typename T>
T* cuda_sort(T* h_src, merge_sort_cfg_t cfg)
{
	// Place the context in the GPU's memory.
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

	// Intermediate results buffer in host memory.
	T* h_tmp = new T[d_ctx->size_bytes];

	// Copy the array to be sorted into GPU memory.
	cudaMemcpy(d_ctx->d_src, h_src, d_ctx->size_bytes, cudaMemcpyHostToDevice);
	check_gpu_err();

	// Keep merging on the GPU as long as it's efficient.
	while (d_ctx->unmerged_chunks >= cfg.max_cpu_threads * 32 /* Found this value to be good */)
	{
		// Figure out the total number of thread blocks needed for the computation.
		d_ctx->total_thread_blocks = d_ctx->unmerged_chunks / d_ctx->threads_per_block;
		if (d_ctx->unmerged_chunks % d_ctx->threads_per_block != 0) ++d_ctx->total_thread_blocks;

		// Do a merge run on the GPU.
		device_merge<T> << <d_ctx->total_thread_blocks, d_ctx->threads_per_block >> > (d_ctx);
		cudaDeviceSynchronize();
		check_gpu_err();

		// Two equally sized subarrays were merged,
		// so double the chunk size and halve the
		// number of unmerged chunks.
		d_ctx->elems_per_chunk *= 2;
		d_ctx->unmerged_chunks /= 2;

		// The temporary results in d_src aren't needed, so d_dest can be used as the source buffer
		// for the next run and the data in d_src overwritten.
		swap(d_ctx->d_src, d_ctx->d_dest);
	}

	// Unswap so d_dest contains the intermediate result.
	swap(d_ctx->d_src, d_ctx->d_dest);

	// Copy the GPU results into host memory.
	cudaMemcpy(h_tmp, d_ctx->d_dest, d_ctx->size_bytes, cudaMemcpyDeviceToHost);
	check_gpu_err();

	// Copy the context into host memory (for efficiency).
	merge_sort_ctx_t<T> h_ctx;
	cudaMemcpy(&h_ctx, d_ctx, sizeof(merge_sort_ctx_t<T>), cudaMemcpyDeviceToHost);
	check_gpu_err();

	vector<thread*> threads;

	// These can be safely swapped.
	T* work_src = h_tmp;
	T* work_dest = h_src;

	// Do the last merge runs on the CPU.
	while (h_ctx.unmerged_chunks != 0)
	{
		// It's not efficient to run too many CPU threads at a time, so
		// run batches of threads with an appropriate number of threads per batch.
		const size_t elems_per_thread = h_ctx.elem_count / h_ctx.unmerged_chunks;
		const size_t threads_per_batch = std::min(h_ctx.unmerged_chunks, cfg.max_cpu_threads);
		const size_t thread_batches = std::max((size_t)1ULL, h_ctx.unmerged_chunks / threads_per_batch);

		for (size_t j = 0; j < thread_batches; ++j)
		{
			for (size_t i = 0; i < threads_per_batch; ++i)
			{
				size_t left = (j * elems_per_thread * threads_per_batch) + (i * elems_per_thread);
				size_t right = left + elems_per_thread;
				threads.push_back(new thread(host_merge<T>, work_dest, work_src, left, right));
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

		swap(work_src, work_dest);
	}

	// Unswap if necessary.
	swap(work_src, work_dest);
	if (h_src != work_dest) memcpy(h_src, work_dest, h_ctx.size_bytes);

#ifndef NDEBUG
	// Ensure that the results are correct.

	cout << "DEBUG: Checking cuda_sort result correctness" << endl;
	for (size_t i = 0; i < h_ctx.elem_count; ++i)
	{
		if (i == 0) continue;

		if (h_src[i] < h_src[i - 1])
		{
			cout << "ERROR: Erroneous cuda_sort output (first erroneous index = " << i << ", value = " << h_src[i] << ")!" << endl;
			break;
		}
	}
#endif // NDEBUG

	cudaFree(d_ctx->d_src);
	check_gpu_err();
	cudaFree(d_ctx->d_dest);
	check_gpu_err();
	cudaFree(d_ctx);
	check_gpu_err();

	delete[] h_tmp;

	return h_src;
}

// Runs a benchmark comparing cuda_sort, qsort and std::sort.
// T indicates the type of the data elements.
// Cfg contains the rest of the necessary information.
template<typename T>
void run_benchmark(merge_sort_cfg_t cfg)
{
	// Seed the RNG.
	srand(time(nullptr));

	// Source buffer with randomized contents.
	T* orig_src = new T[cfg.array_len];
	for (size_t i = 0; i < cfg.array_len; ++i)
		orig_src[i] = (T)rand();

	chrono::steady_clock::time_point cuda_begin;
	chrono::steady_clock::time_point cuda_end;
	chrono::steady_clock::time_point qsort_begin;
	chrono::steady_clock::time_point qsort_end;
	chrono::steady_clock::time_point stdsort_begin;
	chrono::steady_clock::time_point stdsort_end;

	// cuda_sort

	T* cuda_src = new T[cfg.array_len];
	memcpy(cuda_src, orig_src, sizeof(T) * cfg.array_len);

	overwrite_dcache();

	cuda_begin = chrono::steady_clock::now();
	cuda_sort<T>(cuda_src, cfg);
	cuda_end = chrono::steady_clock::now();

	delete[] cuda_src;

	// qsort
	
	T* qsort_src = new T[cfg.array_len];
	memcpy(qsort_src, orig_src, sizeof(T) * cfg.array_len);
	
	overwrite_dcache();

	qsort_begin = chrono::steady_clock::now();
	qsort(qsort_src, cfg.array_len, sizeof(T), int_comp<T>);
	qsort_end = chrono::steady_clock::now();

	delete[] qsort_src;

	// std::sort

	std::vector<T> stdsort_src;
	for (size_t i = 0; i < cfg.array_len; ++i)
		stdsort_src.push_back(0);
	memcpy(stdsort_src.data(), orig_src, sizeof(T) * cfg.array_len);

	overwrite_dcache();
	
	stdsort_begin = chrono::steady_clock::now();
	std::sort(stdsort_src.begin(), stdsort_src.end());
	stdsort_end = chrono::steady_clock::now();
	stdsort_src.clear();
	stdsort_src.shrink_to_fit();

	// Print the results.

	const size_t NAME_FIELD_WIDTH = 11;
	const size_t TIME_FIELD_WIDTH = 6;

	std::vector<std::string> algorithms = {"cuda_sort", "qsort", "std::sort"};
	std::vector<long long> times =
	{
		chrono::duration_cast<chrono::milliseconds>(cuda_end - cuda_begin).count(),
		chrono::duration_cast<chrono::milliseconds>(qsort_end - qsort_begin).count(),
		chrono::duration_cast<chrono::milliseconds>(stdsort_end - stdsort_begin).count()
	};

	for (size_t i = 0; i < algorithms.size(); ++i)
	{
		cout << setw(NAME_FIELD_WIDTH)
			<< left << algorithms[i]
			<< setw(TIME_FIELD_WIDTH) << right << times[i]
			<< " ms"
			<< endl;
	}

	delete[] orig_src;
}

// Primitive command line interface.
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
