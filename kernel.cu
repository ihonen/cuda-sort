#include "cuda_sort.cuh"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cassert>

using namespace std;

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

// If a command line argument is "d", the default value is used. Check if so.
bool default_arg(const char* arg)
{
	return strcmp(arg, "d") == 0;
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
