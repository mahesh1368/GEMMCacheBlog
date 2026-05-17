#ifndef PROFILER_H
#define PROFILER_H

#include <cstdint>

class ARMCacheProfiler {
private:
    int fd_cycles;
    int fd_cache_access;
    int fd_cache_miss;
    bool print_only_one_time;
    int open_perf_counter(uint32_t type, uint64_t config);

public:
    ARMCacheProfiler();
    ~ARMCacheProfiler();
    void start();
    void stop_and_print();
};

#endif