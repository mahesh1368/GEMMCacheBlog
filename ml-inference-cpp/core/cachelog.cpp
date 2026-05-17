#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>            
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>        
#include <linux/perf_event.h>
#include "cachelog.h"

// System call wrapper
long perf_event_open_sys(struct perf_event_attr *hw_event, pid_t pid, 
                         int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int ARMCacheProfiler::open_perf_counter(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    std::memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;         
    pe.exclude_kernel = 1;   
    pe.exclude_hv = 1;       

    int fd = perf_event_open_sys(&pe, 0, -1, -1, 0);
    return fd;
}

ARMCacheProfiler::ARMCacheProfiler() : fd_cycles(-1), fd_cache_access(-1), fd_cache_miss(-1), print_only_one_time(true) {
    fd_cycles = open_perf_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);

    fd_cache_access = open_perf_counter(PERF_TYPE_HW_CACHE, 
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));

    fd_cache_miss = open_perf_counter(PERF_TYPE_HW_CACHE, 
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
}

ARMCacheProfiler::~ARMCacheProfiler() {
    if (fd_cycles >= 0) close(fd_cycles);
    if (fd_cache_access >= 0) close(fd_cache_access);
    if (fd_cache_miss >= 0) close(fd_cache_miss);
}

void ARMCacheProfiler::start() {
    if (fd_cycles >= 0) { ioctl(fd_cycles, PERF_EVENT_IOC_RESET, 0); ioctl(fd_cycles, PERF_EVENT_IOC_ENABLE, 0); }
    if (fd_cache_access >= 0) { ioctl(fd_cache_access, PERF_EVENT_IOC_RESET, 0); ioctl(fd_cache_access, PERF_EVENT_IOC_ENABLE, 0); }
    if (fd_cache_miss >= 0) { ioctl(fd_cache_miss, PERF_EVENT_IOC_RESET, 0); ioctl(fd_cache_miss, PERF_EVENT_IOC_ENABLE, 0); }
}

void ARMCacheProfiler::stop_and_print() {

    if(false == print_only_one_time) return;
    long long cycles = 0, accesses = 0, misses = 0;

    if (fd_cycles >= 0) { ioctl(fd_cycles, PERF_EVENT_IOC_DISABLE, 0); ::read(fd_cycles, &cycles, sizeof(long long)); }
    if (fd_cache_access >= 0) { ioctl(fd_cache_access, PERF_EVENT_IOC_DISABLE, 0); ::read(fd_cache_access, &accesses, sizeof(long long)); }
    if (fd_cache_miss >= 0) { ioctl(fd_cache_miss, PERF_EVENT_IOC_DISABLE, 0); ::read(fd_cache_miss, &misses, sizeof(long long)); }

    double miss_rate = (accesses > 0) ? ((double)misses / accesses) * 100.0 : 0.0;
    double hit_rate = 100.0 - miss_rate;

    std::cout << "\n=== Cache Performance Profile ===\n";
    std::cout << "Total CPU Cycles   : " << cycles << "\n";
    std::cout << "L1D Cache Accesses : " << accesses << "\n";
    std::cout << "L1D Cache Misses   : " << misses << "\n";
    std::cout << "L1D Cache Hit Rate : " << hit_rate << " %\n";
    std::cout << "L1D Cache Miss Rate: " << miss_rate << " %\n";
    std::cout << "=================================\n";

    print_only_one_time = false;
}