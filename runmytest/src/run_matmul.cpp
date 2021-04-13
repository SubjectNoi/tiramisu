#include <iostream>
#include <thread>
#include <sys/time.h>
#include "Halide.h"
#include <omp.h>
#include <pthread.h>
using namespace std;
#define M 1024
#define N 1024
#define K 1024
#define PRINT_AFFINITY 0
#define USE_PTHREAD 1

#ifdef __cplusplus
extern "C" {
#endif

int matmul_1(halide_buffer_t *b1, halide_buffer_t *b2, halide_buffer_t *b3);

#ifdef __cplusplus
}  // extern "C"
#endif

float elapsed(struct timeval a, struct timeval b) {
	return 1000000.0 * (b.tv_sec - a.tv_sec) + 1.0 * (b.tv_usec - a.tv_usec);
}

void wrapper_1(halide_buffer_t *b1, halide_buffer_t *b2, halide_buffer_t *b3, int __iter, int core_num) {
    if (core_num == 0) return;
    halide_set_num_threads(core_num);
    struct timeval st, ed;
    gettimeofday(&st, NULL);
    for (int i = 0; i < __iter; i++) {
        matmul_1(b1, b2, b3);
    }
    gettimeofday(&ed, NULL);
    cout << core_num << " " << st.tv_sec << " " << st.tv_usec << " " << elapsed(st, ed) / (__iter * 1.0) << endl;
}

int main(int argc, char** argv) {
    Halide::Buffer<uint8_t> A_buf(M, K);
    Halide::Buffer<uint8_t> B_buf(K, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A_buf(i, j) = i + j;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B_buf(i, j) = i + j;
        }
    }
    Halide::Buffer<uint8_t> C_buf(M, N);
    cpu_set_t mask1;
    CPU_ZERO(&mask1);
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < atoi(argv[1]); i++) {
        CPU_SET(i, &mask1);
        CPU_SET(i, &mask_total);
    }
#if USE_PTHREAD==1
#if PRINT_AFFINITY==1
    cout << "Thr 0 Affinity = ";
    for (int i = 0; i < nproc; i++) {
        cout << CPU_ISSET(i, &mask1) << " ";
    }
    cout << endl;
#endif
    sched_setaffinity(0, sizeof(cpu_set_t), &mask_total);
    struct timeval st1, ed1, st2, ed2;
    int iter = 100;    cpu_set_t cst1, cst2;
    std::thread t1(wrapper_1, A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer(), 1, atoi(argv[1]));
    pthread_setaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &mask1);
    pthread_getaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &cst1);
    t1.join();
#if PRINT_AFFINITY==1
    cout << "Thr 0 Affinity = ";
    for (int i = 0; i < nproc; i++) {
        cout << CPU_ISSET(i, &cst1) << " ";
    }
    cout << endl;
#endif
#else
#pragma omp barrier
#pragma omp sections
{
#pragma omp section
{
    wrapper_1(A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer(), 1, atoi(argv[1]));
}
}
#endif
	return 0;
}
