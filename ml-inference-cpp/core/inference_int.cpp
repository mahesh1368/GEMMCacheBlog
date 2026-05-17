#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <unistd.h>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include "arm_neon.h"
#include "cachelog.h" // Notice the quotes for local files


using namespace std;

// Stats[10000]: r1:[0.0, 12.292891502380371] r2:[0.0, 10.725931167602539] out:[-9.165567398071289, 10.796649932861328]

// Helper to open the perf event
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}


// define layer which can do matrix multiplication of inputs to weights to generate output vector
class Layer {
public:
    // We need input & output dim to allocate memory
    Layer(int input_dim, int output_dim, string weights_filepath, string bias_filepath, float in_scale, float out_max_val) 
        : in_dim(input_dim), out_dim(output_dim), input_scale(in_scale) {
        
        weights = load_params(weights_filepath, out_dim * in_dim, weight_scale);
        bias = load_params(bias_filepath, out_dim, bias_scale);
        
        // Use the provided output range stats to determine the scale for this layer's output
        output_scale = out_max_val / 127.0f;
        out.resize(output_dim);
    }

    vector<int8_t> load_params(const string& filepath, size_t num_elements, float& scale_out) {
        vector<float> data(num_elements);
        vector<int8_t> data_i(num_elements);

        ifstream file(filepath, ios::binary);
        if (!file) {
            cerr << "Error opening file: " << filepath << endl;
            return {};
        }

        file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

        float max_val = 0.0f;
        for (float val : data) {
            max_val = max(max_val, fabsf(val));
        }

        scale_out = max_val / 127.0f;

        for (size_t i = 0; i < num_elements; i++) {
            data_i[i] = (int8_t)round(data[i] / scale_out);
        }

        return data_i;
    }

    // x[MxK] * W[KxN] = out[MxN]
    // M = 1, K = 784, N = 128
    void matmul(vector<int8_t>& X) {
        for(int oi = 0; oi < out_dim; oi++) {
            int32_t acc = 0;
            int8_t* w_ptr = weights.data() + (oi * in_dim);
            for(int ii = 0; ii < in_dim; ii++) {
                acc += (int32_t)X[ii] * (int32_t)w_ptr[ii];
            }
            
            // Re-scale: (X * W * scale_x * scale_w) + (B * scale_b)
            float val = (acc * input_scale * weight_scale) + (bias[oi] * bias_scale);
            
            // Quantize to the next layer's scale
            out[oi] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
        }
    }

    void gemm(vector<int8_t>& X) {
        /*
            M : batch size
            K : Input size
            N : Output size

            Weights: KxN (Takes K inputs and generates N number of outputs)
            X : MxK (For each input in batch (M), and K number of features for each input)
            Out: MxN (XW = out, generates N number of outputs for each of input in batch (M))
        */

        int M = X.size() / in_dim;
        int K = in_dim;
        int N = out_dim;

        // Accessing weights is still not optimized for cache utilization
        // If observed carefully, weights are re-used for all the inputs
        // Assume that entire weights doesn't fit into cache and if we access them sequentially,
        // we have to bring start of the weights array values into cache once the end of it is processed
        // For example, half of the weights size can be fit into cache, you load first half compute against the input
        // then load the second part and complete the calculation, at this point, first part is removed from cache
        // again when second input is arrived, it tried to bring the first half again into cache
        // Idea is to use the first half as long as possible and then discard it repeat the same with second half
        // This should ensure we use weights while they are already in the cache

        int BS = 8; // block size for efficient cache utilization

        int M_rounded = (M / BS) * BS;
        int N_rounded = (N / BS) * BS;
        int K_rounded = (K / BS) * BS;

        // printf("M:%d K:%d N:%d BS:%d\n", M, K, N, BS);

        // std::fill(out.begin(), out.end(), 0);
        // assume BS divides the array properl
        for (int mm = 0; mm < M_rounded; mm += BS) {
            for (int nn = 0; nn < N_rounded; nn += BS) {
                int32_t accum[BS][BS] = {0};

                for (int kk = 0; kk < K_rounded; kk += BS) {
                    for (int m = 0 ; m < BS; m++) {
                        for (int n = 0; n < BS; n++) {
                            for (int k = kk; k < kk+BS; k++) {
                                accum[m][n] += (int32_t)X[(m + mm) * K + k] * (int32_t)weights[(nn + n) * K + k];
                            }
                        }
                    }
                }

                // if K is not perfect multiple of BS, then handle left over things separately
                if (K_rounded < K) {
                    for (int m = 0 ; m < BS; m++) {
                        for (int n = 0; n < BS; n++) {
                            for (int k = K_rounded; k < K; k++) {
                                accum[m][n] += (int32_t)X[(m + mm) * K + k] * (int32_t)weights[(nn + n) * K + k];
                            }
                        }
                    }
                }

                // once the block processing ends, quantize outputs and store it back

                for (int n = 0; n < BS; n++) {
                    float bias_comp = (bias[nn + n] * bias_scale);
                    for (int m = 0; m < BS; m++) {
                        // Re-scale: (X * W * scale_x * scale_w) + (B * scale_b)
                        float val = (accum[m][n] * input_scale * weight_scale) + bias_comp;
                        out[(mm + m) * N + (nn + n)] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
                    }
                }
            }
        }

        if (M_rounded < M) {
            for(int m = M_rounded; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    int32_t acc = 0;
                    for(int k = 0; k < K; k++) {
                        acc += (int32_t)X[m * K + k] * (int32_t)weights[n * K + k];
                    }
                    float val = (acc * input_scale * weight_scale) + (bias[n] * bias_scale);
                    out[m * N + n] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
                }
            }
        }

        if (N_rounded < N) {
            for(int m = 0; m < M; m++) {
                for (int n = N_rounded; n < N; n++) {
                    int32_t acc = 0;
                    for(int k = 0; k < K; k++) {
                        acc += (int32_t)X[m * K + k] * (int32_t)weights[n * K + k];
                    }
                    float val = (acc * input_scale * weight_scale) + (bias[n] * bias_scale);
                    out[m * N + n] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
                }
            }
        }
    }

    void gemm_neon(vector<int8_t>& X) {
        int M = X.size() / in_dim;
        int K = in_dim;
        int N = out_dim;
        constexpr int BS = 8;

        // printf("M:%d K:%d N:%d BS:%d\n", M, K, N, BS);

        int M_rounded = (M / BS) * BS;
        int N_rounded = (N / BS) * BS;
        int K_rounded = (K / 16) * 16; // NEON processes 16 bytes at a time

        // 1. Fast Path: Full BS x BS blocks with NEON vectorization along K
        for (int mm = 0; mm < M_rounded; mm += BS) {
            for (int nn = 0; nn < N_rounded; nn += BS) {
                // Initialize a 2D array of NEON registers for accumulators (4x4 grid of 32-bit lanes)
                int32x4_t accum[BS][BS];
                for (int m = 0; m < BS; m++) {
                    for (int n = 0; n < BS; n++) {
                        accum[m][n] = vdupq_n_s32(0);
                    }
                }

                // Vectorized K loop
                for (int kk = 0; kk < K_rounded; kk += 16) {
                    for (int m = 0; m < BS; m++) {
                        // Load 16 elements of X
                        int8x16_t x_vec = vld1q_s8(&X[(mm + m) * K + kk]);

                        for (int n = 0; n < BS; n++) {
                            // Load 16 elements of weights
                            int8x16_t w_vec = vld1q_s8(&weights[(nn + n) * K + kk]);

                            // Multiply and accumulate: int8x16 * int8x16 -> int32x4 accumulated
                            // vdotq_s32 computes dot product of groups of 4 bytes and accumulates into int32x4
                            accum[m][n] = vdotq_s32(accum[m][n], x_vec, w_vec);
                        }
                    }
                }

                // Horizontal add the 4 lanes of each NEON register into scalar int32_t
                int32_t scalar_accum[BS][BS];
                for (int m = 0; m < BS; m++) {
                    for (int n = 0; n < BS; n++) {
                        scalar_accum[m][n] = vaddvq_s32(accum[m][n]);
                    }
                }

                // Handle remaining K elements that didn't fit into the 16-byte vector width
                if (K_rounded < K) {
                    for (int m = 0; m < BS; m++) {
                        for (int n = 0; n < BS; n++) {
                            for (int k = K_rounded; k < K; k++) {
                                scalar_accum[m][n] += (int32_t)X[(mm + m) * K + k] * (int32_t)weights[(nn + n) * K + k];
                            }
                        }
                    }
                }

                // Quantize and write back
                for (int n = 0; n < BS; n++) {
                    float bias_comp = (bias[nn + n] * bias_scale);
                    for (int m = 0; m < BS; m++) {
                        float val = (scalar_accum[m][n] * input_scale * weight_scale) + bias_comp;
                        out[(mm + m) * N + (nn + n)] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
                    }
                }
            }
        }

        // 2. Unified Cleanup: Remaining N columns for the main M rows
        if (N_rounded < N) {
            for (int mm = 0; mm < M_rounded; mm++) {
                for (int nn = N_rounded; nn < N; nn++) {
                    int32x4_t acc_vec = vdupq_n_s32(0);
                    for (int kk = 0; kk < K_rounded; kk += 16) {
                        int8x16_t x_vec = vld1q_s8(&X[mm * K + kk]);
                        int8x16_t w_vec = vld1q_s8(&weights[nn * K + kk]);
                        acc_vec = vdotq_s32(acc_vec, x_vec, w_vec);
                    }
                    int32_t acc = vaddvq_s32(acc_vec);
                    for (int k = K_rounded; k < K; k++) {
                        acc += (int32_t)X[mm * K + k] * (int32_t)weights[nn * K + k];
                    }
                    float val = (acc * input_scale * weight_scale) + (bias[nn] * bias_scale);
                    out[mm * N + nn] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
                }
            }
        }

        // 3. Unified Cleanup: Handle all remaining M rows (including bottom-right corner)
        if (M_rounded < M) {
            for (int mm = M_rounded; mm < M; mm++) {
                for (int nn = 0; nn < N; nn++) {
                    int32x4_t acc_vec = vdupq_n_s32(0);
                    for (int kk = 0; kk < K_rounded; kk += 16) {
                        int8x16_t x_vec = vld1q_s8(&X[mm * K + kk]);
                        int8x16_t w_vec = vld1q_s8(&weights[nn * K + kk]);
                        acc_vec = vdotq_s32(acc_vec, x_vec, w_vec);
                    }
                    int32_t acc = vaddvq_s32(acc_vec);
                    for (int k = K_rounded; k < K; k++) {
                        acc += (int32_t)X[mm * K + k] * (int32_t)weights[nn * K + k];
                    }
                    float val = (acc * input_scale * weight_scale) + (bias[nn] * bias_scale);
                    out[mm * N + nn] = (int8_t)clamp((int)round(val / output_scale), -127, 127);
                }
            }
        }
    }

    void add_bias() {
        // Handled inside matmul to maintain precision before re-quantization
    }

    vector<int8_t> operator() (vector<int8_t>& x) {
        assert(x.size() == in_dim && "Input size mismatch with layer");

        profile.start();
        // matmul(x);
        // gemm(x);
        gemm_neon(x);
        profile.stop_and_print();

        return out;
    }

    float get_output_scale() { return output_scale; }

    int getInputSize() {return in_dim;};

    int getOutputSize() {return out_dim;};

private:
    int in_dim;
    int out_dim;
    vector<int8_t> weights;
    vector<int8_t> bias;
    vector<int8_t> out;
    float input_scale;
    float weight_scale;
    float bias_scale;
    float output_scale;
    ARMCacheProfiler profile;
};

// define an enum to configure activation function
enum ActivationMethod {
    relu = 0,
    totalMethods
};

class Activation {
public:
    Activation(ActivationMethod act) : m_act(act) {};

    void relu(vector<int8_t>& X) {
        for(int i = 0; i < X.size(); i++) {
            X[i] = max((int8_t)0, X[i]);
        }
    }

    void operator() (vector<int8_t>& X) {
        switch(m_act) {
            case ActivationMethod::relu:
                relu(X);
            default:
                break;
        }
    }

private:
    ActivationMethod m_act;
};

vector<int8_t> load_input(const string& filepath, size_t num_elements, float& scale_out) {
    vector<float> data(num_elements);
    vector<int8_t> data_i(num_elements);

    ifstream file(filepath, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << filepath << endl;
        return {};
    }

    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

    float max_val = 0.0f;
    for (float val : data) {
        max_val = max(max_val, fabsf(val));
    }
    
    scale_out = max_val / 127.0f;

    for (size_t i = 0; i < num_elements; i++) {
        data_i[i] = (int8_t)round(data[i] / scale_out);
    }

    return data_i;
}

struct Sample {
    std::string filename;
    int label;
};

std::vector<Sample> load_labels(const std::string& filepath) {
    std::vector<Sample> samples;

    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "Error opening label file\n";
        return samples;
    }

    std::string fname;
    int label;

    while (file >> fname >> label) {
        samples.push_back({fname, label});
    }

    return samples;
}

void compare(vector<float>& A, vector<float>& B) {
    bool pass = true;
    if(A.size() != B.size()) {pass = false;};
    for(int i = 0; i < A.size(); i++) {
        if (fabs(A[i] - B[i]) > 1e-4f) {
            pass = false;
            break;
        }
        if (i % 1 == 0) {
            printf("A:%.4f B:%.4f\n", A[i], B[i]);
        }
    }
    
    (pass) ? (printf("A=B\n")) : printf("A!=B\n");
}

vector<int> infer(
    vector<int8_t>& X,
    Layer& fc1,
    Layer& fc2,
    Layer& fc3,
    Activation& act
) {
    // ==========================================
    // Derive batch size from input tensor
    // Input shape: [M x 784]
    // ==========================================
    int batch_size = X.size() / fc1.getInputSize();

    // ==========================================
    // Forward Pass
    // ==========================================
    X = fc1(X);

    act.relu(X);

    X = fc2(X);

    act.relu(X);

    X = fc3(X);

    // ==========================================
    // Output shape:
    // [batch_size x 10]
    // ==========================================
    const int NUM_CLASSES = fc3.getOutputSize();

    vector<int> preds(batch_size);

    for (int b = 0; b < batch_size; b++)
    {
        auto begin_it =
            X.begin() + b * NUM_CLASSES;

        auto end_it =
            begin_it + NUM_CLASSES;

        auto max_it =
            max_element(begin_it, end_it);

        preds[b] =
            distance(begin_it, max_it);
    }

    return preds;
}

void Eval() {
    // Initial input scale needs to be captured from the first loaded file
    float current_in_scale = 1.0f / 127.0f; // Placeholder until first file load

    // Stats[10000]: r1:[0.0, 12.29...] r2:[0.0, 10.72...] out:[-9.16, 10.79...]
    Layer fc1 = Layer(784, 128, "../weights/fc1.weight.bin", "../weights/fc1.bias.bin", current_in_scale, 12.292891502380371f);
    Layer fc2 = Layer(128, 64, "../weights/fc2.weight.bin", "../weights/fc2.bias.bin", fc1.get_output_scale(), 10.725931167602539f);
    Layer fc3 = Layer(64, 10, "../weights/fc3.weight.bin", "../weights/fc3.bias.bin", fc2.get_output_scale(), 10.796649932861328f);

    Activation act(ActivationMethod::relu);

    auto samples = load_labels("../data_input/labels.txt");
    int count = 0;
    int i = 0;
    int batchSize = 64;

    double total_inference_time = 0;
    for (const auto& s : samples) {
        std::string full_path = "../data_input/" + s.filename;

        float dummy_scale;
        vector<int8_t> X = load_input(full_path, 784, dummy_scale);

        auto start = chrono::high_resolution_clock::now();
        vector<int> pred = infer(X, fc1, fc2, fc3, act);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_inference_time += elapsed.count();

        count += (pred[0] == s.label);
        i++;
    }

    cout << left << setw(20) << "Latency:"      << (total_inference_time * 1000)/(float)count << " ms" << endl;
    printf("Accuracy (total:%d): %.4f\n", samples.size(), ((float)count) / ((float)samples.size()));
}

void Eval_Batch() {
    // Initial input scale needs to be captured from the first loaded file
    float current_in_scale = 1.0f / 127.0f;

    // Stats[10000]: r1:[0.0, 12.29...] r2:[0.0, 10.72...] out:[-9.16, 10.79...]
    Layer fc1 = Layer(
        784,
        128,
        "../weights/fc1.weight.bin",
        "../weights/fc1.bias.bin",
        current_in_scale,
        12.292891502380371f
    );

    Layer fc2 = Layer(
        128,
        64,
        "../weights/fc2.weight.bin",
        "../weights/fc2.bias.bin",
        fc1.get_output_scale(),
        10.725931167602539f
    );

    Layer fc3 = Layer(
        64,
        10,
        "../weights/fc3.weight.bin",
        "../weights/fc3.bias.bin",
        fc2.get_output_scale(),
        10.796649932861328f
    );

    Activation act(ActivationMethod::relu);

    auto samples = load_labels("../data_input/labels.txt");

    int correct = 0;
    int total = 0;

    const int INPUT_SIZE = 784;
    const int BATCH_SIZE = 4;

    double total_inference_time = 0.0;

    // ==================================================
    // Batch Loop
    // ==================================================
    for (size_t batch_start = 0;
         batch_start < samples.size();
         batch_start += BATCH_SIZE)
    {
        int current_batch_size =
            std::min(
                BATCH_SIZE,
                (int)(samples.size() - batch_start)
            );

        // [M x INPUT_SIZE]
        vector<int8_t> batch_X(
            current_batch_size * INPUT_SIZE
        );

        vector<int> batch_labels(current_batch_size);

        // ==============================================
        // Load Batch
        // ==============================================
        for (int b = 0; b < current_batch_size; b++)
        {
            const auto& s = samples[batch_start + b];

            std::string full_path =
                "../data_input/" + s.filename;

            float dummy_scale;

            vector<int8_t> single_X =
                load_input(
                    full_path,
                    INPUT_SIZE,
                    dummy_scale
                );

            memcpy(
                &batch_X[b * INPUT_SIZE],
                single_X.data(),
                INPUT_SIZE * sizeof(int8_t)
            );

            batch_labels[b] = s.label;
        }

        // ==============================================
        // Batched Inference
        // ==============================================
        auto start = chrono::high_resolution_clock::now();

        // infer() now returns predictions for batch
        vector<int> preds =
            infer(
                batch_X,
                fc1,
                fc2,
                fc3,
                act
            );

        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;
        total_inference_time += elapsed.count();

        // ==============================================
        // Accuracy
        // ==============================================
        for (int b = 0; b < current_batch_size; b++)
        {
            correct += (preds[b] == batch_labels[b]);
            total++;
        }
    }

    cout << left << setw(20)
         << "Latency:"
         << (total_inference_time * 1000.0) / (double)total
         << " ms"
         << endl;

    printf(
        "Accuracy: %.4f\n",
        ((float)correct) / ((float)samples.size())
    );
}

int main()
{
    Eval();
}