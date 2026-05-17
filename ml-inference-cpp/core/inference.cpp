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
#include <cstring>
#include "cachelog.h" // Notice the quotes for local files

using namespace std;

// Helper to open the perf event
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}


// define layer which can do matrix multiplication of inputs to weights to generate output vector
class Layer {
public:
    // We need input & output dim to allocate memory
    Layer(int input_dim, int output_dim, string weights_filepath, string bias_filepath) : in_dim(input_dim), out_dim(output_dim) {
        // allocate memory to hold weights and bias for each neuron
        // This layer contains input_dim number of inputs (weights) and output_dim number of neurons

        weights = load_params(weights_filepath, out_dim * in_dim);
        bias = load_params(bias_filepath, out_dim);
        out.resize(out_dim);
    }

    vector<float> load_params(const string& filepath, size_t num_elements) {
        vector<float> data(num_elements);

        ifstream file(filepath, ios::binary);
        if (!file) {
            cerr << "Error opening file: " << filepath << endl;
            return {};
        }

        file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

        if (!file) {
            cerr << "Error reading file: " << filepath << endl;
            return {};
        }

        return data;
    }

    // x[MxK] * W[KxN] = out[MxN]
    // M = 1, K = 784, N = 128
    void matmul(vector<float>& X) {
        // out[i] = x[k=0...in_dim] * [weights[k=0...in_dim] n=0...out_dim]
        for(int oi = 0; oi < out_dim; oi++) {
            float sum = 0;
            float* w_ptr = weights.data();
            w_ptr = (w_ptr + oi * in_dim);
            for(int ii = 0; ii < in_dim; ii++) {
                sum += X[ii] * w_ptr[ii];
            }
            out[oi] = sum;
        }
    }

    // ==============================
    // Matrix Multiply (TO IMPLEMENT)
    // ==============================
    void gemm(vector<float>& X) {
        /*
            M : batch size
            K : Input size
            N : Output size

            Weights: KxN (Takes K inputs and generates N number of outputs)
            X : MxK (For each input in batch (M), and K number of features for each input)
            Out: MxN (XW = out, generates N number of outputs for each of input in batch (M))
        */

        int M = 1; // batch size
        int K = in_dim;
        int N = out_dim;

        for(int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    sum += X[m * K + k] * weights[n * K + k];
                }
                out[m * N + n] = sum;
            }
        }
    }

    void add_bias() {
        for(int oi = 0; oi < out_dim; oi++) {
            out[oi] += bias[oi];
        }
    }

    vector<float> operator() (vector<float>& x) {
        assert(x.size() == in_dim && "Input size mismatch with layer");

        // matrix multiply x with weights and get output sized vector
        // matmul(x);
        profile.start();
        // gemm(x);
        matmul(x);
        profile.stop_and_print();
        add_bias();

        return out;
    }

private:
    int in_dim;
    int out_dim;
    vector<float> weights;
    vector<float> bias;
    vector<float> out;
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

    void relu(vector<float>& X) {
        for(int i = 0; i < X.size(); i++) {
            X[i] = max(0.0f, X[i]);
        }
    }

    void operator() (vector<float>& X) {
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

vector<float> load_input(const string& filepath, size_t num_elements) {
    vector<float> data(num_elements);

    ifstream file(filepath, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << filepath << endl;
        return {};
    }

    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

    if (!file) {
        cerr << "Error reading file: " << filepath << endl;
        return {};
    }

    return data;
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

int infer(vector<float>& X, Layer& fc1, Layer& fc2, Layer& fc3, Activation& act) {
    X = fc1(X);
    act.relu(X);
    X = fc2(X);
    act.relu(X);
    X = fc3(X);

    auto max_it = max_element(X.begin(), X.end());
    return distance(X.begin(), max_it);
}

void Eval() {
    // build model
    Layer fc1 = Layer(784, 128, "../weights/fc1.weight.bin", "../weights/fc1.bias.bin");
    Layer fc2 = Layer(128, 64, "../weights/fc2.weight.bin", "../weights/fc2.bias.bin");
    Layer fc3 = Layer(64, 10, "../weights/fc3.weight.bin", "../weights/fc3.bias.bin");
    // activation functions
    Activation act(ActivationMethod::relu);

    // vector<float> X = load_input("../data_reference/input_0.bin", 784);
    // vector<float> fc1_py = load_input("../data_reference/fc1_linear_0.bin", 128);
    // vector<float> fc1_cpp = fc1(X);
    // act.relu(fc1_cpp);

    // vector<float> fc2_py = load_input("../data_reference/fc2_linear_0.bin", 64);
    // vector<float> fc2_cpp = fc2(fc1_cpp);
    // act.relu(fc2_cpp);

    // vector<float> fc3_py = load_input("../data_reference/output_0.bin", 10);
    // vector<float> fc3_cpp = fc3(fc2_cpp);

    auto samples = load_labels("../data_input/labels.txt");
    int count = 0;
    int i = 0;

    double total_inference_time = 0;

    for (const auto& s : samples) {
        std::string full_path = "../data_input/" + s.filename;

        vector<float> X = load_input(full_path, 784);

        auto start = chrono::high_resolution_clock::now();
        int pred = infer(X, fc1, fc2, fc3, act);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_inference_time += elapsed.count();
        // X = fc1(X);
        // act.relu(X);
        // X = fc2(X);
        // act.relu(X);
        // X = fc3(X);

        // auto max_it = max_element(X.begin(), X.end());
        // int pred = distance(X.begin(), max_it);

        count += (pred == s.label);
        
        // printf("Samples inferred:%d\n", i);
        // if ( i > 100) break;
        i++;
    }

    cout << left << setw(20) << "Latency:"      << (total_inference_time * 1000) / (float)count << " ms" << endl;
    printf("Accuracy (total:%d): %.4f\n", samples.size(), ((float)count) / ((float)samples.size()));
}

int main()
{
    Eval();
}