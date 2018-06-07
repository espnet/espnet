#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>

#include <ctc.h>

#include "test.h"

bool small_test() {
    const int alphabet_size = 5;
    const int T = 2;

    std::vector<float> activations = {0.1, 0.6, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.6, 0.1, 0.1};

    // Calculate the score analytically
    float expected_score;
    {
        std::vector<float> probs(activations.size());
        softmax(activations.data(), alphabet_size, T, probs.data());

        // Score calculation is specific to the given activations above
        expected_score = probs[1] * probs[7];
    }

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *activations_gpu;
    throw_on_error(cudaMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(activations_gpu, nullptr,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score,
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    score = std::exp(-score);
    const float eps = 1e-6;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");

    return (score > lb && score < ub);
}

int offset(int t, int n, int a) {
    constexpr int minibatch = 2;
    constexpr int alphabet_size = 6;
    return (t * minibatch + n) * alphabet_size + a;
}

bool options_test() {
    const int alphabet_size = 6;
    const int T = 5;
    const int minibatch = 2;

    std::vector<float> activations =
            {0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553,
             0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508,

             0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436,
             0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549,

             0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688,
             0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456,

             0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533,
             0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345,

             0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107,
             0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046};

    std::vector<float> expected_grads = // from tensorflow
            {-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553,
             -0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508,

             0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436,
             0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549,

             0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688,
             0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544,

             0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533,
             0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345,

             -0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107,
             -0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046};

    // Calculate the expected scores analytically
    auto& a = activations;
    double expected_score[2];
    expected_score[0] =
            -std::log(a[offset(0, 0, 0)] * a[offset(1, 0, 1)] * a[offset(2, 0, 2)]
                     * a[offset(3, 0, 1)] * a[offset(4, 0, 0)]);
    expected_score[1] = 5.42262; // from tensorflow

    // now take the log to account for the softmax
    for (auto& a : activations) {
        a = std::log(a);
    }

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *activations_gpu;
    throw_on_error(cudaMalloc(&activations_gpu,
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> labels = {0, 1, 2, 1, 0,
                               0, 1, 1, 0};

    std::vector<int> label_lengths = {5, 4};

    std::vector<int> lengths = {5, 5};

    float score[2];

    float *grads_gpu;
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T * minibatch) * sizeof(float)),
                   "cudaMalloc");

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;
    options.blank_label = 5;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(activations_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score[0],
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in options_test");

    std::vector<float> grads(alphabet_size * T * minibatch);
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");

    const double eps = 1e-4;

    bool result = true;
    for (int i = 0; i < grads.size(); i++) {
        const double lb = expected_grads[i] - eps;
        const double ub = expected_grads[i] + eps;
        if (!(grads[i] > lb && grads[i] < ub)) {
            std::cerr << "grad mismatch in options_test"
                      << " expected grad: " << expected_grads[i]
                      << " calculated score: " << grads[i]
                      << " !(" << lb << " < " << grads[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_score[i] - eps;
        const double ub = expected_score[i] + eps;

        if (!(score[i] > lb && score[i] < ub)) {
            std::cerr << "score mismatch in options_test"
                      << " expected score: " << expected_score[i]
                      << " calculated score: " << score[i] << std::endl;
            result = false;
        }
    }
    return result;
}


bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L);
    labels[0] = 2;
    std::vector<int> label_lengths = {L};

    std::vector<float> acts = genActs(alphabet_size * T * minibatch);

    for (int i = 0; i < T; ++i)
        acts[alphabet_size * i + 2] = -1e30;

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *acts_gpu;
    throw_on_error(cudaMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> lengths;
    lengths.push_back(T);

    float *grads_gpu;
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T) * sizeof(float)),
                   "cudaMalloc");

    float cost;

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &cost,
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in inf_test");

    bool status = std::isinf(cost);

    std::vector<float> grads(alphabet_size * T);
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu,
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    for (int i = 0; i < alphabet_size * T; ++i)
        status &= !std::isnan(grads[i]);

    throw_on_error(cudaFree(acts_gpu), "cudaFree");
    throw_on_error(cudaFree(grads_gpu), "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace), "cudaFree");
    throw_on_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    return status;
}

float grad_check(int T, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& lengths) {

    float epsilon = 1e-2;

    const int minibatch = labels.size();

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *acts_gpu;
    throw_on_error(cudaMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch);

    float *grads_gpu;
    throw_on_error(cudaMalloc(&grads_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");

    ctcOptions options{};
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(),
                                      lengths.data(),
                                      alphabet_size,
                                      lengths.size(),
                                      options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    flat_labels.data(),
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss (0) in grad_check");

    std::vector<float> grads(acts.size());
    throw_on_error(cudaMemcpyAsync(grads.data(),
                                   grads_gpu, grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    for (int i = 0; i < T * alphabet_size * minibatch; ++i) {
        acts[i] += epsilon;

        throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        throw_on_error(compute_ctc_loss(acts_gpu, NULL,
                                        flat_labels.data(),
                                        label_lengths.data(),
                                        lengths.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        ctc_gpu_workspace,
                                        options),
                       "Error: compute_ctc_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;
        throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");

        throw_on_error(compute_ctc_loss(acts_gpu, NULL,
                                        flat_labels.data(),
                                        label_lengths.data(),
                                        lengths.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        ctc_gpu_workspace,
                                        options),
                       "Error: compute_ctc_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

        acts[i] += epsilon;

        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }

    float diff = rel_diff(grads, num_grad);

    throw_on_error(cudaFree(acts_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(grads_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");

    return diff;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
        { std::make_tuple(28, 50, 15, 1, 1e-5) };

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts = genActs(alphabet_size * T * minibatch);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        float diff = grad_check(T, alphabet_size, acts, labels, sizes);
        status &= (diff < tol);
    }

    return status;
}

int main(void) {
    if (get_warpctc_version() != 2) {
        std::cerr << "Invalid WarpCTC version." << std::endl;
        return 1;
    }

    std::cout << "Running GPU tests" << std::endl;
    throw_on_error(cudaSetDevice(0), "cudaSetDevice");

    bool status = true;
    status &= small_test();
    status &= options_test();
    status &= inf_test();
    status &= run_tests();

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}
