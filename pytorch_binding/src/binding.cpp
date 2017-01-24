#include <iostream>
#include <vector>

#include <numeric>

#include "ctc.h"

#ifdef TORCH_NOGPU
    #include "TH.h"
#else
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
#endif

extern THCState* state;

extern "C" int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THIntTensor *labels,
                        THIntTensor *label_sizes,
                        THIntTensor *sizes,
                        int minibatch_size,
                        THFloatTensor *costs) {

    float *probs_ptr = probs->storage->data + probs->storageOffset;
    float *grads_ptr;
    if (grads->storage) {
            grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
            grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = sizes->storage->data + sizes->storageOffset;
    int *labels_ptr = labels->storage->data + labels->storageOffset;
    int *label_sizes_ptr = label_sizes->storage->data + label_sizes->storageOffset;
    float *costs_ptr = costs->storage->data + costs->storageOffset;

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       (int) probs->size[2], minibatch_size,
                       options, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs->size[2],
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete cpu_workspace;
    return 1;
}

extern "C" int gpu_ctc(THCudaTensor *probs,
                        THCudaTensor *grads,
                        THIntTensor *labels,
                        THIntTensor *label_sizes,
                        THIntTensor *sizes,
                        int minibatch_size,
                        THFloatTensor *costs) {

    float *probs_ptr = probs->storage->data + probs->storageOffset;
    float *grads_ptr;
    if (grads->storage) {
            grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
            grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = sizes->storage->data + sizes->storageOffset;
    int *labels_ptr = labels->storage->data + labels->storageOffset;
    int *label_sizes_ptr = label_sizes->storage->data + label_sizes->storageOffset;
    float *costs_ptr = costs->storage->data + costs->storageOffset;

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.stream = THCState_getCurrentStream(state);

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       (int) probs->size[2], minibatch_size,
                       options, &gpu_size_bytes);

    float* gpu_workspace;
    THCudaMalloc(state, (void **) &gpu_workspace, gpu_size_bytes);

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs->size[2],
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, (void *) gpu_workspace);
    return 1;
}
