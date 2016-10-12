#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"

#include "ctc.h"


REGISTER_OP("WarpCTC")
    .Input("data: float32")
    .Input("data_lengths: int32")
    .Input("flat_labels: int32")
    .Input("label_lengths: int32")
    .Attr("alphabet_size: int")
    .Output("loss: float32")
    .Output("gradient: float32");

using namespace tensorflow;

class WarpCTCOpCPU : public OpKernel {
 public:
  explicit WarpCTCOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("alphabet_size", &alphabet_size_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& data_t = context->input(0);
    const Tensor& data_lens_t = context->input(1);
    const Tensor& labels_t = context->input(2);
    const Tensor& label_lens_t = context->input(3);
    auto data = data_t.flat<float>();
    auto data_lens = data_lens_t.flat<int>();
    auto labels = labels_t.flat<int>();
    auto label_lens = label_lens_t.flat<int>();
    int alphabet_size = alphabet_size_;
    int n_minibatches = data_t.dim_size(1);

    auto options = ctcOptions{};
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = context->device()->tensorflow_cpu_worker_threads()->num_threads;

    size_t cpu_alloc_bytes;
    ctcStatus_t stat_alloc = get_workspace_size(label_lens.data(), data_lens.data(),
                                          alphabet_size, n_minibatches, options,
                                          &cpu_alloc_bytes);

    OP_REQUIRES(context, (stat_alloc == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC memory estimation"))

    // allocate scratch space for ctc computation
    Allocator* a = context->device()->GetAllocator(AllocatorAttributes());

    void* scratch = a->AllocateRaw(1, cpu_alloc_bytes);

    // allocate gradient tensor
    Tensor* gradients = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, data_t.shape(),
                                                     &gradients));
    auto grads = gradients->flat<float>();

    // compute CTC
    std::vector<float> costs(n_minibatches);
    ctcStatus_t stat_compute = compute_ctc_loss(data.data(),
                                                grads.data(),
                                                labels.data(),
                                                label_lens.data(),
                                                data_lens.data(),
                                                alphabet_size,
                                                n_minibatches,
                                                costs.data(),
                                                scratch,
                                                options);
    // std::raise(SIGINT);

    a->DeallocateRaw(scratch);

    OP_REQUIRES(context, (stat_compute == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC computation"))

    Tensor* loss_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n_minibatches}), &loss_t));
    auto loss = loss_t->flat<float>();
    for (int i = 0; i < n_minibatches; ++i) {
      loss(i) = costs[i];
    }
  }
 private:
  int alphabet_size_;
};

class WarpCTCOpGPU : public OpKernel {
 public:
  explicit WarpCTCOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("alphabet_size", &alphabet_size_));
  }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& data_t = ctx->input(0);
    const Tensor& data_lens_t = ctx->input(1);
    const Tensor& labels_t = ctx->input(2);
    const Tensor& label_lens_t = ctx->input(3);
    auto data = data_t.flat<float>();
    auto data_lens = data_lens_t.flat<int32>();
    auto labels = labels_t.flat<int32>();
    auto label_lens = label_lens_t.flat<int32>();
    int alphabet_size = alphabet_size_;
    int n_minibatches = data_t.dim_size(1);

    auto cuda_stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
    auto options = ctcOptions{};
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.stream = cuda_stream;

    size_t workspace_size_bytes;
    ctcStatus_t stat_alloc = get_workspace_size(label_lens.data(), data_lens.data(),
                                                alphabet_size, n_minibatches, options,
                                                &workspace_size_bytes);
    OP_REQUIRES(ctx, (stat_alloc == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC memory estimation"));

    //  allocate scratch space for ctc computation
    auto workspace_shape = TensorShape{static_cast<int64_t>(workspace_size_bytes)};
    Tensor workspace;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_UINT8, workspace_shape, &workspace));
    auto workspace_t = workspace.flat<uint8_t>();

    // allocate gradient tensor
    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", data_lens_t.shape(), &loss));
    auto loss_t = loss->vec<float>();
    
    Tensor* gradient;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gradient", data_t.shape(), &gradient));
    auto gradient_t = gradient->tensor<float, 3>();
    cudaMemset(gradient_t.data(), 0, gradient->NumElements()*sizeof(float));

    ctcStatus_t stat_compute = compute_ctc_loss(data.data(),
                                                gradient_t.data(),
                                                labels.data(),
                                                label_lens.data(),
                                                data_lens.data(),
                                                alphabet_size,
                                                n_minibatches,
                                                loss_t.data(),
                                                workspace_t.data(),
                                                options);

    OP_REQUIRES(ctx, (stat_compute == CTC_STATUS_SUCCESS),
                errors::Internal("Error in CTC computation"));
  }
 private:
  int alphabet_size_;
};

#undef EIGEN_USE_GPU

REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(DEVICE_CPU), WarpCTCOpCPU);
REGISTER_KERNEL_BUILDER(Name("WarpCTC")
                        .Device(DEVICE_GPU)
                        .HostMemory("flat_labels")
                        .HostMemory("label_lengths")
                        .HostMemory("data_lengths")
                        .HostMemory("loss"),
                        WarpCTCOpGPU);
