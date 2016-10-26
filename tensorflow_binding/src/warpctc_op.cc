#ifdef WARPCTC_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cuda.h>
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"
#include "ctc.h"


REGISTER_OP("WarpCTC")
    .Input("activations: float32")
    .Input("flat_labels: int32")
    .Input("label_lengths: int32")
    .Input("input_lengths: int32")
    .Attr("blank_label: int = 0")
    .Output("costs: float32")
    .Output("gradients: float32");

namespace tf = tensorflow;

namespace warp_ctc {

class WarpCTCOpBase : public tf::OpKernel {
  public:
    explicit WarpCTCOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
    }

    void Compute(tf::OpKernelContext* ctx) override {
        // Grab the input tensors
        const tf::Tensor* activations;
        const tf::Tensor* flat_labels;
        const tf::Tensor* label_lengths;
        const tf::Tensor* input_lengths;
        OP_REQUIRES_OK(ctx, ctx->input("activations", &activations));
        OP_REQUIRES_OK(ctx, ctx->input("flat_labels", &flat_labels));
        OP_REQUIRES_OK(ctx, ctx->input("label_lengths", &label_lengths));
        OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths));

        OP_REQUIRES(ctx, activations->shape().dims() == 3,
                    tf::errors::InvalidArgument("activations is not a 3-Tensor"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(flat_labels->shape()),
                     tf::errors::InvalidArgument("flat_labels is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(label_lengths->shape()),
                     tf::errors::InvalidArgument("label_lengths is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lengths->shape()),
                     tf::errors::InvalidArgument("input_lengths is not a vector"));

        const auto& acts_shape = activations->shape();
        const auto max_time = acts_shape.dim_size(0);
        const auto batch_size = acts_shape.dim_size(1);
        const auto num_classes_raw = acts_shape.dim_size(2);

        auto activations_t = activations->tensor<float, 3>();
        auto flat_labels_t = flat_labels->vec<int32_t>();

        OP_REQUIRES(
                ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
                tf::errors::InvalidArgument("num_classes cannot exceed max int"));
        const auto alphabet_size = static_cast<const int>(num_classes_raw);

        OP_REQUIRES(
                ctx, batch_size == input_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(input_lengths) != batch_size.  ",
                                            "len(input_length):  ", input_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto input_lengths_t = input_lengths->vec<int32_t>();

        OP_REQUIRES(
                ctx, batch_size == label_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(label_lengths) != batch_size.  ",
                                            "len(label_length):  ", label_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto label_lengths_t = label_lengths->vec<int32_t>();

        // check that labels are in the alphabet?

        for (int b = 0; b < batch_size; b++) {
            OP_REQUIRES(ctx, input_lengths_t(b) <= max_time,
                        tf::errors::InvalidArgument("input_lengths(", b, ") <= ", max_time));
        }

        tf::Tensor* costs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", input_lengths->shape(), &costs));
        auto costs_t = costs->vec<float>();

        tf::Tensor* grads = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("gradients", activations->shape(),
                                                 &grads));
        set_zero(grads);
        auto grads_t = grads->tensor<float, 3>();

        auto options = create_options(ctx);
        options.blank_label = blank_label_;

        size_t workspace_size_bytes;
        auto warp_status = get_workspace_size(label_lengths_t.data(),
                                              input_lengths_t.data(),
                                              alphabet_size, batch_size,
                                              options, &workspace_size_bytes);

        OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                    tf::errors::Internal("warp_ctc error in get_workspace_size: ",
                                         ctcGetStatusString(warp_status)));

        auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
        tf::Tensor workspace;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
        auto workspace_t = workspace.flat<uint8_t>();

        // compute CTC
        warp_status = compute_ctc_loss(activations_t.data(),
                                       grads_t.data(),
                                       flat_labels_t.data(),
                                       label_lengths_t.data(),
                                       input_lengths_t.data(),
                                       alphabet_size, batch_size,
                                       costs_t.data(), workspace_t.data(), options);

        OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                    tf::errors::Internal("warp_ctc error in compute_ctc_loss: ",
                                         ctcGetStatusString(warp_status)));

    }
  private:
    int blank_label_;
    virtual void set_zero(tf::Tensor* t) = 0;
    virtual ctcOptions create_options(tf::OpKernelContext* ctx) = 0;
};

class WarpCTCOpCPU : public WarpCTCOpBase {
  public:
    explicit WarpCTCOpCPU(tf::OpKernelConstruction* ctx) : WarpCTCOpBase(ctx) {
    }

  private:
    void set_zero(tf::Tensor* t) override {
        t->flat<float>().setZero();
    }

    ctcOptions create_options(tf::OpKernelContext* ctx) override {
        auto options = ctcOptions{};
        options.loc = CTC_CPU;
        options.num_threads = ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(::tensorflow::DEVICE_CPU), WarpCTCOpCPU);

#ifdef WARPCTC_ENABLE_GPU

class WarpCTCOpGPU : public WarpCTCOpBase {
  public:
    explicit WarpCTCOpGPU(tf::OpKernelConstruction* ctx) : WarpCTCOpBase(ctx) {
    }

  private:
    void set_zero(tf::Tensor* t) override {
        cudaMemset(t->flat<float>().data(), 0, t->NumElements()*sizeof(float));
    }

    ctcOptions create_options(tf::OpKernelContext* ctx) override {
        auto cuda_stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        auto options = ctcOptions{};
        options.loc = CTC_GPU;
        options.stream = cuda_stream;
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("WarpCTC").Device(::tensorflow::DEVICE_GPU)
                        .HostMemory("flat_labels")
                        .HostMemory("label_lengths")
                        .HostMemory("input_lengths")
                        .HostMemory("costs"),
                        WarpCTCOpGPU);
#undef EIGEN_USE_GPU
#endif

}

