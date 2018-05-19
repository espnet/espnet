#ifdef WARPCTC_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cuda.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "ctc.h"

namespace tf = tensorflow;

namespace warp_ctc {

class CTCLossOpBase : public tf::OpKernel {
  public:
    explicit CTCLossOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
        bool preprocess_collapse_repeated;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                         &preprocess_collapse_repeated));
        OP_REQUIRES(ctx, preprocess_collapse_repeated == false,
                    tf::errors::InvalidArgument("preprocess collapse repeated is not currently "
                                                "supported in the WarpCTC kernel."));

        bool ctc_merge_repeated;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ctc_merge_repeated", &ctc_merge_repeated));
        OP_REQUIRES(ctx, ctc_merge_repeated == true,
                    tf::errors::InvalidArgument("ctc_merge_repeated == false is not currently "
                                                "supported. WarpCTC always merges repeated symbols."));
    }

    void Compute(tf::OpKernelContext* ctx) override {
        const tf::Tensor* inputs;
        const tf::Tensor* labels_indices;
        const tf::Tensor* labels_values;
        const tf::Tensor* seq_len;
        OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
        OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
        OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
        OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));

        OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                    tf::errors::InvalidArgument("inputs is not a 3-Tensor"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(seq_len->shape()),
                    tf::errors::InvalidArgument("sequence_length is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(labels_indices->shape()),
                    tf::errors::InvalidArgument("labels_indices is not a matrix"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(labels_values->shape()),
                    tf::errors::InvalidArgument("labels_values is not a vector"));

        const auto& inputs_shape = inputs->shape();
        const auto max_time = inputs_shape.dim_size(0);
        const auto batch_size = inputs_shape.dim_size(1);
        const auto num_classes_raw = inputs_shape.dim_size(2);
        OP_REQUIRES(
                ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
                tf::errors::InvalidArgument("num_classes cannot exceed max int"));
        const auto num_classes = static_cast<const int>(num_classes_raw);

        OP_REQUIRES(
                ctx, batch_size == seq_len->dim_size(0),
                tf::errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                            "len(sequence_length):  ", seq_len->dim_size(0),
                                            " batch_size: ", batch_size));
        auto seq_len_t = seq_len->vec<int32_t>();

        OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                    tf::errors::InvalidArgument(
                            "labels_indices and labels_values must contain the "
                            "same number of rows, but saw shapes: ",
                            labels_indices->shape().DebugString(), " vs. ",
                            labels_values->shape().DebugString()));

        auto labels_shape = tf::TensorShape({batch_size, max_time});
        auto order = std::vector<tf::int64>{0, 1};
        auto labels_sp = tf::sparse::SparseTensor(*labels_indices, *labels_values,
                                                  labels_shape, order);

        auto labels_sp_valid = labels_sp.IndicesValid();
        OP_REQUIRES(ctx, labels_sp_valid.ok(),
                    tf::errors::InvalidArgument("label SparseTensor is not valid: ",
                                            labels_sp_valid.error_message()));

        auto label_lengths = std::vector<int>{};
        for (const auto& g : labels_sp.group({0})) {  // iterate by batch
            const auto batch_indices = g.group()[0];
            OP_REQUIRES(ctx, tf::FastBoundsCheck(batch_indices, batch_size),
                        tf::errors::InvalidArgument("labels batch index must be between ",
                                                    0, " and ", batch_size, " but saw: ",
                                                    batch_indices));
            
            auto values = g.values<int32_t>();
            label_lengths.push_back(values.size());
        }
        auto label_values_t = labels_values->vec<int>();


        OP_REQUIRES(ctx, static_cast<size_t>(batch_size) == label_lengths.size(),
                    tf::errors::InvalidArgument("len(labels) != batch_size.  ",
                                                "len(labels):  ", label_lengths.size(),
                                                " batch_size: ", batch_size));

        for (int b = 0; b < batch_size; ++b) {
            OP_REQUIRES(
                    ctx, seq_len_t(b) <= max_time,
                    tf::errors::InvalidArgument("sequence_length(", b, ") <= ", max_time));
        }

        tf::Tensor* loss = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len->shape(), &loss));
        auto loss_t = loss->vec<float>();

        tf::Tensor* gradient;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output("gradient", inputs_shape, &gradient));
        set_zero(gradient);
        auto gradient_t = gradient->tensor<float, 3>();

        auto inputs_t = inputs->tensor<float, 3>();

        auto options = create_options(ctx);
        options.blank_label = num_classes - 1;

        size_t workspace_size_bytes;
        auto warp_status = get_workspace_size(label_lengths.data(), seq_len_t.data(),
                                              num_classes, batch_size,
                                              options, &workspace_size_bytes);
        OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                    tf::errors::Internal("warp_ctc error in get_workspace_size: ",
                                         ctcGetStatusString(warp_status)));

        auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
        tf::Tensor workspace;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
        auto workspace_t = workspace.flat<uint8_t>();

        warp_status = compute_ctc_loss(inputs_t.data(),
                                       gradient_t.data(),
                                       label_values_t.data(),
                                       label_lengths.data(),
                                       seq_len_t.data(),
                                       num_classes, batch_size,
                                       loss_t.data(), workspace_t.data(), options);
        
        OP_REQUIRES(ctx, warp_status == CTC_STATUS_SUCCESS,
                    tf::errors::Internal("warp_ctc error in compute_ctc_loss: ",
                                         ctcGetStatusString(warp_status)));

    }

  private:
    virtual void set_zero(tf::Tensor* t) = 0;
    virtual ctcOptions create_options(tf::OpKernelContext* ctx) = 0;
};

class CTCLossOpCPU : public CTCLossOpBase {
  public:
    explicit CTCLossOpCPU(tf::OpKernelConstruction* ctx) : CTCLossOpBase(ctx) {
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

REGISTER_KERNEL_BUILDER(Name("CTCLoss")
                        .Device(::tensorflow::DEVICE_CPU)
                        .Label("WarpCTC"),
                        CTCLossOpCPU);

#ifdef WARPCTC_ENABLE_GPU

class CTCLossOpGPU : public CTCLossOpBase {
  public:
    explicit CTCLossOpGPU(tf::OpKernelConstruction* ctx) : CTCLossOpBase(ctx) {
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

// Register GPU kernel both with and without the label
REGISTER_KERNEL_BUILDER(Name("CTCLoss")
                        .Device(::tensorflow::DEVICE_GPU)
                        .Label("WarpCTC")
                        .HostMemory("labels_indices")
                        .HostMemory("labels_values")
                        .HostMemory("sequence_length")
                        .HostMemory("loss"),
                        CTCLossOpGPU);
REGISTER_KERNEL_BUILDER(Name("CTCLoss")
                        .Device(::tensorflow::DEVICE_GPU)
                        .HostMemory("labels_indices")
                        .HostMemory("labels_values")
                        .HostMemory("sequence_length")
                        .HostMemory("loss"),
                        CTCLossOpGPU);

#undef EIGEN_USE_GPU
#endif

}
