cdef extern from "ctc.h":
    struct ctcOptions:
        CTCComputeLocation loc
        Stream stream
        unsigned int num_threads
        int blank_label

    CTCStatus compute_ctc_loss(float* activations,
                             float* gradients,
                             int* flat_labels,
                             int* label_lengths,
                             int* input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float* costs,
                             void* workspace,
                             ctcOptions info)

    CTCStatus get_workspace_size(int* label_lengths,
                           int* input_lengths,
                           int alphabet_size,
                           int minibatch,
                           ctcOptions info,
                           size_t* size_bytes)

    const char* ctcGetStatusString(CTCStatus status)


cdef dict STATUS = {
    0: 'CTC_STATUS_SUCCESS',
    1: 'CTC_STATUS_MEMOPS_FAILED',
    2: 'CTC_STATUS_INVALID_VALUE',
    3: 'CTC_STATUS_EXECUTION_FAILED',
    4: 'CTC_STATUS_UNKNOWN_ERROR',
}


class CTCError(RuntimeError):
    def __init__(self, CTCStatus status):
        self.status = status
        msg = ctcGetStatusString(<CTCStatus>status)
        super(CTCError, self).__init__('%s: %s' % (STATUS[status], msg))


cpdef inline check_status(int status):
    if status != 0:
        raise CTCError(status)


cpdef void ctc_get_workspace_size_gpu(size_t label_lengths,
                                   size_t input_lengths,
                                   int alphabet_size, int minibatch,
                                   size_t size_bytes,
                                   size_t stream):
    cdef ctcOptions ctc_options
    ctc_options.loc = <CTCComputeLocation> CTC_GPU
    ctc_options.stream = <Stream> stream
    ctc_options.blank_label = 0
    status = get_workspace_size(
        <int*> label_lengths, <int*> input_lengths,
        alphabet_size, minibatch, ctc_options,
        <size_t*> size_bytes
    )
    check_status(status)


cpdef void ctc_compute_ctc_loss_gpu(size_t activations,
                                 size_t gradients,
                                 size_t flat_labels,
                                 size_t label_lengths,
                                 size_t input_lengths,
                                 int alphabet_size,
                                 int minibatch,
                                 size_t costs,
                                 size_t workspace,
                                 size_t stream) except *:
    cdef ctcOptions ctc_options
    ctc_options.loc = <CTCComputeLocation> CTC_GPU
    ctc_options.stream = <Stream> stream
    ctc_options.blank_label = 0
    status = compute_ctc_loss(
        <float*> activations, <float*> gradients, <int*> flat_labels,
        <int*> label_lengths, <int*> input_lengths, alphabet_size,
        minibatch, <float*> costs, <void*> workspace,
        ctc_options
    )
    check_status(status)


cpdef void ctc_get_workspace_size_cpu(size_t label_lengths,
                                   size_t input_lengths,
                                   int alphabet_size, int minibatch,
                                   size_t size_bytes):
    cdef ctcOptions ctc_options
    ctc_options.loc = <CTCComputeLocation> CTC_CPU
    ctc_options.num_threads = 1
    ctc_options.blank_label = 0
    status = get_workspace_size(
        <int*> label_lengths, <int*> input_lengths,
        alphabet_size, minibatch, ctc_options,
        <size_t*> size_bytes
    )
    check_status(status)


cpdef void ctc_compute_ctc_loss_cpu(size_t activations,
                                 size_t gradients,
                                 size_t flat_labels,
                                 size_t label_lengths,
                                 size_t input_lengths,
                                 int alphabet_size,
                                 int minibatch,
                                 size_t costs,
                                 size_t workspace,
                                 unsigned int num_threads) except *:
    cdef ctcOptions ctc_options
    ctc_options.loc = <CTCComputeLocation> CTC_CPU
    ctc_options.num_threads = 1
    ctc_options.blank_label = 0
    status = compute_ctc_loss(
        <float*> activations, <float*> gradients, <int*> flat_labels,
        <int*> label_lengths, <int*> input_lengths, alphabet_size,
        minibatch, <float*> costs, <void*> workspace,
        ctc_options
    )
    check_status(status)
