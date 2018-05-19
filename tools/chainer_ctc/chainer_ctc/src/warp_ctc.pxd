###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int CTCStatus 'ctcStatus_t'
    ctypedef int CTCComputeLocation 'ctcComputeLocation'
    ctypedef void* Stream 'struct CUstream_st*'

###############################################################################
# Enum
###############################################################################

cpdef enum:
    CTC_STATUS_SUCCESS = 0
    CTC_STATUS_MEMOPS_FAILED = 1
    CTC_STATUS_INVALID_VALUE = 2
    CTC_STATUS_EXECUTION_FAILED = 3
    CTC_STATUS_UNKNOWN_ERROR = 4

    CTC_CPU = 0
    CTC_GPU = 1

###############################################################################
# Functions
###############################################################################

cpdef void ctc_get_workspace_size_gpu(size_t label_lengths,
                                   size_t input_lengths,
                                   int alphabet_size, int minibatch,
                                   size_t size_bytes,
                                   size_t stream);

cpdef void ctc_compute_ctc_loss_gpu(size_t activations,
                                 size_t gradients,
                                 size_t flat_labels,
                                 size_t label_lengths,
                                 size_t input_lengths,
                                 int alphabet_size,
                                 int minibatch,
                                 size_t costs,
                                 size_t workspace,
                                 size_t stream) except *;

cpdef void ctc_get_workspace_size_cpu(size_t label_lengths,
                                   size_t input_lengths,
                                   int alphabet_size, int minibatch,
                                   size_t size_bytes);

cpdef void ctc_compute_ctc_loss_cpu(size_t activations,
                                 size_t gradients,
                                 size_t flat_labels,
                                 size_t label_lengths,
                                 size_t input_lengths,
                                 int alphabet_size,
                                 int minibatch,
                                 size_t costs,
                                 size_t workspace,
                                 unsigned int num_threads) except *;
