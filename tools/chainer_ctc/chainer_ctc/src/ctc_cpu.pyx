import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange

DTYPE = np.float32
ITYPE = np.int
ctypedef np.float32_t DTYPE_t
ctypedef np.int_t ITYPE_t

DEF log_zero_ = -1e30
DEF exp_limit_ = 88.722839
DEF log_inf_ = 1e30
DEF max_ = 3.4028235e38

cdef extern from "math.h":
    float exp(float x) nogil
    float log(float x) nogil

cdef DTYPE_t c_log_mul(DTYPE_t a, DTYPE_t b) nogil:
    if a == log_zero_ or b == log_zero_:
        return log_zero_
    else:
        return a + b

cdef DTYPE_t c_log_div(DTYPE_t a, DTYPE_t b) nogil:
    if a == log_zero_:
        return log_zero_
    elif b == log_zero_:
        return log_inf_
    else:
        return a - b

cdef DTYPE_t c_safe_exp(DTYPE_t a) nogil:
    if a <= log_zero_:
        return 0
    elif a >= exp_limit_:
        return max_
    else:
        return exp(a)

cdef DTYPE_t c_log_add(DTYPE_t a, DTYPE_t b) nogil:
    if b < a:
        return c_log_mul(a, log(1 + c_safe_exp(c_log_div(b, a))))
    else:
        return c_log_mul(b, log(1 + c_safe_exp(c_log_div(a, b))))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def c_calc_alpha(np.ndarray[int, ndim=1] target_sequence,
                 np.ndarray[DTYPE_t, ndim=2] log_label_probs,
                 int blank_symbol):
    cdef int T = log_label_probs.shape[0]
    cdef int L = target_sequence.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] alpha = np.empty((T, L), dtype=DTYPE)
    cdef size_t t, l
    cdef int id
    for t in range(T):
        for l in prange(L, nogil=True):
            id = target_sequence[l]
            if t == 0:
                if l < 2:
                    alpha[t, l] = log_label_probs[t, id]
                else:
                    alpha[t, l] = log_zero_
            else:
                if l > 1:
                    if l % 2 == blank_symbol or target_sequence[l-2] == id:
                        alpha[t, l] = c_log_mul(
                            log_label_probs[t, id],
                            c_log_add(alpha[t-1, l], alpha[t-1, l-1])
                        )
                    else:
                        alpha[t, l] = c_log_mul(
                            log_label_probs[t, id],
                            c_log_add(
                                c_log_add(alpha[t-1, l-2], alpha[t-1, l-1]),
                                alpha[t-1, l]
                            )
                        )
                elif l == 1:
                    alpha[t, l] = c_log_mul(log_label_probs[t, id],
                                            c_log_add(alpha[t-1, l-1],
                                                      alpha[t-1, l])
                                            )
                else:
                    alpha[t, l] = c_log_mul(log_label_probs[t, id],
                                            alpha[t-1, l])
    return alpha

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def c_calc_beta(np.ndarray[int, ndim=1] target_sequence,
                np.ndarray[DTYPE_t, ndim=2] log_label_probs,
                int blank_symbol):
    cdef int T = log_label_probs.shape[0]
    cdef int L = target_sequence.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] beta = np.empty((T, L), dtype=DTYPE)
    cdef size_t t, l
    cdef int id
    for t in range(T-1, -1, -1):
        for l in prange(L, nogil=True):
            id = target_sequence[l]
            if t == T-1:
                if l > L - 3:
                    beta[t, l] = log_label_probs[t, id]
                else:
                    beta[t, l] = log_zero_
            else:
                if l < L - 2:
                    if l % 2 == blank_symbol or target_sequence[l+2] == id:
                        beta[t, l] = c_log_mul(
                            log_label_probs[t, id],
                            c_log_add(beta[t+1, l], beta[t+1, l+1])
                        )
                    else:
                        beta[t, l] = c_log_mul(
                            log_label_probs[t, id],
                            c_log_add(
                                c_log_add(beta[t+1, l+2], beta[t+1, l+1]),
                                beta[t+1, l]
                            )
                        )
                elif l == L - 2:
                    beta[t, l] = c_log_mul(log_label_probs[t, id],
                                           c_log_add(beta[t+1, l+1],
                                                     beta[t+1, l])
                                           )
                else:
                    beta[t, l] = c_log_mul(log_label_probs[t, id], beta[t+1, l])
    return beta

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def c_calc_label_grads(np.ndarray[DTYPE_t, ndim=2] alpha,
                       np.ndarray[DTYPE_t, ndim=2] beta,
                       np.ndarray[DTYPE_t, ndim=2] log_label_probs,
                       np.ndarray[int, ndim=1] target_sequence):
    cdef int T = log_label_probs.shape[0]
    cdef int IDS = log_label_probs.shape[1]
    cdef int L = target_sequence.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] label_grads = np.zeros((T, IDS), dtype=DTYPE)
    cdef size_t t, l
    cdef int id
    cdef DTYPE_t norm
    for t in prange(T, nogil=True):
        norm = log_zero_
        for l in range(L):
            id = target_sequence[l]
            norm = c_log_add(norm,
                             c_log_div(
                                 c_log_mul(alpha[t, l], beta[t, l]),
                                 log_label_probs[t, id]
                             ))
        for l in range(L):
            id = target_sequence[l]
            label_grads[t, id] += c_safe_exp(
                c_log_div(
                    c_log_div(
                        c_log_mul(alpha[t, l], beta[t, l]),
                        log_label_probs[t, id]
                    ),
                    norm)
            )
    return label_grads

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def c_calc_ll(alpha):
    cdef DTYPE_t ll = 0
    cdef int batches = len(alpha)
    for b in range(batches):
        ll -= c_log_add(alpha[b][-1, -1], alpha[b][-1, -2])
    return <DTYPE_t> ll / batches


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def c_best_path(np.ndarray[DTYPE_t, ndim=2] obs):
    cdef int T = obs.shape[0]
    cdef int S = obs.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] trellis = np.zeros((T, S), dtype=DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=2] backpt = np.zeros((T, S), dtype=ITYPE)
    cdef size_t t
    cdef ITYPE_t s
    cdef DTYPE_t norm
    for s in range(S):
        trellis[0, s] = obs[0, s]
    for t in range(1, T):
        trellis[t, 0] = c_log_mul(trellis[t-1, 0], obs[t, 0])
        for s in prange(S, nogil=True):
            if s == 0 or trellis[t-1, s-1] < trellis[t-1, s]:
                trellis[t, s] = c_log_mul(trellis[t-1, s], obs[t, s])
                backpt[t, s] = s
            else:
                trellis[t, s] = c_log_mul(trellis[t-1, s-1], obs[t, s])
                backpt[t, s] = s-1
    cdef np.ndarray[ITYPE_t, ndim=1] tokens = np.zeros(T, dtype=ITYPE)
    tokens[T-1] = S-1
    for t in range(T-2, -1, -1):
        tokens[t] = backpt[t+1, tokens[t+1]]
    return tokens