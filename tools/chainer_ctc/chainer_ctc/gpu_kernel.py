import numpy as np
from chainer import cuda

if cuda.available:
    import cupy

preamble = """
static const float log_zero_ = {log_zero};
static const float exp_limit_ = {exp_limit};
static const float log_inf_ = {log_inf};
static const float max_ = {float_max};
""".format(log_zero=-1e30,
           exp_limit=np.log(np.finfo(np.float32).max),
           log_inf=1e30,
           float_max=np.finfo(np.float32).max)

preamble += """
static inline __host__ __device__ T log_mul(T a, T b)
{
  if (a == log_zero_ || b == log_zero_)
    return log_zero_;
  else
    return a + b;
}

static inline __host__ __device__ T log_div(T a, T b)
{
  if (a == log_zero_)
    return log_zero_;
  else if (b == log_zero_)
    return log_inf_;
  else
    return a - b;
}

static inline __host__ __device__ T safe_exp(T a)
{
  if (a <= log_zero_)
    return 0;
  else if (a >= exp_limit_)
    return max_;
  else
    return exp(a);
}

static inline __host__ __device__ T log_add(T a, T b)
{
if (b < a)
  return log_mul(a, log(1 + safe_exp(log_div(b, a))));
else
  return log_mul(b, log(1 + safe_exp(log_div(a, b))));
}"""

if cuda.available:
    calc_alpha_kernel = cuda.elementwise(
        'raw I target_sequence, raw T log_label_probs, int32 t, int32 classes,'
        'int32 blank_symbol',
        'raw T alpha',
        """
int idx_alpha = t*_ind.size() + i; // index for alpha
int idx_tm1_alpha = (t-1)*_ind.size() + i; // index for alpha in last time step
int id = target_sequence[i]; // class id for label prob
int idx_probs = t*classes + id; //idx for label prob
if(t == 0)
{
    if (i < 2) {
        alpha[idx_alpha] = log_label_probs[idx_probs];
        }
    else {
        alpha[idx_alpha] = log_zero_;
        }
} else {
    if (i > 1) {
        if (i % 2 == blank_symbol || target_sequence[i-2] == target_sequence[i])
        {
            alpha[idx_alpha] = log_mul(log_label_probs[idx_probs],
                log_add(alpha[idx_tm1_alpha], alpha[idx_tm1_alpha-1]));
        } else {
            alpha[idx_alpha] = log_mul(log_label_probs[idx_probs], log_add(
                log_add(alpha[idx_tm1_alpha-2], alpha[idx_tm1_alpha-1]),
                alpha[idx_tm1_alpha]));
        }
    } else if (i == 1) {
        alpha[idx_alpha] = log_mul(log_label_probs[idx_probs],
            log_add(alpha[idx_tm1_alpha-1], alpha[idx_tm1_alpha]));
    } else {
        alpha[idx_alpha] = log_mul(log_label_probs[idx_probs],
            alpha[idx_tm1_alpha]);
    }
}
        """, 'ctc_calc_alpha', preamble=preamble
    )

if cuda.available:
    calc_beta_kernel = cuda.elementwise(
        'raw I target_sequence, raw T log_label_probs, int32 t, int32 t_max,'
        'int32 classes, int32 blank_symbol',
        'raw T beta',
        """
int idx_beta = t*_ind.size() + i; // index for beta
int idx_tp1_beta = (t+1)*_ind.size() + i; // index for beta in previous time step
int id = target_sequence[i]; // class id for label prob
int idx_probs = t*classes + id; //idx for label prob

if(t == t_max)
{
    if (i > _ind.size() - 3) {
        beta[idx_beta] = log_label_probs[idx_probs];
        }
    else {
        beta[idx_beta] = log_zero_;
        }
} else {
    if (i < _ind.size() - 2) {
        if (i % 2 == blank_symbol || target_sequence[i+2] == id)
        {
            beta[idx_beta] = log_mul(log_label_probs[idx_probs],
                log_add(beta[idx_tp1_beta], beta[idx_tp1_beta+1]));
        } else {
            beta[idx_beta] = log_mul(log_label_probs[idx_probs], log_add(
                log_add(beta[idx_tp1_beta+2], beta[idx_tp1_beta+1]),
                beta[idx_tp1_beta]));
        }
    } else if (i == _ind.size() - 2) {
        beta[idx_beta] = log_mul(log_label_probs[idx_probs],
            log_add(beta[idx_tp1_beta+1], beta[idx_tp1_beta]));
    } else {
        beta[idx_beta] = log_mul(log_label_probs[idx_probs],
            beta[idx_tp1_beta]);
    }
}
        """, 'ctc_calc_beta', preamble=preamble
    )

if cuda.available:
    calc_label_grad_kernel = cuda.elementwise(
        'raw T alpha, raw T beta, raw T log_label_probs, raw I target_sequence,'
        'int32 labels, int32 classes',
        'raw T label_grads',
        """
T norm = log_zero_;  // This will accumulate the normalization constant
int l_offset = i * labels; // Offset for the matrices with labels
int c_offset = i * classes; // Offset for the matrices with classes
int l;
for (l=0; l < labels; l++) {
    int id = target_sequence[l]; // Class id for current label
    norm = log_add(norm, log_div(log_mul(alpha[l_offset+l], beta[l_offset+l]),
        log_label_probs[c_offset+id]));
}
for (l=0; l < labels; l++) {
    int id = target_sequence[l];
    label_grads[c_offset+id] += safe_exp(
        log_div(
            log_div(
                log_mul(alpha[l_offset+l], beta[l_offset+l]),
                log_label_probs[c_offset+id]
            ),
            norm)
        );
    }
        """, 'ctc_calc_label_grads', preamble=preamble
    )

if cuda.available:
    calc_ll_kernel = cuda.elementwise(
        'raw T alpha, int32 offset, int32 batch_no',
        'raw T ll',
        """
ll[batch_no] = -log_add(alpha[(i+1)*offset-1], alpha[(i+1)*offset-2])
        """, 'ctc_ll', preamble=preamble
    )


def calc_alpha(target_sequence, log_label_probs, blank_symbol):
    """ Calculates the alpha for CTC

    :param target_sequence: The target sequence WITH blanks encoded using ints
    :param log_label_probs: The log probability for each target for each frame.
        Dimension is TxL where T is the time and L are the number of labels
    :return: alpha
    """

    L = target_sequence.shape[0]
    T = log_label_probs.shape[0]
    classes = log_label_probs.shape[1]
    # Prepare kernel
    alpha = cupy.empty((T, L), dtype=np.float32)
    for t in range(T):
        calc_alpha_kernel(target_sequence, log_label_probs, t, classes,
                          blank_symbol, alpha, size=L)
    return alpha


def calc_beta(target_sequence, log_label_probs, blank_symbol):
    """ Calculates the beta for CTC

    :param target_sequence: The target sequence WITH blanks encoded using ints
    :param log_label_probs: The log probability for each target for each frame.
        Dimension is TxL where T is the time and L are the number of labels
    :return: beta
    """

    L = target_sequence.shape[0]
    T = log_label_probs.shape[0]
    classes = log_label_probs.shape[1]
    # Prepare kernel
    beta = cupy.empty((T, L), dtype=np.float32)

    for t in range(T - 1, -1, -1):
        calc_beta_kernel(target_sequence, log_label_probs, t, T - 1,
                         classes, blank_symbol, beta, size=L)
    return beta


def calc_label_grads(alpha, beta, log_label_probs, target_sequence):
    """ Calculates the gradient for the labels.

    .. note:: To get the final gradient, this has to be substracted from the
        output of the network.

    :param alpha: forward variable
    :param beta: backward variable
    :param log_label_probs: Log probabilities
    :param target_sequence: The target label sequence
    :return: Gradients for the labels
    """

    labels = target_sequence.shape[0]
    classes = log_label_probs.shape[1]
    T = log_label_probs.shape[0]
    label_grads = cupy.zeros((T, classes), dtype=np.float32)
    calc_label_grad_kernel(
        alpha, beta, log_label_probs, target_sequence, labels, classes,
        label_grads, size=T)
    return label_grads


def calc_ll(alpha_list):
    """ Calculates the negative log-likelihood

    :param alpha_list: list of alphas
    :return: negative log-likelihood
    """
    ll = cupy.empty((len(alpha_list),), dtype=np.float32)
    for idx, alpha in enumerate(alpha_list):
        offset = alpha.shape[0] * alpha.shape[1]
        calc_ll_kernel(alpha, offset, idx, ll, size=1)
    return ll.mean()
