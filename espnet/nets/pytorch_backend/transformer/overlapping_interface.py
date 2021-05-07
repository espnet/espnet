import torch
import numpy as np
from itertools import groupby
import pdb
import logging


def get_token_level_ids_probs(ctc_ids, ctc_probs):

    bh = len(ctc_ids[0]) // 2
    y_hat = [x[0].item() for x in groupby(ctc_ids[0])]
    # y_idx = torch.nonzero(y_hat != 0).squeeze(-1)
    # y_hat = y_hat[y_idx]
    score = [i for i in range(bh)]
    score = score + score[-1::-1]
    probs_hat = []
    y_score = []
    cnt = 0
    cidx = 0
    for i, y in enumerate(y_hat):
        probs_hat.append(-1)
        y_score.append([])
        while y != ctc_ids[0][cnt]:
            cnt += 1
            if cnt == bh + 1:
                cidx = i
        while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
            if probs_hat[i] < ctc_probs[0][cnt]:
                probs_hat[i] = ctc_probs[0][cnt].item()
            try:
                y_score[i].append(score[cnt])
            except:
                pdb.set_trace()
            cnt += 1
            if cnt == bh + 1:
                cidx = i
    probs_hat = torch.from_numpy(np.array(probs_hat))
    score = []
    for i, s in enumerate(y_score):
        score.append(s[len(s) // 2])

    y_hat_score = torch.from_numpy(np.array([(i, j) for i, j in zip(y_hat, score)]))

    return y_hat_score, probs_hat, cidx


def tie_breaking(pairs, probs):

    total_len = len(pairs)
    path = []
    probs_path = []
    for idx, pair in enumerate(pairs):
        if pair[0][1] <= pair[1][1]:
            path.append(pair[1][0])
            probs_path.append(probs[idx][1])
        else:
            path.append(pair[0][0])
            probs_path.append(probs[idx][0])

    return torch.tensor(path, dtype=torch.int64), torch.tensor(
        probs_path, dtype=torch.float
    )


def dynamic_matching(
    tensor1, tensor2, prob1=None, prob2=None, reserved_t=None, reserved_p=None
):

    tensor1 = tensor1.tolist()
    tensor2 = tensor2.tolist()
    prob1 = prob1.tolist()
    prob2 = prob2.tolist()

    if reserved_t == None:
        reserved_p = None
    else:
        tensor1 = reserved_t[0] + tensor1
        tensor2 = reserved_t[1] + tensor2
        prob1 = reserved_p[0] + prob1
        prob2 = reserved_p[1] + prob2

    M, N = len(tensor1), len(tensor2)
    """
    if M == 0:
        return None, [(0, t) for t in tensor2], [(0.0, p) for p in prob2], tensor1, tensor2, None, None
    if N == 0:
        return None, [(t, 0) for t in tensor1], [(p, 0.0) for p in prob1], tensor1, tensor2, None, None
    """
    dp = [[0 for _ in range(N + 1)] for _ in range(M + 1)]
    dp[0][0] = 0, [], []
    s1, s2 = 0, 0
    if len(tensor1) > 0:
        s1 = tensor1[0][1]

    if len(tensor2) > 0:
        s2 = tensor2[0][1]

    for i in range(1, N + 1):
        dp[0][i] = (
            i,
            dp[0][i - 1][1] + [([0, s1], tensor2[i - 1])],
            dp[0][i - 1][2] + [(0, prob2[i - 1])],
        )
    for i in range(1, M + 1):
        if len(tensor2) > 0:
            score = tensor2[0][1]
        dp[i][0] = (
            i,
            dp[i - 1][0][1] + [(tensor1[i - 1], [0, s2])],
            dp[i - 1][0][2] + [(prob1[i - 1], 0)],
        )

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if tensor1[i - 1][0] == tensor2[j - 1][0]:
                dp[i][j] = (
                    dp[i - 1][j - 1][0],
                    dp[i - 1][j - 1][1] + [(tensor1[i - 1], tensor2[j - 1])],
                    dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])],
                )
            else:
                num, idx = torch.min(
                    torch.tensor(
                        [dp[i - 1][j - 1][0], dp[i - 1][j][0], dp[i][j - 1][0]]
                    ),
                    0,
                )
                dp[i][j] = [0, 0, 0]
                dp[i][j][0] = 1 + num
                if idx == 0:
                    dp[i][j][1] = dp[i - 1][j - 1][1] + [
                        (tensor1[i - 1], tensor2[j - 1])
                    ]
                    dp[i][j][2] = dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])]
                if idx == 1:
                    dp[i][j][1] = dp[i - 1][j][1] + [
                        (tensor1[i - 1], [0, tensor2[j - 1][1]])
                    ]
                    dp[i][j][2] = dp[i - 1][j][2] + [(prob1[i - 1], 0)]
                if idx == 2:
                    dp[i][j][1] = dp[i][j - 1][1] + [
                        ([0, tensor1[i - 1][1]], tensor2[j - 1])
                    ]
                    dp[i][j][2] = dp[i][j - 1][2] + [(0, prob2[j - 1])]
    dp_light = np.array([[n[0] for n in m] for m in dp])
    opt1 = np.min(dp_light[:, -1])
    pos1 = np.nonzero(dp_light[:, -1] == opt1)[0][-1]
    opt2 = np.min(dp_light[-1, :])
    pos2 = np.nonzero(dp_light[-1, :] == opt2)[0][-1]

    if pos1 == M and pos2 == N:
        end_point = (M, N)
    elif opt1 < opt2:
        end_point = (pos1, N)
    elif opt1 > opt2:
        end_point = (M, pos2)
    else:
        end_point = (M, N)
    end_point = (M, N)
    reserved_t = (tensor2[end_point[1] :], tensor1[end_point[0] :])
    reserved_p = (prob2[end_point[1] :], prob1[end_point[0] :])

    if pos1 == M and pos2 == N:
        dp_light = None
    else:
        dp_light = dp_light[: end_point[0] + 1, : end_point[1] + 1]
    logging.info("1")

    return (
        dp_light,
        dp[end_point[0]][end_point[1]][1],
        dp[end_point[0]][end_point[1]][2],
        tensor1[: end_point[0]],
        tensor2[: end_point[1]],
        reserved_t,
        reserved_p,
    )


def dynamic_matching_xl(
    t1,
    t2,
    prob1,
    prob2,
    dp_prev=None,
    t1_prev=None,
    t2_prev=None,
    reserved_t=None,
    reserved_p=None,
):

    if dp_prev is None:
        return dynamic_matching(t1, t2, prob1, prob2)
    # pdb.set_trace()
    if reserved_t == None:
        reserved_t = ([], [])
        reserved_p = ([], [])
    t1 = t1.tolist()
    t2 = t2.tolist()
    prob1 = prob1.tolist()
    prob2 = prob2.tolist()
    """
    if len(tensor1) == 0:
        return None, [(0, t) for t in tensor2], [(0, p) for p in prob2], tensor1, tensor2
    if len(tensor2) == 0:
        return None, [(t, 0) for t in tensor1], [(p, 0) for p in prob1], tensor1, tensor2
    """
    sM = len(t1_prev)
    sN = len(t2_prev)
    tensor1 = t1_prev + reserved_t[0] + t1
    tensor2 = t2_prev + reserved_t[1] + t2
    prob1 = [0 for _ in range(sM)] + reserved_p[0] + prob1
    prob2 = [0 for _ in range(sN)] + reserved_p[1] + prob2
    M, N = len(tensor1), len(tensor2)
    dp = [[0 for _ in range(N + 1)] for _ in range(M + 1)]

    for m in range(sM + 1):
        for n in range(sN + 1):
            dp[m][n] = dp_prev[m][n], [], []
    for j in range(sN + 1, N + 1):
        dp[0][j] = dp[0][j - 1][0] + 1, [], []
    # pdb.set_trace()
    for i in range(1, sM + 1):
        for j in range(sN + 1, N + 1):
            if tensor1[i - 1] == tensor2[j - 1]:
                dp[i][j] = (
                    dp[i - 1][j - 1][0],
                    dp[i - 1][j - 1][1] + [(tensor1[i - 1], tensor2[j - 1])],
                    dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])],
                )
            else:
                num, idx = torch.min(
                    torch.tensor(
                        [dp[i - 1][j - 1][0], dp[i - 1][j][0], dp[i][j - 1][0]]
                    ),
                    0,
                )
                dp[i][j] = 1 + num.item(), [], []
    for i in range(sM + 1, M + 1):
        dp[i][0] = dp[i - 1][0][0] + 1, [], []
    for i in range(sM + 1, M + 1):
        for j in range(1, sN + 1):
            if tensor1[i - 1] == tensor2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1][0], [], []
            else:
                num, idx = torch.min(
                    torch.tensor(
                        [dp[i - 1][j - 1][0], dp[i - 1][j][0], dp[i][j - 1][0]]
                    ),
                    0,
                )
                dp[i][j] = 1 + num.item(), [], []

    for i in range(sM + 1, M + 1):
        for j in range(sN + 1, N + 1):
            if tensor1[i - 1] == tensor2[j - 1]:
                dp[i][j] = (
                    dp[i - 1][j - 1][0],
                    dp[i - 1][j - 1][1] + [(tensor1[i - 1], tensor2[j - 1])],
                    dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])],
                )
            else:
                num, idx = torch.min(
                    torch.tensor(
                        [dp[i - 1][j - 1][0], dp[i - 1][j][0], dp[i][j - 1][0]]
                    ),
                    0,
                )
                dp[i][j] = [0, 0, 0]
                dp[i][j][0] = 1 + num.item()
                if idx == 0:
                    dp[i][j][1] = dp[i - 1][j - 1][1] + [
                        (tensor1[i - 1], tensor2[j - 1])
                    ]
                    dp[i][j][2] = dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])]
                if idx == 1:
                    dp[i][j][1] = dp[i - 1][j][1] + [(tensor1[i - 1], 0)]
                    dp[i][j][2] = dp[i - 1][j][2] + [(prob1[i - 1], 0)]
                if idx == 2:
                    dp[i][j][1] = dp[i][j - 1][1] + [(0, tensor2[j - 1])]
                    dp[i][j][2] = dp[i][j - 1][2] + [(0, prob2[j - 1])]

    dp_light = np.array([[n[0] for n in m] for m in dp])

    opt1 = np.min(dp_light[:, -1])
    pos1 = np.nonzero(dp_light[:, -1] == opt1)[0][-1]
    opt2 = np.min(dp_light[-1, :])
    pos2 = np.nonzero(dp_light[-1, :] == opt2)[0][-1]

    if pos1 == M and pos2 == N:
        end_point = (M, N)
    elif opt1 < opt2:
        end_point = (pos1, N)
    elif opt1 > opt2:
        end_point = (M, pos2)
    else:
        end_point = (M, N)  # end_point = (pos1, pos2)

    if pos1 == M and pos2 == N:  # or \
        # len(reserved_t[0]) > len(t1) \
        # or len(reserved_t[1]) > len(t2):
        dp_light = None
    else:
        dp_light = dp_light[: end_point[0] + 1, : end_point[1] + 1]

    # end_point = (pos1, pos2)
    # pdb.set_trace()
    reserved_t = (tensor2[end_point[1] :], tensor1[end_point[0] :])
    reserved_p = (prob2[end_point[1] :], prob1[end_point[0] :])

    logging.info("tensor1:{}".format(tensor1))
    logging.info("tensor2:{}".format(tensor2))
    logging.info("reserved:{}".format(reserved_t))
    # pdb.set_trace()
    # reserve_dim = 5
    # dim1, dim2 = max(0, M - reserve_dim), max(0, N - reserve_dim)
    # dp_light = dp_light[dim1:, dim2:]
    # tensor1 = tensor1[dim1:]
    # tensor2 = tensor2[dim2:]
    # pdb.set_trace()
    return (
        dp_light,
        dp[end_point[0]][end_point[1]][1],
        dp[end_point[0]][end_point[1]][2],
        tensor1[: end_point[0]],
        tensor2[: end_point[1]],
        reserved_t,
        reserved_p,
    )
