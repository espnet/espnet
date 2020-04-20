from distutils.version import LooseVersion
import logging

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import to_device


class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param str ctc_type: builtin or warpctc
    :param bool reduce: reduce the CTC loss into a scalar
    """

    def __init__(self, odim, eprojs, dropout_rate, ctc_type="warpctc", reduce=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        # In case of Pytorch >= 1.2.0, CTC will be always builtin
        self.ctc_type = (
            ctc_type
            if LooseVersion(torch.__version__) < LooseVersion("1.2.0")
            else "builtin"
        )
        if ctc_type != self.ctc_type:
            logging.warning(f"CTC was set to {self.ctc_type} due to PyTorch version.")
        if self.ctc_type == "builtin":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        else:
            raise ValueError(
                'ctc_type must be "builtin" or "warpctc": {}'.format(self.ctc_type)
            )

        self.ignore_id = -1
        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        if self.ctc_type == "builtin":
            th_pred = th_pred.log_softmax(2)
            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            # Batch-size average
            loss = loss / th_pred.size(1)
            return loss
        elif self.ctc_type == "warpctc":
            return self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad):
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        self.loss = None
        hlens = torch.from_numpy(np.fromiter(hlens, dtype=np.int32))
        olens = torch.from_numpy(np.fromiter((x.size(0) for x in ys), dtype=np.int32))

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        # zero padding for ys
        ys_true = torch.cat(ys).cpu().int()  # batch x olen

        # get length info
        logging.info(
            self.__class__.__name__
            + " input lengths:  "
            + "".join(str(hlens).split("\n"))
        )
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + "".join(str(olens).split("\n"))
        )

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        dtype = ys_hat.dtype
        ys_hat = ys_hat.transpose(0, 1)
        if self.ctc_type == "warpctc" or dtype == torch.float16:
            # warpctc only supports float32
            # torch.ctc does not support float16 (#1751)
            ys_hat = ys_hat.to(dtype=torch.float32)
        if self.ctc_type == "builtin":
            # use GPU when using the cuDNN implementation
            ys_true = to_device(self, ys_true)
        self.loss = to_device(self, self.loss_fn(ys_hat, ys_true, hlens, olens)).to(
            dtype=dtype
        )
        if self.reduce:
            # NOTE: sum() is needed to keep consistency
            # since warpctc return as tensor w/ shape (1,)
            # but builtin return as tensor w/o shape (scalar).
            self.loss = self.loss.sum()
            logging.info("ctc loss:" + str(float(self.loss)))

        return self.loss

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


def ctc_for(args, odim, reduce=True):
    """Returns the CTC module for the given args and output dimension

    :param Namespace args: the program args
    :param int odim : The output dimension
    :param bool reduce : return the CTC loss in a scalar
    :return: the corresponding CTC module
    """
    num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
    if num_encs == 1:
        # compatible with single encoder asr mode
        return CTC(
            odim, args.eprojs, args.dropout_rate, ctc_type=args.ctc_type, reduce=reduce
        )
    elif num_encs >= 1:
        ctcs_list = torch.nn.ModuleList()
        if args.share_ctc:
            # use dropout_rate of the first encoder
            ctc = CTC(
                odim,
                args.eprojs,
                args.dropout_rate[0],
                ctc_type=args.ctc_type,
                reduce=reduce,
            )
            ctcs_list.append(ctc)
        else:
            for idx in range(num_encs):
                ctc = CTC(
                    odim,
                    args.eprojs,
                    args.dropout_rate[idx],
                    ctc_type=args.ctc_type,
                    reduce=reduce,
                )
                ctcs_list.append(ctc)
        return ctcs_list
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
