import argparse
import logging
import numpy
import os
import platform
import random
import subprocess
import sys


def get_common_argparser():
    """Returns a common argparser for all scripts

    :return: the common argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='pytorch', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    return parser


def get_train_argparser(typ="asr"):
    """Returns a common argparser for all train scripts

    :param str typ: The type of training
    :return: the common training argparser
    """
    parser = get_common_argparser()
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    parser.add_argument('--batch-size', '-b', default=32 if typ == 'tts' else 300 if typ == 'lm' else 50, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    return parser


def get_recog_argparser(typ='asr'):
    """Returns a common argparser for all recognition scripts

    :param str typ: the type of recognition
    :return: the common recognition argparser
    """
    parser = get_common_argparser()
    parser.add_argument('--maxlenratio', default=0.0 if typ == "asr" else 5, type=float,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain min output length')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    return parser


def set_logging_level(verbose):
    """Sets the logging level given the verbose value

    :param int verbose: the logging level
    """
    # logging info
    if verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")


def check_cuda_visible_devices(ngpu, max_gpu=None):
    """Checks that ngpu conforms with CUDA_VISIBLE_DEVICES and sets some John Hopkins University variables if available

    :param int ngpu: The number of gpus to use (given by the program arguments)
    :param int max_gpu: The maximum number of gpus to use for the given task
    """
    if ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == '2':
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(ngpu)]).strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(ngpu)]).decode().strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        if max_gpu is not None and ngpu > max_gpu:
            logging.error("The program only supports ngpu=" + str(max_gpu))
            sys.exit(1)


def set_seed(seed):
    """Sets the random seed

    :param int seed: The seed to use
    """
    logging.info('random seed = %d' % seed)
    random.seed(seed)
    numpy.random.seed(seed)
