#!/usr/bin/env python3
import argparse

from pytypes import typechecked

from espnet2.train.base_task import BaseTask


class LMRNNTask(BaseTask):
    @classmethod
    @typechecked
    def get_parser(cls, cmd=None) -> argparse.ArgumentParser:
        raise NotImplementedError

    @classmethod
    @typechecked
    def build_model(cls, idim: int, odim: int, args: argparse.Namespace):
        raise NotImplementedError


if __name__ == '__main__':
    LMRNNTask.main()
