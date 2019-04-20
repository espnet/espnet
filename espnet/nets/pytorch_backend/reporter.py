# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import logging

import chainer
from chainer import reporter


class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                reporter.report({k: v}, self)
        logging.info('mtl loss:' + str(kwargs['loss']))
