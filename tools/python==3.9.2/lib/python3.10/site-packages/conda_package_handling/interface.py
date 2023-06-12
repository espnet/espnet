import abc
import os


class AbstractBaseFormat(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def supported(fn):
        return False

    @staticmethod
    @abc.abstractmethod
    def extract(fn, dest_dir, **kw):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def create(prefix, file_list, out_fn, out_folder=os.getcwd(), **kw):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_pkg_details(in_file):
        raise NotImplementedError
