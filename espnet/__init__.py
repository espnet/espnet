import pkg_resources
try:
    __version__ = pkg_resources.get_distribution('espnet').version
except Exception:
    __version__ = '(Not installed from setup.py)'
del pkg_resources
