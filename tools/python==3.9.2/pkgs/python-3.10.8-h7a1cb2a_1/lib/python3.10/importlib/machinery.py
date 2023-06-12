"""The machinery of importlib: finders, loaders, hooks, etc."""

from ._bootstrap import BuiltinImporter, FrozenImporter, ModuleSpec
from ._bootstrap_external import (
    BYTECODE_SUFFIXES,
    DEBUG_BYTECODE_SUFFIXES,
    EXTENSION_SUFFIXES,
    OPTIMIZED_BYTECODE_SUFFIXES,
    SOURCE_SUFFIXES,
    ExtensionFileLoader,
    FileFinder,
    PathFinder,
    SourceFileLoader,
    SourcelessFileLoader,
    WindowsRegistryFinder,
)


def all_suffixes():
    """Returns a list of all recognized module suffixes for this process"""
    return SOURCE_SUFFIXES + BYTECODE_SUFFIXES + EXTENSION_SUFFIXES
