"""Sized Dict.

This module provides a utility for tracking the memory size of dictionary-like objects.

Classes:
    SizedDict: A mutable mapping that tracks the approximate memory size of its
    contents.
        - Supports both standard and multiprocessing-shared dictionaries.
        - Updates the tracked size on item insertion, update, and deletion.

Functions:
    get_size(obj, seen=None): Recursively computes the memory size of an object,
    including nested containers.
        - Handles self-referential objects gracefully to avoid infinite recursion.
        - Supports dicts, lists, sets, tuples, and objects with __dict__ attributes.
"""

import collections
import sys

from torch import multiprocessing


def get_size(obj, seen=None):
    """Recursively finds size of objects.

    Taken from https://github.com/bosswissam/pysize

    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif isinstance(obj, (list, set, tuple)):
        size += sum([get_size(i, seen) for i in obj])

    return size


class SizedDict(collections.abc.MutableMapping):
    """A mutable mapping that tracks the approximate memory size of its contents.

    SizedDict is a dictionary-like object that automatically tracks the memory size
    of stored items. It supports both standard in-memory dictionaries and
    multiprocessing-shared dictionaries for concurrent access.

    Attributes:
        cache: The underlying dictionary storing the key-value pairs. For shared
            mode, this is a multiprocessing-safe dictionary. For non-shared mode,
            this is a standard Python dict.
        size: The total approximate memory size in bytes of all keys and values
            currently stored in the dictionary.

    """

    def __init__(self, shared: bool = False, data: dict = None):
        """Initialize a SizedDict instance.

        Args:
            shared: If True, uses a multiprocessing-safe dictionary that can be
                shared across processes. If False (default), uses a standard
                Python dictionary. Defaults to False.
            data: Optional dictionary to initialize the SizedDict with. If None,
                starts with an empty dictionary. Defaults to None.

        Note:
            When shared=True, the manager object is not stored as an instance
            attribute to avoid pickling issues with the "spawn" method in
            multiprocessing, which cannot pickle weakref objects.

        """
        if data is None:
            data = {}

        if shared:
            # NOTE(kamo): Don't set manager as a field because Manager, which includes
            # weakref object, causes following error with method="spawn",
            # "TypeError: can't pickle weakref objects"
            self.cache = multiprocessing.Manager().dict(**data)
        else:
            self.manager = None
            self.cache = dict(**data)
        self.size = 0

    def __setitem__(self, key, value):
        """Set a key-value pair and update the tracked memory size.

        When a key is set:
        - If the key already exists, the old value's size is subtracted before
          adding the new value's size.
        - If the key is new, the size of both the key and value are added.

        Args:
            key: The dictionary key.
            value: The value to associate with the key.

        """
        if key in self.cache:
            self.size -= get_size(self.cache[key])
        else:
            self.size += sys.getsizeof(key)
        self.size += get_size(value)
        self.cache[key] = value

    def __getitem__(self, key):
        """Get the value associated with a key.

        Args:
            key: The dictionary key to retrieve.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key is not found in the dictionary.

        """
        return self.cache[key]

    def __delitem__(self, key):
        """Delete a key-value pair and update the tracked memory size.

        Removes the key and its associated value from the dictionary, and
        subtracts both the key and value sizes from the tracked size.

        Args:
            key: The dictionary key to delete.

        Raises:
            KeyError: If the key is not found in the dictionary.

        """
        self.size -= get_size(self.cache[key])
        self.size -= sys.getsizeof(key)
        del self.cache[key]

    def __iter__(self):
        """Iterate over the dictionary keys.

        Returns:
            An iterator over the keys in the dictionary.

        """
        return iter(self.cache)

    def __contains__(self, key):
        """Check if a key exists in the dictionary.

        Args:
            key: The key to check for.

        Returns:
            True if the key is in the dictionary, False otherwise.

        """
        return key in self.cache

    def __len__(self):
        """Get the number of key-value pairs in the dictionary.

        Returns:
            The number of items currently stored in the dictionary.

        """
        return len(self.cache)
