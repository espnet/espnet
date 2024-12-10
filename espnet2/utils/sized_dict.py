import collections
import sys

from torch import multiprocessing


def get_size(obj, seen=None):
    """
        Recursively finds the size of objects in bytes.

    This function calculates the total memory footprint of an object by recursively
    traversing through its contents. It accounts for various object types such as
    dictionaries, lists, sets, tuples, and custom objects with `__dict__`.

    This implementation is based on the code found at:
    https://github.com/bosswissam/pysize

    Args:
        obj: The object whose size is to be calculated. It can be of any type.
        seen: A set of object IDs that have already been encountered during
            recursion. This is used to prevent infinite loops in case of
            self-referential objects.

    Returns:
        int: The total size of the object in bytes.

    Examples:
        >>> get_size([1, 2, 3])
        80  # The size may vary based on the system and Python version.

        >>> get_size({'a': 1, 'b': [2, 3]})
        200  # The size may vary based on the system and Python version.

    Raises:
        TypeError: If the object cannot be sized.
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
    """
        A dictionary-like class that keeps track of the total size of its items.

    This class extends `collections.abc.MutableMapping` and allows for the
    creation of a dictionary that can either be shared among multiple processes
    or kept local to a single process. The `SizedDict` automatically computes
    and maintains the total size of its contents, making it useful for
    memory-sensitive applications.

    Attributes:
        shared (bool): Indicates whether the dictionary is shared among processes.
        cache (dict): The underlying dictionary that stores the key-value pairs.
        size (int): The total size in bytes of the dictionary's contents.

    Args:
        shared (bool): If True, use a multiprocessing manager to allow sharing
            between processes. Defaults to False.
        data (dict, optional): Initial data to populate the dictionary.
            Defaults to an empty dictionary.

    Examples:
        >>> my_dict = SizedDict()
        >>> my_dict['a'] = [1, 2, 3]
        >>> my_dict.size
        64  # Size may vary based on the object
        >>> my_dict['b'] = 'hello'
        >>> my_dict.size
        90  # Size may vary based on the object
        >>> del my_dict['a']
        >>> my_dict.size
        48  # Size may vary based on the object

    Note:
        The size calculation includes the size of keys, values, and their
        references. This class is especially useful when dealing with large
        datasets where memory consumption needs to be monitored.

    Todo:
        - Implement methods for size limiting or eviction strategies.
    """

    def __init__(self, shared: bool = False, data: dict = None):
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
        if key in self.cache:
            self.size -= get_size(self.cache[key])
        else:
            self.size += sys.getsizeof(key)
        self.size += get_size(value)
        self.cache[key] = value

    def __getitem__(self, key):
        return self.cache[key]

    def __delitem__(self, key):
        self.size -= get_size(self.cache[key])
        self.size -= sys.getsizeof(key)
        del self.cache[key]

    def __iter__(self):
        return iter(self.cache)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)
