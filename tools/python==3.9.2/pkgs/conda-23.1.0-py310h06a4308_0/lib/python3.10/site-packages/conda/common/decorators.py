# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

import os
from functools import wraps


def env_override(envvar_name, convert_empty_to_none=False):
    """Override the return value of the decorated function with an environment variable.

    If convert_empty_to_none is true, if the value of the environment variable
    is the empty string, a None value will be returned.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = os.environ.get(envvar_name, None)

            if value is not None:
                if value == "" and convert_empty_to_none:
                    return None
                else:
                    return value
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
