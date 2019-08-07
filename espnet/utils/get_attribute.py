#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging


def get_attribute(obj, name, default=None):
    """Get a specified attribute from a given object.

    Args:
        obj (object): Object.
        name (str): Attribute name.
        default (any, optional): Default value when specified attribute is missing.

    Returns:
        any: Attribute of object.

    """
    if hasattr(obj, name):
        return getattr(obj, name)
    elif default is not None:
        logging.info("%s does not exist in the object. use default value." % name)
        return default
    else:
        raise AttributeError("%s does not exist in the object." % name)
