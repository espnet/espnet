#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging


def get_attribute(obj, name, *default):
    """Get a specified attribute from a given object.

    Args:
        obj (object): Object.
        name (str): Attribute name.
        default (any, optional): Default value when specified attribute is missing.

    Returns:
        any: Attribute of object.

    """
    # default should be empty or a single argument
    assert len(default) < 2
    if hasattr(obj, name):
        return getattr(obj, name)
    elif len(default) != 0:
        logging.warning("attribute \"%s\" does not exist. use default value %s." % (name, str(default[0])))
        return default[0]
    else:
        raise AttributeError("%s does not exist in the object." % name)
