#!/bin/env python

"""
lazy_properties.py

Decorator that makes properties "lazy" i.e. they are only computed
when they are actually used.

For detailed documentation about the usage of this decorator, see
aperture_properties.py.
"""

from typing import Callable


def lazy_property(fn: Callable) -> Callable:
    """
    Decorator for class methods to turn them into lazily evaluated properties.

    Decorating a function with @lazy_property is practically the same as
    decorating it with @property, except that the function is only evaluated
    once, and only when it is actually used.

    Parameters:
     - fn:
       Function to decorate. Should have the call signature
         def function(self)
       with self an object to which @property functions can be
       attached.

    Returns a new @property function that does the lazy evaluation
    of fn.
    """

    # unique identifier for the attribute that will be used to
    # store the property after it has been used for the first
    # time
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        """
        Function that is actually called when an
        @lazy_property is used.

        If the function has not been called before,
        we call it and store its return value in
        a new attribute that is attached to the
        object.
        Returns the value of that attribute.
        """
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
