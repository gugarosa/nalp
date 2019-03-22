import functools

import tensorflow as tf


def wrapper(function):
    """This function serves as a wrapper for custom decorators. Please use it when
    defining a new decorator.

    Args:
        function (callable): An arbitrary function.

    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])

        else:
            return lambda wrap: function(wrap, *args, **kwargs)

    return decorator


@wrapper
def define_scope(function, scope=None, *args, **kwargs):
    """A decorator used to help when defining new Tensorflow operations.
    It servers as an helper by making them avaliable with tf.variable_scope().

    Args:
        function (callable): An arbitrary function.
        scope (str): A string containing the scope's name.

    """

    # Gathers the attribute as a variable
    attribute = '_cache_' + function.__name__

    # Gathers the name as a variable
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))

        return getattr(self, attribute)

    return decorator
