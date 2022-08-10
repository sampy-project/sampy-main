from functools import wraps


def sampy_class(cls):
    """
    A sampy class is obtained by composition of multiple building blocks using multiple inheritance. This decorator
    removes the need for the user to deal with MRO himself, as well as calling the parents classes while defining new
    sampy classes.

    :param cls: class decorated.

    :return: class with a modified init function.
    """
    def modify_init(init):

        def sampy_init(self, **kwargs):

            # call the init of all the inherited classes.
            ordered_init_call = []
            for i, sub_cls in enumerate(type(self).mro()):
                if i == 0:
                    continue
                if sub_cls.__name__.startswith('Base'):
                    ordered_init_call.append(sub_cls)
            for i, sub_cls in enumerate(type(self).mro()):
                if i == 0:
                    continue
                if not sub_cls.__name__.startswith('Base'):
                    ordered_init_call.append(sub_cls)

            for i, sub_cls in enumerate(ordered_init_call):
                if sub_cls.__name__ == 'object':
                    sub_cls.__init__(self)
                else:
                    sub_cls.__init__(self, **kwargs)

            init(self, **kwargs)

        return sampy_init

    setattr(cls, '__init__', modify_init(cls.__init__))

    return cls


def debug_method(method):
    @wraps(method)
    def returned_function(self, *args, **kwargs):
        # print(args, kwargs)
        if hasattr(self, '_sampy_debug_' + method.__name__):
            getattr(self, '_sampy_debug_' + method.__name__)(*args, **kwargs)
        rv = method(self, *args, **kwargs)
        return rv
    return returned_function


def use_debug_mode(cls):
    for name in dir(cls):
        if not name.startswith('_') and hasattr(getattr(cls, name), '__call__'):
            setattr(cls, name, debug_method(getattr(cls, name)))
