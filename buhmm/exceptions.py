
class buhmmException(Exception):
    """
    Base class for all `buhmm` exceptions.

    """
    default_message = ''
    def __init__(self, *args, **kwargs):
        if 'msg' in kwargs:
            # Override the message in the first argument.
            self.msg = kwargs['msg']
        elif args:
            self.msg = args[0]
        else:
            self.msg = self.default_message
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.msg

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, repr(self.args))

class NonunifilarException(buhmmException):
    def __init__(self, *args, **kwargs):
        msg = 'A unifilar machine is required.'
        args = (msg,) + args
        buhmmException.__init__(self, *args, **kwargs)

class InvalidInitialNode(buhmmException):
    def __init__(self, *args, **kwargs):
        msg = "Invalid initial node: {0!r}".format(args[0])
        args = (msg,) + args
        buhmmException.__init__(self, *args, **kwargs)

class InvalidNode(buhmmException):
    def __init__(self, *args, **kwargs):
        msg = "Invalid node: {0!r}".format(args[0])
        args = (msg,) + args
        buhmmException.__init__(self, *args, **kwargs)

class UnexpectedSymbol(buhmmException):
    pass

class InvalidTopology(buhmmException):
    pass
