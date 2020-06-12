import inspect


class Serializable(object):

    def __initialize(self, locals_):
        if getattr(self, "_Serializable__initialized", False):
            return

        signature = inspect.signature(self.__init__)
        positional_keys = [
            p.name for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

        var_positional_keys = [
            p.name for p in signature.parameters.values()
            if p.kind == p.VAR_POSITIONAL
        ]

        keyword_keys = [
            p.name for p in signature.parameters.values()
            if p.kind == p.KEYWORD_ONLY
        ]

        var_keyword_keys = [
            p.name for p in signature.parameters.values()
            if p.kind == p.VAR_KEYWORD
        ]

        if len(var_positional_keys) > 1:
            raise NotImplementedError(
                "Can't yet handle more than one variable args. Got: {}"
                "".format(var_positional_keys))
        if len(var_keyword_keys) > 1:
            raise NotImplementedError(
                "Can't yet handle more than one variable kwargs. Got: {}"
                "".format(var_keyword_keys))

        positional_values = [
            locals_[key] for key in positional_keys if key in locals_
        ]
        var_positional_values = (
            locals_.get(var_positional_keys[0], ())
            if var_positional_keys
            else ())
        keyword_values = {
            key: locals_[key]
            for key in keyword_keys if key in locals_
        }
        var_keyword_values = (
            locals_.get(var_keyword_keys[0], {})
            if var_keyword_keys
            else {})

        bound_arguments = signature.bind(*positional_values,
                                         *var_positional_values,
                                         **keyword_values, **var_keyword_values)
        bound_arguments.apply_defaults()

        self.__args = bound_arguments.args
        self.__kwargs = bound_arguments.kwargs

        self.__initialized = True

    def __getstate__(self):
        assert getattr(self, '_Serializable__initialized', False), (
            "Cannot get state from uninitialized Serializable. Forgot to call"
            " `self._Serializable__initialize` in your __init__ method?")

        state = {
            '__args': self.__args,
            '__kwargs': self.__kwargs,
        }

        return state

    def __setstate__(self, state):
        out = type(self)(*state["__args"], **state["__kwargs"])
        self.__dict__.update(out.__dict__)

    @staticmethod
    def clone(instance):
        assert isinstance(instance, Serializable), (
            "Can only clone Serializable objects. Got: {}"
            "".format(type(instance)))
        assert getattr(instance, '_Serializable__initialized', False), (
            "Cannot clone an uninitialized Serializable. Forgot to call"
            " `self._Serializable__initialize` in your __init__ method?")

        state = instance.__getstate__()

        signature = inspect.signature(instance.__init__)
        bound_arguments = signature.bind(*state['__args'], **state['__kwargs'])
        bound_arguments.apply_defaults()

        out = type(instance)(*bound_arguments.args, **bound_arguments.kwargs)

        return out
