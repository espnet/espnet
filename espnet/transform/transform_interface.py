# TODO(karita): add this to all the transform impl.
class TransformInterface:
    def __call__(self, x):
        raise NotImplementedError("__call__ method is not implemented")

    @classmethod
    def add_arguments(cls, parser):
        return parser
