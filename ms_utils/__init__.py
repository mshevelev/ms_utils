from . import config

def set_namespace(name: str):
    """Set the default namespace for registered extension methods.
    
    This should be called before importing any extension subpackages:
    >>> import ms_utils
    >>> ms_utils.set_namespace("my_ns")
    >>> import ms_utils.pandas
    """
    config.DEFAULT_NAMESPACE = name
