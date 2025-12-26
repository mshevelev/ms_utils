"""Custom namespace registration system for extending classes with accessor methods.

This module provides a lightweight framework for registering custom methods as 
namespace accessors on existing classes (similar to pandas' `.str`, `.dt` accessors).
It enables clean extension of third-party classes without modifying their source code.

Key Components
--------------
- `CustomNamespace`: A descriptor that creates namespaced method accessors
- `NamespaceInstance`: Wrapper that binds instance methods to namespace accessors
- `register_method`: Decorator for registering methods to class namespaces

Common Use Cases
----------------
1. **Extending pandas/xarray objects**: Add domain-specific methods to DataFrame/Series
2. **Creating fluent APIs**: Chain custom methods with standard library methods
3. **Organizing utility functions**: Group related methods under a common namespace

Example Usage
-------------
>>> import pandas as pd
>>> from ms_utils.method_registration import register_method
>>>
>>> # Register a custom method on pandas DataFrame under 'custom' namespace
>>> @register_method([pd.DataFrame], namespace='custom')
... def summarize(df):
...     '''Return a summary of the DataFrame.'''
...     return f"Shape: {df.shape}, Columns: {list(df.columns)}"
>>>
>>> # Use the registered method
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
>>> df.custom.summarize()
"Shape: (3, 2), Columns: ['a', 'b']"

Notes
-----
- The first argument of registered methods receives the instance automatically
- Supports registration on multiple classes simultaneously
- Compatible with pandas/xarray `_accessors` pattern for namespace discovery
"""

import functools 
from . import config
 
class NamespaceInstance:
    """Instance-level wrapper that binds namespace methods to a specific object.
    
    This class is returned when accessing a `namespace` attribute on an instance
    (e.g., `df.custom`). It intercepts attribute access and automatically binds
    the instance as the first argument to namespace methods.
    
    The binding mechanism allows registered methods to work like instance methods,
    receiving the object they're called on as their first parameter.
    
    Parameters
    ----------
    namespace : CustomNamespace
        The namespace descriptor containing registered methods
    instance : object
        The instance to bind methods to
    
    Examples
    --------
    >>> # When you access df.custom, a NamespaceInstance is created
    >>> # that wraps 'df' and the 'custom' namespace
    >>> ns_instance = NamespaceInstance(custom_namespace, df)
    >>> # Calling methods automatically passes 'df' as first argument
    >>> ns_instance.method()  # Equivalent to method(df)
    """
    
    def __init__(self, namespace, instance):
        self._namespace = namespace
        self._instance = instance

    def __getattribute__(self, name):
        """Intercept attribute access to bind instance to namespace methods."""
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            pass
        else:
            return attr

        method = self._namespace.__getattribute__(name)
        
        @functools.wraps(method)
        def _wrapper(*args, **kwargs):
            return method(self._instance, *args, **kwargs)

        return _wrapper

    def __dir__(self) -> list[str]:
        """Return list of available methods in the namespace."""
        return self._namespace.__dir__()
  

class CustomNamespace:
    """Descriptor that creates namespaced method accessors on classes.
    
    This descriptor implements the descriptor protocol to provide different behavior
    when accessed from a class vs. an instance:
    - Class access (e.g., `DataFrame.custom`): Returns the descriptor itself
    - Instance access (e.g., `df.custom`): Returns a NamespaceInstance bound to that instance
    
    Parameters
    ----------
    name : str
        The name of the namespace (e.g., 'custom', 'ms', 'ml')
    
    Examples
    --------
    >>> # Typically created automatically by register_method
    >>> namespace = CustomNamespace(name='custom')
    >>> # Methods are added via setattr
    >>> setattr(namespace, 'my_method', my_function)
    """
    
    def __init__(self, name: str) -> None:
        self._name = name
    
    def __get__(self, obj, cls):
        """Descriptor protocol: return self for class access, NamespaceInstance for instance access.
        
        Parameters
        ----------
        obj : object or None
            The instance that the descriptor is accessed from (None if class access)
        cls : type
            The class that owns the descriptor
        
        Returns
        -------
        CustomNamespace or NamespaceInstance
            Returns self for class access, NamespaceInstance for instance access
        """
        if obj is None:  # Class-level access (e.g., DataFrame.custom)
            return self
        return NamespaceInstance(self, obj)


def register_method(classes: list, namespace: str = None):
    """Decorator to register a function as a method in a namespace on multiple classes.
    
    This decorator adds the decorated function to a specified namespace on one or more
    classes. If the namespace doesn't exist, it creates a CustomNamespace descriptor.
    The function will be callable as `instance.namespace.function_name()`.
    
    Parameters
    ----------
    classes : list of type
        List of classes to register the method on (e.g., [pd.DataFrame, pd.Series])
    namespace : str, optional
        Name of the namespace to add the method to (e.g., 'custom', 'ms').
        If None, use config.DEFAULT_NAMESPACE.
    
    Returns
    -------
    callable
        Decorator function that registers the wrapped function
    
    Notes
    -----
    - The decorated function's first parameter receives the instance automatically
    - If the class has a `_accessors` set attribute, the namespace is registered there
      (this is used by pandas/xarray for namespace discovery)
    - Multiple methods can be registered to the same namespace
    - The same method can be registered on multiple classes
    
    Examples
    --------
    **Basic usage with a single class:**
    
    >>> @register_method([pd.DataFrame], namespace='stats')
    ... def summary(df):
    ...     return df.describe()
    >>> 
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> df.stats.summary()  # Calls summary(df)
    
    **Register on multiple classes:**
    
    >>> @register_method([pd.DataFrame, pd.Series], namespace='custom')
    ... def info(obj):
    ...     return f"Type: {type(obj).__name__}, Shape: {obj.shape}"
    >>>
    >>> df.custom.info()  # Works on DataFrame
    >>> series.custom.info()  # Also works on Series
    
    **Multiple methods in the same namespace:**
    
    >>> @register_method([pd.DataFrame], namespace='ml')
    ... def scale(df):
    ...     return (df - df.mean()) / df.std()
    >>>
    >>> @register_method([pd.DataFrame], namespace='ml')
    ... def normalize(df):
    ...     return (df - df.min()) / (df.max() - df.min())
    >>>
    >>> df.ml.scale()  # Both methods available
    >>> df.ml.normalize()
    """
    if namespace is None:
        namespace = config.DEFAULT_NAMESPACE

    def decorator(func):
        for class_ in classes:
            if not hasattr(class_, namespace):
                setattr(class_, namespace, CustomNamespace(name=namespace))
                if hasattr(class_, '_accessors'):
                    class_._accessors.add(namespace)
            _ns = getattr(class_, namespace)
            setattr(_ns, func.__name__, func)
        return func
    return decorator

