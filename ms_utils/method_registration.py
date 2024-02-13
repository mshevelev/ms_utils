import functools 
 
class NamespaceInstance:
  def __init__(self, namespame, instance):
    self._namespace = namespame
    self._instance = instance

  def __getattribute__(self, name):
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
    return self._namespace.__dir__()
  

class CustomNamespace:
    def __init__(self, name: str) -> None:
      self._name = name
    
    def __get__(self, obj, cls):
      if obj is None: # we're accessing the attribute of the class
        return self
      return NamespaceInstance(self, obj)


def register_method(classes: list, namespace: str):
  """for each class in `classes` register decorated_function as `class`.`namespace`.`decorated_function`
  self will be passed in place of the first argument 
  """
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

