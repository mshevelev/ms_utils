from . import config

def set_namespace(name: str):
    """Set the default namespace for registered extension methods."""
    config.DEFAULT_NAMESPACE = name

def set_conflict_mode(mode: str):
    """Set the mode for handling registration conflicts.
    
    Supported modes:
    - 'raise': Raise an error if a method with the same name already exists (default).
    - 'override': Overwrite existing method with the new one.
    - 'ignore': Skip registration and log a warning.
    """
    valid_modes = ["raise", "override", "ignore"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid conflict mode: {mode}. Must be one of {valid_modes}")
    config.REGISTRATION_CONFLICT_MODE = mode
