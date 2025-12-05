import re


def get_formatter(formatter_str: str):
    """
    Converts a short-hand string into a python format string.

    This function takes a string that describes a format in a concise way
    and returns a python format string that can be used with str.format().

    Args:
      formatter_str: The string describing the format.

    Returns:
      A python format string.

    Raises:
      ValueError: If the formatter_str is not a recognized format.

    Examples:
      - If `formatter_str` is already a format string (e.g. "{:,.2f}"), it is returned as is.
      - "usd", "pnl", "size", "$" will return a currency format, e.g. get_formatter("usd0") -> "${:,.0f}"
      - "%", "pct", "ret" will return a percentage format, e.g. get_formatter("pct2") -> "{:.2%}"
      - "#", "num", "count" will return an integer format, e.g. get_formatter("#") -> "{:,.0f}"
      - "sigma", "std" will return a float with 2 decimal places and sigma symbol, e.g. get_formatter("sigma") -> "{:.2f}σ"
    """
    if formatter_str.startswith("{") and formatter_str.endswith("}"):
        return formatter_str
    match = re.match(r"(?P<prefix>[^0-9]*)(?P<digits>\d*)$", formatter_str)
    prefix = match.group("prefix")
    digits = match.group("digits")
    if prefix in ("usd", "pnl", "size", "$"):
        return "${:,.0" + (digits or "0") + "f}"
    elif prefix in ("%", "pct", "ret"):
        return "{:.0" + (digits or "2") + "%}"
    elif prefix in ("#", "num", "count") and digits == "":
        return "{:,.0f}"
    elif prefix in ("sigma", "std"):
        return "{:.2f}σ"
    else:
        raise ValueError(f"cannot parse formatter string {formatter_str!r}")


def python_format_to_c_format(py_format_str: str) -> str:
    # Floating-point numbers with possible thousands separator
    # Translate Python's comma for thousands separator to C's apostrophe
    old_py_format_str = str(py_format_str)
    py_format_str = re.sub(r"\{:[ ]*([+])?(\d+)?,\.(\d+)f\}", r"%'\2.\3f", py_format_str)
    py_format_str = re.sub(r"\{:[ ]*,\.(\d+)f\}", r"%'.\1f", py_format_str)

    # Integers with possible thousands separator
    py_format_str = re.sub(r"\{:[ ]*([+])?(\d+)?,d\}", r"%'\2d", py_format_str)
    py_format_str = re.sub(r"\{:[ ]*([+])?0(\d+),d\}", r"%0'\2d", py_format_str)

    # Regular expressions for other formats remain the same
    # Integers without thousands separator
    py_format_str = re.sub(r"\{:[ ]*([+])?(\d+)?d\}", r"%\2d", py_format_str)
    py_format_str = re.sub(r"\{:[ ]*([+])?0(\d+)d\}", r"%0\2d", py_format_str)
    py_format_str = re.sub(r"\{:[ ]*d\}", r"%d", py_format_str)

    # Strings
    py_format_str = re.sub(r"\{:[ ]*\.(\d+)s\}", r"%.\1s", py_format_str)
    py_format_str = re.sub(r"\{:[ ]*s\}", r"%s", py_format_str)

    # Handle simple placeholders without specific format
    py_format_str = re.sub(r"\{\}", r"%s", py_format_str)

    print(f"{old_py_format_str} --> {py_format_str}")
    return py_format_str
