from typing import Dict, List, Tuple


def prepare_tooltip_formatters(tooltip_formatters: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Prepare tooltip formatters for Bokeh/hvplot hover tools.

    This function converts a dictionary of column names and their format strings
    into a list of tuples suitable for use with Bokeh's hover tool tooltips.

    Args:
        tooltip_formatters: A dictionary mapping column names to format strings.
                           Format strings should follow Bokeh's formatting syntax.

    Returns:
        A list of tuples where each tuple contains (column_name, tooltip_format_string).

    Examples:
        >>> # Basic usage with common format strings
        >>> formatters = {
        ...     "Sharpe252": "0.00",
        ...     "Return": "0.00%",
        ...     "Volume": "0,0",
        ...     "Price": "$0,0.00"
        ... }
        >>> tooltips = prepare_tooltip_formatters(formatters)
        >>> tooltips
        [('Sharpe252', '@{Sharpe252}{0.00}'), ('Return', '@{Return}{0.00%}'), ('Volume', '@{Volume}{0,0}'), ('Price', '@{Price}{$0,0.00}')]

        >>> # Usage with date formatting
        >>> date_formatters = {
        ...     "Date": "%F",
        ...     "Timestamp": "%Y-%m-%d %H:%M:%S"
        ... }
        >>> date_tooltips = prepare_tooltip_formatters(date_formatters)
        >>> date_tooltips
        [('Date', '@{Date}{%F}'), ('Timestamp', '@{Timestamp}{%Y-%m-%d %H:%M:%S}')]

        >>> # Empty dictionary returns empty list
        >>> prepare_tooltip_formatters({})
        []

    Usage with hvplot:
        >>> import pandas as pd
        >>> import hvplot.pandas  # noqa: F401
        >>>
        >>> # Create sample data
        >>> df = pd.DataFrame({
        ...     'Date': pd.date_range('2023-01-01', periods=5),
        ...     'Return': [0.05, -0.02, 0.03, 0.01, -0.01],
        ...     'Volume': [1000000, 800000, 1200000, 900000, 1100000]
        ... })
        >>>
        >>> # Prepare tooltips
        >>> tooltips = prepare_tooltip_formatters({
        ...     "Date": "%F",
        ...     "Return": "0.00%",
        ...     "Volume": "0,0"
        ... })
        >>>
        >>> # Use with hvplot
        >>> plot = df.hvplot(x='Date', y='Return').opts(
        ...     tools=['hover'],
        ...     hover_tooltips=tooltips
        ... )

    Note:
        The format strings should follow Bokeh's formatting syntax:
        - Numbers: "0.00", "0,0", "$0,0.00", "0.00%"
        - Dates: "%F", "%Y-%m-%d", "%H:%M:%S"
        - Custom: Any valid Bokeh format string
    """
    tooltips = [(k, f"@{{{k}}}{{{v}}}") for k, v in tooltip_formatters.items()]
    return tooltips
