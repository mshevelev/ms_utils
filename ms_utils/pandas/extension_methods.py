import datetime as dt
import logging

from IPython.display import HTML
import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
import itertools
from typing import Any, Optional, Union, TypeVar, overload, Literal, Iterable
import pandas.api.types as ptypes


from . import styler a pd_styler

SeriesOrDataFrame = TypeVar("U", bound=Union[pd.Series, pd.DataFrame])


def add_fake_rows(
    data: Union[pd.DataFrame, pd.Series], breaks: Union[str, Iterable], fake_value=np.nan
) -> Union[pd.DataFrame, pd.Series]:
    """Add break rows to a DataFrame or Series at specified intervals or positions.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data with DatetimeIndex.
    breaks : str or iterable
        Specification for where to add breaks:
        - 'year': Add breaks at the start of each year
        - 'month': Add breaks at the start of each month
        - 'quarter': Add breaks at the start of each quarter
        - iterable: Custom break positions (list, pd.Series, or 1D numpy array)
    fake_value : scalar, default np.nan
        The value to insert at break positions. Can be any scalar value
        (e.g., np.nan, 0, -999, 'BREAK', etc.).

    Returns
    -------
    pd.DataFrame or pd.Series
        Data structure with break rows inserted at specified positions,
        filled with fake_value. Returns same type as input.

    Raises
    ------
    AssertionError
        If data.index is not a DatetimeIndex or not monotonic increasing.
    ValueError
        If any custom break values already exist in the index.
    """
    assert isinstance(data.index, pd.DatetimeIndex), "data.index must be a DatetimeIndex"
    assert data.index.is_monotonic_increasing, "index must be monotonic increasing"

    if isinstance(breaks, str):
        if breaks == "year":
            break_positions = data.index.to_series().groupby(data.index.year).first() - pd.Timedelta("1ns")
        elif breaks == "month":
            break_positions = data.index.to_series().groupby(
                [data.index.year, data.index.month]
            ).first() - pd.Timedelta("1ns")
        elif breaks == "quarter":
            break_positions = data.index.to_series().groupby(
                [data.index.year, data.index.quarter]
            ).first() - pd.Timedelta("1ns")
        else:
            raise ValueError(
                f"Unsupported break type: '{breaks}'. Use 'year', 'month', 'quarter', or provide custom iterable."
            )
    else:
        # Handle custom iterable (list, pd.Series, or 1D numpy array)
        break_positions = pd.to_datetime(breaks)

        # Check if any break values already exist in the index
        existing_breaks = break_positions.isin(data.index)
        if existing_breaks.any():
            conflicting_values = break_positions[existing_breaks].tolist()
            raise ValueError(f"Break values already exist in index: {conflicting_values}")

    # Handle both DataFrame and Series
    if isinstance(data, pd.DataFrame):
        fake_data = pd.DataFrame(fake_value, index=break_positions, columns=data.columns)
    else:  # pd.Series
        fake_data = pd.Series(fake_value, index=break_positions, name=data.name)

    return pd.concat([fake_data, data]).sort_index()


def trim_nans(x: SeriesOrDataFrame, *, how: Literal["any", "all"] = "any", subset=None):
    """Remove rows containing NaNs from the beginning and end of the Series or DataFrame.

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        The input data structure to trim NaNs from.
    how : {'any', 'all'}, default 'any'
        Determines the logic for NaN detection:
        - 'any': Remove rows if at least one element is NaN
        - 'all': Remove rows only if all elements in a row are NaN
    subset : list-like, optional
        Column labels to consider for NaN detection. If None, uses all columns.
        Only applicable for DataFrames.

    Returns
    -------
    pd.Series or pd.DataFrame
        The trimmed data structure with NaN rows removed from beginning and end.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [np.nan, 1, 2, np.nan], 'B': [np.nan, 3, 4, 5]})
    >>> trim_nans(df, how='any')
         A    B
    1  1.0  3.0
    2  2.0  4.0
    """
    df = x.to_frame() if isinstance(x, pd.Series) else x
    if subset is not None:
        df = df[subset]

    # Find first and last rows that don't match the NaN criteria
    if how == "any":
        # Remove rows where ANY column has NaN
        has_nan = df.isna().any(axis=1)
    elif how == "all":
        # Remove rows where ALL columns have NaN
        has_nan = df.isna().all(axis=1)
    else:
        raise ValueError(f"{how=} not supported")

    # Find first and last valid indices
    valid_indices = ~has_nan
    if not valid_indices.any():
        # All rows have NaN according to criteria, return empty
        return x.iloc[0:0]

    first_valid_idx = valid_indices.idxmax()  # First True value
    last_valid_idx = valid_indices[::-1].idxmax()  # Last True value

    return x.loc[first_valid_idx:last_valid_idx]


def normalize(x: SeriesOrDataFrame, *, ord=1, axis=0, skipna=True) -> SeriesOrDataFrame:
    """Normalize a Series or DataFrame using vector norms.

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        The input data to normalize.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, default 1
        Order of the norm. See numpy.linalg.norm for details.
        Common values:
        - 1: L1 norm (sum of absolute values)
        - 2: L2 norm (Euclidean norm)
        - np.inf: Maximum norm
    axis : {0, 1}, default 0
        Axis along which to compute the norm:
        - 0: Normalize along rows (each column normalized separately)
        - 1: Normalize along columns (each row normalized separately)
    skipna : bool, default True
        Whether to skip NaN values when computing the norm.
        If True, NaN values are treated as zero for norm calculation.

    Returns
    -------
    pd.Series or pd.DataFrame
        The normalized data structure.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([3, 4])
    >>> normalize(s, ord=2)  # L2 normalization
    0    0.6
    1    0.8
    dtype: float64
    """
    assert axis in (0, 1)
    if skipna:
        denom = np.linalg.norm(np.nan_to_num(x), ord=ord, axis=axis)
    else:
        denom = np.linalg.norm(x, ord=ord, axis=axis)
    if isinstance(x, pd.Series):
        return x.div(denom)
    else:
        return x.div(denom, axis=(1 if axis == 0 else 0))


def flatten_columns(x: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single-level columns with string labels.

    This function converts MultiIndex columns to regular columns by joining
    the level values with commas. Useful for simplifying complex column
    structures for display or export purposes.

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame with potentially MultiIndex columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened column names. If input doesn't have MultiIndex
        columns, returns the original DataFrame unchanged.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({('A', 'X'): [1, 2], ('A', 'Y'): [3, 4], ('B', 'Z'): [5, 6]})
    >>> flatten_columns(df)
       A, X  A, Y  B, Z
    0     1     3     5
    1     2     4     6
    """
    if not isinstance(x.columns, pd.MultiIndex):
        return x
    x = x.copy()
    x.columns = [", ".join(map(str, col)) for col in x.columns]
    return x


def ix2date(x: SeriesOrDataFrame, format="%Y%m%d") -> SeriesOrDataFrame:
    """Convert index values from YYYYMMDD format to datetime objects.

    This function converts integer or string index values in YYYYMMDD format
    to pandas datetime objects. Handles both single-level and MultiIndex
    structures, specifically looking for 'date' and 'stock' levels.

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        Input data with index in YYYYMMDD format.

    Returns
    -------
    pd.Series or pd.DataFrame
        Data structure with datetime-converted index.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2, 3], index=[20230101, 20230102, 20230103])
    >>> ix2date(s)
    2023-01-01    1
    2023-01-02    2
    2023-01-03    3
    dtype: int64
    """
    x = x.copy()
    if isinstance(x.index, pd.MultiIndex):
        new_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(x.index.get_level_values("date"), format=format), x.index.get_level_values("stock")],
            names=["date", "stock"],
        )
        x.index = new_index
    else:
        x.index = pd.to_datetime(x.index, format=format)
    return x


def ix2dt(x: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Create a new pd.Series or pd.DataFrame with `date` or `time` indices (or levels of MultiIndex)
    converted to datetime format. If both `date` and `time` are present, they are merged into `datetime`.
    """
    index = x.index
    # Create new index without 'date' and 'time' levels
    date_time_levels = [level for level in index.names if level in ("date", "time")]
    # Extract date and time levels, and combine them into a datetime string

    if set(date_time_levels) == set(["date", "time"]):
        date_time_strs = (
            index.get_level_values("date").astype(str).str.replace("-", "")
            + ":"
            + index.get_level_values("time").astype(str).str.replace(":", "")
        )
        new_index = pd.to_datetime(date_time_strs, format="%Y%m%d:%H%M%S").rename("datetime")
    elif set(date_time_levels) == set(["date"]):
        date_strs = index.get_level_values("date").astype(str).str.replace("-", "")
        # if date_strs.dtype != int:
        #   logging.warning(f"cannot apply ix2dt to `date` with dtype='{date_strs.dtype}' ")
        #   return x
        new_index = pd.to_datetime(date_strs, format="%Y%m%d").rename("date")
    elif set(date_time_levels) == set(["time"]):
        time_strs = index.get_level_values("time").astype(str).str.replace(":", "")
        # if time_strs.dtype != int:
        #   logging.warning(f"cannot apply ix2dt to `time` with dtype='{time_strs.dtype=}' ")
        #   return x
        new_index = pd.to_datetime(time_strs, format="%H%M%S").rename("time")
    else:
        logging.warning("no 'date' or 'time' level")
        return x

    remaining_levels = [level for level in index.names if level not in ["date", "time"]]
    if remaining_levels:
        new_index = pd.MultiIndex.from_arrays(
            [new_index] + [index.get_level_values(level) for level in remaining_levels],
            #      names=['datetime'] + remaining_levels
        )
    else:
        new_index = pd.Index(new_index, name="datetime")  # pd.DatetimeIndex(datetime_index, name='datetime')

    x = x.copy()
    x.index = new_index
    return x


def split_datetime_index(obj: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Split a datetime index into separate 'date' and 'time' levels.

    This function takes a Series or DataFrame with a datetime index (or datetime
    level in MultiIndex) and splits it into separate 'date' and 'time' components,
    creating a MultiIndex with these two levels.

    Parameters
    ----------
    obj : pd.Series or pd.DataFrame
        Input object with datetime index or datetime level in MultiIndex.

    Returns
    -------
    pd.Series or pd.DataFrame
        Object with MultiIndex containing separate 'date' and 'time' levels.

    Raises
    ------
    ValueError
        If the index or any level is not named 'datetime'.
    TypeError
        If input is not a pandas Series or DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2023-01-01 10:00:00', periods=3, freq='H')
    >>> s = pd.Series([1, 2, 3], index=dates)
    >>> s.index.name = 'datetime'
    >>> split_datetime_index(s)  # doctest: +NORMALIZE_WHITESPACE
    date        time
    2023-01-01  10:00:00    1
                11:00:00    2
                12:00:00    3
    dtype: int64
    """
    obj = obj.copy()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if obj.index.name == "datetime" or "datetime" in obj.index.names:
            if isinstance(obj.index, pd.MultiIndex):
                # Find the level named 'datetime'
                idx = obj.index.names.index("datetime")
                # Create new levels 'date' and 'time'
                new_levels = [level if level != "datetime" else ("date", "time") for level in obj.index.names]
                # Split 'datetime' level into 'date' and 'time'
                new_index = obj.index.to_frame(index=False)
                new_index[["date", "time"]] = new_index["datetime"].apply(lambda x: pd.Series([x.date(), x.time()]))
                new_index = new_index.drop(columns=["datetime"])
                obj.index = pd.MultiIndex.from_frame(new_index)
            else:
                # Handle a single index named 'datetime'
                new_index = pd.DataFrame(index=obj.index)
                new_index["date"] = new_index.index.date
                new_index["time"] = new_index.index.time
                obj.index = pd.MultiIndex.from_frame(new_index)
                obj.index.names = ["date", "time"]
        else:
            raise ValueError("The index or any level must be named 'datetime'")
    else:
        raise TypeError("Input must be a pandas Series or DataFrame")

    return obj


# def int2time(times: Iterable[int]):
#   return [dt.time(hour=t // 10000, minute=(t // 100) % 100, second=t%100) for t in times]


def ix2str(x: SeriesOrDataFrame, axis: Literal["index", "columns", 0, 1] = "index") -> SeriesOrDataFrame:
    """Convert index or column labels to string format (useful for display purposes, e.g. hvplot)

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        Input data structure.
    axis : {'index', 'columns', 0, 1}, default 'index'
        Which axis to convert to strings:
        - 'index' or 0: Convert index labels to strings
        - 'columns' or 1: Convert column labels to strings (DataFrame only)

    Returns
    -------
    pd.Series or pd.DataFrame
        Data structure with string-converted labels on specified axis.

    Raises
    ------
    ValueError
        If axis is 'columns' or 1 for a Series input.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2, 3], index=[10, 20, 30])
    >>> ix2str(s)
    10    1
    20    2
    30    3
    dtype: int64
    """
    x = x.copy(deep=False)
    if isinstance(x, pd.Series) and axis not in [0, "index"]:
        raise ValueError(f"{axis=} not supported for pd.Series")
    if axis in [0, "index"]:
        x.index = x.index.map(str)
    else:
        x.columns = x.columns.map(str)
    return x


def move_columns_to_position(df: pd.DataFrame, col_pos: dict[str, int]) -> pd.DataFrame:
    """Move specified columns to new positions while preserving order of remaining columns.

    This function reorders DataFrame columns by moving specified columns to
    designated positions. Columns not specified in col_pos maintain their
    relative order among themselves.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to reorder.
    col_pos : dict[str, int]
        Dictionary mapping column names to their desired positions.
        Positions can be negative (counted from the end).

    Returns
    -------
    pd.DataFrame
        DataFrame with reordered columns.

    Raises
    ------
    ValueError
        If a specified column is not in the DataFrame, if position is out of
        bounds, or if multiple columns are assigned to the same position.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]})
    >>> move_columns_to_position(df, {'D': 0, 'A': 2})
       D  B  A  C
    0  7  3  1  5
    1  8  4  2  6
    """
    for col, pos in col_pos.items():
        if col not in df.columns:
            raise ValueError(f"{col} is not in df.columns")
        if not -df.shape[1] <= pos < df.shape[1]:
            raise ValueError(f"{col} cannot be moved to position {pos} if DataFrame with {df.shape[1]} columns")
    col_pos = {col: (pos % df.shape[1]) for col, pos in col_pos.items()}
    if len(set(col_pos.values())) < len(col_pos):
        raise ValueError("some columns are being moved to same position")
    new_cols = list(df.columns)
    for col, pos in sorted(col_pos.items(), key=lambda x: x[1]):
        new_cols.remove(col)
        new_cols.insert(pos, col)
    return df[new_cols]


def isfinite(x: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Check for finite values (not NaN, not infinite).

    This function returns a boolean mask indicating which values are finite
    (neither NaN nor positive/negative infinity).

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        Input data to check for finite values.

    Returns
    -------
    pd.Series or pd.DataFrame
        Boolean mask with True for finite values, False for NaN or infinite values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series([1.0, np.nan, np.inf, -np.inf, 2.0])
    >>> isfinite(s)
    0     True
    1    False
    2    False
    3    False
    4     True
    dtype: bool
    """
    return ~(x.isna() | x.isin([-np.inf, np.inf]))


@overload
def describe_values(s: pd.Series, **kwargs) -> pd.Series: ...
@overload
def describe_values(df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...


def describe_values(s_or_df: SeriesOrDataFrame, **kwargs):
    """Generate enhanced descriptive statistics including NaN, zero, and negative percentages.

    This function extends pandas' describe() method by adding additional
    statistics about the proportion of NaN values, zeros, and negative values
    in the data.

    Parameters
    ----------
    s_or_df : pd.Series or pd.DataFrame
        Input data to describe.
    **kwargs
        Additional keyword arguments passed to pandas describe() method.

    Returns
    -------
    pd.Series or pd.DataFrame
        Descriptive statistics with additional rows:
        - pct_nans: Percentage of NaN values
        - pct_zeros: Percentage of zero values
        - pct_neg: Percentage of negative values

    Raises
    ------
    TypeError
        If input is neither Series nor DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series([1, -2, 0, np.nan, 5])
    >>> describe_values(s)  # doctest: +NORMALIZE_WHITESPACE
    count        4.00000
    mean         1.00000
    std          2.94392
    min         -2.00000
    25%         -0.50000
    50%          0.50000
    75%          2.00000
    max          5.00000
    pct_nans     0.20000
    pct_zeros    0.20000
    pct_neg      0.20000
    dtype: float64
    """
    if isinstance(s_or_df, pd.Series):
        res = s_or_df.describe(**kwargs)
        res.loc["pct_nans"] = s_or_df.isnull().sum() / s_or_df.size
        res.loc["pct_zeros"] = (s_or_df == 0).sum() / s_or_df.size
        res.loc["pct_neg"] = (s_or_df < 0).sum() / s_or_df.size
        return res
    elif isinstance(s_or_df, pd.DataFrame):
        res = pd.DataFrame({col: describe_values(s_or_df[col], **kwargs) for col in s_or_df.columns})
        return res
    else:
        raise TypeError(f"{type(s_or_df)=} not supported")


def ecdf_transform(data: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Transform data using Empirical Cumulative Distribution Function (ECDF).

    This function assigns each non-NaN value to its normalized rank, effectively
    transforming the data to a uniform distribution on [0, 1]. NaN values remain as NaN.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data to transform.

    Returns
    -------
    pd.Series or pd.DataFrame
        ECDF-transformed data with values between 0 and 1.
        NaN values remain as NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([10, 20, 30, 40])
    >>> ecdf_transform(s)
    0    0.25
    1    0.50
    2    0.75
    3    1.00
    dtype: float64
    """
    return data.rank(method="first") / len(data)


def get_most_recent_index_before(idx: pd.Index, key: Any, include: bool = True) -> Optional[Any]:
    """
    Return the largest index value <= key (if include=True) or < key (if include=False).
    Returns None if there is no such index value.

    Parameters
    - idx: pd.Index (must be sortable; MultiIndex is not supported)
    - key: lookup key (type should be comparable with index values; if index is datetime-like,
           key will be converted with pd.to_datetime)
    - include: bool. If True, an exact match for key is returned. If False, only strictly
               smaller values are considered.

    Returns
    - index value (e.g. pd.Timestamp or whatever the index stores) or None
    """
    if not isinstance(idx, pd.Index):
        raise TypeError("idx must be a pandas Index")
    if isinstance(idx, pd.MultiIndex):
        raise TypeError("MultiIndex is not supported by this helper")

    # If index is datetime-like, coerce the key to datetime
    if ptypes.is_datetime64_any_dtype(idx) or isinstance(idx, pd.DatetimeIndex):
        key = pd.to_datetime(key)

    # searchsorted requires an increasing-sorted index
    if not idx.is_monotonic_increasing:
        sidx = idx.sort_values()
    else:
        sidx = idx

    side = "right" if include else "left"
    pos = sidx.searchsorted(key, side=side) - 1
    return sidx[pos] if pos >= 0 else None


def tabulator(
    df: pd.DataFrame,
    *,
    editable=False,
    height=500,
    page_size: int = None,
    header_filters=True,
    freeze_index=True,
    **kwargs,
):
    """Create an interactive Tabulator widget for displaying and editing DataFrame data.

    Parameters
    ----------
    editable : bool, default False
        Whether the table should be editable.
        (!!!) If True, users can modify cell values and this updates dataframe inplace (!!!)
    height : int, default 500
        The height of the table widget in pixels. Controls the vertical space
        allocated to the table display. If does not fit, then scroll.
    page_size : int, optional
        If specified, enables local pagination with `page_size` records.
    header_filters : bool, default True
        Whether to enable header filters for columns. When enabled, adds filter
        controls to column headers. Boolean columns get specialized tickCross filters.
    freeze_index : bool, default False
        Whether to freeze columns corresponding to the DataFrame index. If True,
        index columns will be frozen (remain visible when scrolling horizontally).
        For MultiIndex, all index levels will be frozen.

    Returns
    -------
    panel.widgets.Tabulator
        A Panel Tabulator widget configured with the DataFrame data and specified
        options. The widget can be displayed in Jupyter notebooks, Panel applications,
        or served as a web application.

    Notes
    -----
    This method requires the Panel library to be installed and automatically enables
    the Tabulator extension.
    See :panel.widgets.Tabulator: for additional options
    """
    import panel as pn

    pn.extension("tabulator")
    pagination = None if page_size is None else "local"

    # Configure filters for boolean columns if header_filters is enabled
    filters = None
    if header_filters:
        filters = {}
        # override default filter for boolean columns
        for col in df.select_dtypes(include="bool").columns:
            filters[col] = {"type": "tickCross", "tristate": True, "indeterminateValue": None}

        # for col in s_or_df.select_dtypes(include=('int', 'float')).columns:
        #   filters[col] = {'type': 'number', 'placeholder': 'min'}
        # print(filters)

    # Configure frozen columns for index if freeze_index is enabled
    frozen_cols = None
    if freeze_index:
        # Get index column names - Panel Tabulator internally calls reset_index()
        # so index levels become regular columns with their names
        if isinstance(df.index, pd.MultiIndex):
            # For MultiIndex, freeze all index level columns by name
            frozen_cols = list(df.index.names)
        else:
            # For regular index, freeze the index column by name
            # If index has no name, Panel will use 'index' as the column name
            index_name = df.index.name if df.index.name is not None else "index"
            frozen_cols = [index_name]

    # Build the tabulator arguments
    tabulator_args = {
        "disabled": not editable,
        "height": height,
        "pagination": pagination,
        "page_size": page_size,
        "header_filters": True,
        "editors": filters,
        "configuration": {"clipboard": True},
        **kwargs,
    }

    # Only add frozen_columns if we have columns to freeze
    if frozen_cols is not None:
        tabulator_args["frozen_columns"] = frozen_cols

    return pn.widgets.Tabulator(df, **tabulator_args)


@pd.api.extensions.register_index_accessor("ms")
class MShevelevAccessor:
    """Custom pandas Index accessor providing additional utility methods.

    This accessor adds the 'ms' namespace to pandas Index objects, providing
    convenient methods for index manipulation and querying.

    """

    def __init__(self, index):
        self._validate(index)
        self._index = index

    @staticmethod
    def _validate(obj):
        """Validate that the object is a pandas Index."""
        if not isinstance(obj, pd.Index):
            raise TypeError("object must be pd.Index")

    def get_most_recent_index_before(self, key: Any, include: bool = True) -> Optional[Any]:
        """
        Return the largest index value <= key (if include=True) or < key (if include=False).
        Returns None if there is no such index value.

        Parameters
        - idx: pd.Index (must be sortable; MultiIndex is not supported)
        - key: lookup key (type should be comparable with index values; if index is datetime-like,
               key will be converted with pd.to_datetime)
        - include: bool. If True, an exact match for key is returned. If False, only strictly
                   smaller values are considered.

        Returns
        - index value (e.g. pd.Timestamp or whatever the index stores) or None
        """
        return get_most_recent_index_before(self._index, key, include=include)

    def between(self, left, right, inclusive: Literal["both", "neither", "left", "right"] = "both") -> pd.Index:
        """Return index values between left and right boundaries.

        Parameters
        ----------
        left : scalar
            Left boundary for the range.
        right : scalar
            Right boundary for the range.
        inclusive : {'both', 'neither', 'left', 'right'}, default 'both'
            Include boundaries in the result:
            - 'both': Include both boundaries
            - 'neither': Exclude both boundaries
            - 'left': Include only left boundary
            - 'right': Include only right boundary

        Returns
        -------
        pd.Index
            Index values within the specified range.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([1, 2, 3, 4, 5])
        >>> idx.ms.between(2, 4)
        Int64Index([2, 3, 4], dtype='int64')
        >>> idx.ms.between(2, 4, inclusive='neither')
        Int64Index([3], dtype='int64')
        """
        return self._index[self._index.to_series().between(left, right, inclusive=inclusive)]


@pd.api.extensions.register_series_accessor("ms")
class MShevelevAccessor:
    """Custom pandas Series accessor providing additional utility methods.

    This accessor adds the 'ms' namespace to pandas Series objects, providing
    convenient methods for data manipulation, visualization, and analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> s.ms.normalize()  # Normalize the series
    >>> s.ms.ecdf()  # ECDF transformation
    >>> s.ms.describe_values()  # Enhanced descriptive statistics
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Validate that the object is a pandas Series."""
        if not isinstance(obj, pd.Series):
            raise TypeError("object must be pd.Series")

    @property
    def style(self):
        """Return Styler object for the Series (converted to DataFrame).

        Returns
        -------
        pandas.io.formats.style.Styler
            Styler object for formatting and display.
        """
        return self._obj.to_frame().style

    def ix2date(self, format="%Y%m%d") -> pd.Series:
        """Convert index from YYYYMMDD format to datetime objects.

        Returns
        -------
        pd.Series
            Series with datetime-converted index.
        """
        return ix2date(self._obj, format=format)

    def ix2dt(self) -> pd.Series:
        """Convert date/time index levels to datetime format. See :ix2dt: docs.

        Returns
        -------
        pd.Series
            Series with datetime-converted index.
        """
        return ix2dt(self._obj)

    def ix2str(self) -> pd.Series:
        """Convert index labels to string format. See :ix2str: docs.

        Returns
        -------
        pd.Series
            Series with string-converted index.
        """
        return ix2str(self._obj)

    def trim_nans(self, how="any") -> pd.Series:
        """Remove NaN values from beginning and end of Series. See :trim_nans: docs.

        Parameters
        ----------
        how : {'any', 'all'}, default 'any'
            Logic for NaN detection.

        Returns
        -------
        pd.Series
            Trimmed Series.
        """
        return trim_nans(self._obj, how=how)

    def normalize(self, *, ord=1, skipna=True) -> pd.Series:
        """Normalize the Series using vector norms. See :normalize: docs.

        Parameters
        ----------
        ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, default 1
            Order of the norm.
        skipna : bool, default True
            Whether to skip NaN values.

        Returns
        -------
        pd.Series
            Normalized Series.
        """
        return normalize(self._obj, ord=ord, axis=0, skipna=skipna)

    def isfinite(self):
        """Check for finite values (not NaN, not infinite). See :isfinite: docs.

        Returns
        -------
        pd.Series
            Boolean mask for finite values.
        """
        return isfinite(self._obj)

    def ecdf(self) -> pd.Series:
        """Apply ECDF transformation to the Series. See :ecdf_transform: docs.

        Returns
        -------
        pd.Series
            ECDF-transformed Series with values between 0 and 1.
        """
        return ecdf_transform(self._obj)

    def describe_values(self, *, style=False, **kwargs):
        """Generate enhanced descriptive statistics. See :describe_values: docs.

        Parameters
        ----------
        style : bool, default False
            Whether to return styled output.
        **kwargs
            Additional arguments passed to pandas describe().

        Returns
        -------
        pd.Series or Styler
            Enhanced descriptive statistics.
        """
        res = describe_values(self._obj, **kwargs)
        if style:
            return pd_styler.style_describe_table(res.to_frame().style)
        return res

    @property
    def hvplot(self):
        """HoloViews plotting interface with datetime index conversion preliminary applied.

        Returns
        -------
        hvplot accessor
            HoloViews plotting interface.
        """
        return self._obj.ms.ix2dt().hvplot

    def hvplot_ecdf(self, kind: Literal["scatter", "line"] = "scatter") -> Union[hv.Curve, hv.Scatter]:
        """Create ECDF plot using HoloViews. See .holoviews.plot_ecdf: docs.

        Parameters
        ----------
        kind : {'scatter', 'line'}, default 'scatter'
            Type of plot to create.

        Returns
        -------
        hv.Curve or hv.Scatter
            HoloViews ECDF plot.
        """
        from .holoviews import plot_ecdf

        return plot_ecdf(self._obj, kind=kind)

    def hvplot_qqplot(self):
        """Create Q-Q plot using HoloViews. See :.holoviews.hvplot_qqplot: docs.

        Returns
        -------
        HoloViews plot
            Q-Q plot visualization.
        """
        from .holoviews import hvplot_qqplot

        return hvplot_qqplot(self._obj)

    def html_display(self, max_rows=60, show_dimensions=True, **kwargs) -> HTML:
        """Return an HTML representation of the Series.

        Parameters
        ----------
        max_rows : int, default 60
            Maximum number of rows to include in the HTML.
        show_dimensions : bool, default True
            Whether to include the dimensions of the Series in the HTML.
        **kwargs
            Additional keyword arguments passed to DataFrame.to_html().

        Returns
        -------
        IPython.display.HTML
            An HTML object that can be displayed in a Jupyter environment.
        """
        # Convert Series to DataFrame for HTML representation
        df = self._obj.to_frame()
        html = df.to_html(max_rows=max_rows, show_dimensions=show_dimensions, **kwargs)
        return HTML(html)

    def tabulator(
        self, *, editable=False, height=500, page_size: int = None, header_filters=True, freeze_index=True, **kwargs
    ):
        """Create panel.Tabulator representation

        see `tabulator` method for docs
        """

    def add_fake_rows(self, breaks: Union[str, Iterable], fake_value=np.nan) -> pd.Series:
        """Add fake rows to the Series at specified intervals or positions. See :add_fake_rows: docs.

        Parameters
        ----------
        breaks : str or iterable
            Specification for where to add breaks.
        fake_value : scalar, default np.nan
            The value to insert at break positions.

        Returns
        -------
        pd.Series
            Series with fake rows inserted at specified positions.
        """
        return add_fake_rows(self._obj, breaks, fake_value)

        return tabulator(
            self._obj.to_frame(),
            editable=editable,
            height=height,
            page_size=page_size,
            header_filters=header_filters,
            freeze_index=freeze_index,
            **kwargs,
        )


@pd.api.extensions.register_dataframe_accessor("ms")
class MShevelevAccessor:
    """Custom pandas DataFrame accessor providing additional utility methods.

    This accessor adds the 'ms' namespace to pandas DataFrame objects, providing
    convenient methods for data manipulation, visualization, and analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> normalized = df.ms.normalize()  # Normalize the DataFrame
    >>> flattened = df.ms.flatten_columns()  # Flatten MultiIndex columns
    >>> stats = df.ms.describe_values()  # Enhanced descriptive statistics
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Validate that the object is a pandas DataFrame."""
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("object must be pd.DataFrame")

    @property
    def style(self):
        """Return Styler object for the DataFrame.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styler object for formatting and display.
        """
        return self._obj.style

    def ix2date(self, format="%Y%m%d") -> pd.DataFrame:
        """Convert index from YYYYMMDD format to datetime objects. See :ix2date: docs."""
        return ix2date(self._obj, format=format)

    def ix2dt(self) -> pd.DataFrame:
        """Convert date/time index levels to datetime format. See :ix2dt: docs."""
        return ix2dt(self._obj)

    def ix2str(self, axis: Literal["index", "columns", 0, 1] = "index") -> pd.DataFrame:
        """Convert index or column labels to string format. See :ix2str: docs."""
        return ix2str(self._obj, axis=axis)

    def trim_nans(self, how="any", *, subset=None) -> pd.DataFrame:
        """Remove NaN values from beginning and end of DataFrame. See :trim_nans: docs."""
        return trim_nans(self._obj, how=how, subset=subset)

    def normalize(self, axis=0, *, ord=1, skipna=True) -> pd.DataFrame:
        """Normalize the DataFrame using vector norms. See :normalize: docs."""
        return normalize(self._obj, ord=ord, axis=axis, skipna=skipna)

    def isfinite(self):
        """Check for finite values (not NaN, not infinite). See :isfinite: docs."""
        return isfinite(self._obj)

    def ecdf(self) -> pd.DataFrame:
        """Apply ECDF transformation to the DataFrame. See :ecdf_transform: docs."""
        return ecdf_transform(self._obj)

    def hvplot_ecdf(self, kind: Literal["scatter", "line"] = "scatter") -> hv.NdOverlay:
        """Create ECDF plot using HoloViews. See :.holoviews.plot_ecdf: docs."""
        from .holoviews import plot_ecdf

        return plot_ecdf(self._obj, kind=kind)

    def hvplot_heatmap(
        self, *, annotate: bool = False, annotate_format: str = "{:.2f}"
    ) -> Union[hv.HeatMap, hv.Overlay]:
        """Create a heatmap of the DataFrame using HoloViews.

        DataFrame is assumed to be in a "rectangular" form (not in a tidy form).

        Parameters
        ----------
        annotate : bool, default False
            If True, overlay the heatmap with text labels showing the values.
        annotate_format : str, default "{:.2f}"
            Format string for the annotation text. Only used when annotate=True.
            Examples: "{:.2f}" for 2 decimal places, "{:.2%}" for percentages.

        Returns
        -------
        hv.HeatMap or hv.Overlay
            A HoloViews HeatMap object, or an Overlay with Labels when annotate=True.
            When limits are specified, the underlying data is subsampled to show
            only the selected rows/columns.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame(np.random.randn(20, 15))
        >>> heatmap = df.ms.hvplot_heatmap()
        >>> annotated_heatmap = df.ms.hvplot_heatmap(annotate=True, annotate_format="{:.1f}")

        """
        df = self._obj
        df_tidy = df.stack().rename("values").reset_index()

        # After stacking and reset_index(), the columns are named 'level_0' and 'level_1'
        # We can use the actual column names from the tidy DataFrame
        kdims = df_tidy.columns[:2].tolist()  # First two columns are the dimensions

        heatmap = hv.HeatMap(df_tidy, kdims=kdims).opts(cmap="coolwarm")

        if not annotate:
            return heatmap

        # Create annotations using hv.Labels
        # Format the values according to the specified format string
        df_labels = df_tidy.copy()
        df_labels["text"] = df_labels["values"].apply(lambda x: annotate_format.format(x))

        # Create Labels element with the same kdims as the heatmap, plus 'text' as vdim
        labels = hv.Labels(df_labels, kdims=kdims, vdims=["text"]).opts(text_color="black")

        # Return overlay of heatmap and labels
        return heatmap * labels

    def describe_values(self, *, style=False, **kwargs) -> pd.DataFrame:
        """Generate enhanced descriptive statistics. See :describe_values: docs."""
        res = describe_values(self._obj, **kwargs)
        if style:
            return pd_styler.style_describe_table(res.style)
        return res

    def flatten_columns(self) -> pd.DataFrame:
        """Flatten MultiIndex columns to single-level columns. See :flatten_columns: docs."""
        return flatten_columns(self._obj)

    def move_columns_to_position(self, col_pos: dict[str, int]) -> pd.DataFrame:
        """Move specified columns to new positions. See :move_columns_to_position: docs."""
        return move_columns_to_position(self._obj, col_pos)

    def prepend(self, index, x=0.0) -> pd.DataFrame:
        """Prepend a row or rows to the DataFrame.

        Parameters
        ----------
        index : Index or list-like
            The index for the new row(s).
        x : scalar or list-like, default 0.
            The value(s) to fill the new row(s) with.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the new row(s) prepended.
        """
        assert isinstance(self._obj, pd.DataFrame)
        pre_df = pd.DataFrame(index=index, data=[pd.Series(x, index=self._obj.columns)])
        res = pd.concat([pre_df, self._obj])
        return res

    def display(self, max_rows=20, max_columns=11, **kwargs):
        """Display the DataFrame in a Jupyter environment with custom options.

        Parameters
        ----------
        max_rows : int, default 20
            Maximum number of rows to display.
        max_columns : int, default 11
            Maximum number of columns to display.
        **kwargs
            Additional keyword arguments passed to pandas.option_context.
        """
        from IPython.display import display

        with pd.option_context(
            "display.max_rows", max_rows, "display.max_columns", max_columns, *itertools.chain(*kwargs.items())
        ):
            display(self._obj)

    def html_display(self, max_rows=20, max_cols=11, show_dimensions=True, **kwargs) -> HTML:
        """Return an HTML representation of the DataFrame.

        Parameters
        ----------
        max_rows : int, default 20
            Maximum number of rows to include in the HTML.
        max_cols : int, default 11
            Maximum number of columns to include in the HTML.
        show_dimensions : bool, default True
            Whether to include the dimensions of the DataFrame in the HTML.
        **kwargs
            Additional keyword arguments passed to DataFrame.to_html().

        Returns
        -------
        IPython.display.HTML
            An HTML object that can be displayed in a Jupyter environment.
        """
        html = self._obj.to_html(max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions, **kwargs)
        return HTML(html)

    def to_dataarray(self) -> xr.DataArray:
        """Convert the DataFrame to an xarray.DataArray.

        Returns
        -------
        xr.DataArray
            An xarray.DataArray representation of the DataFrame.
        """
        return xr.DataArray(self._obj, coords=[self._obj.index, self._obj.columns])

    @property
    def hvplot(self):
        """HoloViews plotting interface with preliminary column flattening and datetime index conversion."""
        df = flatten_columns(self._obj)
        return df.ms.ix2dt().hvplot

    def tabulator(
        self, *, editable=False, height=500, page_size: int = None, header_filters=True, freeze_index=True, **kwargs
    ):
        """Create panel.Tabulator representation

        see `tabulator` method for docs
        """
        return tabulator(
            self._obj,
            editable=editable,
            height=height,
            page_size=page_size,
            header_filters=header_filters,
            freeze_index=freeze_index,
            **kwargs,
        )

    def add_fake_rows(self, breaks: Union[str, Iterable], fake_value=np.nan) -> pd.DataFrame:
        """Add fake rows to the DataFrame at specified intervals or positions. See :add_fake_rows: docs.

        Parameters
        ----------
        breaks : str or iterable
            Specification for where to add breaks.
        fake_value : scalar, default np.nan
            The value to insert at break positions.

        Returns
        -------
        pd.DataFrame
            DataFrame with fake rows inserted at specified positions.
        """
        return add_fake_rows(self._obj, breaks, fake_value)
