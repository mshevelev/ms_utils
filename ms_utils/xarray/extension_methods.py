"""Refactored xarray extension methods using register_method decorator.

This module contains xarray extension methods that use the @register_method decorator
to automatically register as accessor methods in the `.ms` namespace for DataArray and Dataset.

Key improvements over the old approach:
- Single source of truth (one function, one docstring)
- Docstrings automatically propagate to accessor methods
- No duplicate wrapper methods needed
- Cleaner, more maintainable code

All core utility functions migrated with comprehensive docstrings.
"""

import numpy as np
import pandas as pd
import xarray as xr
import functools
from typing import Union, Literal, TypeVar
from ms_utils.method_registration import register_method

T = TypeVar("T")
DataArrayOrDataset = TypeVar("DataArrayOrDataset", bound=Union[xr.DataArray, xr.Dataset])


def xarray_ufunc(func):
    """Wraps a function that works on np.ndarray with xr.apply_ufunc.
    
    Only works for ufuncs and not on gufuncs, i.e. no core dimensions.
    
    Parameters
    ----------
    func : callable
        Function to wrap.
    
    Returns
    -------
    callable
        Wrapped function that works with xarray objects.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return xr.apply_ufunc(func, *args, kwargs=kwargs)
    return wrapper


@register_method([xr.DataArray, xr.Dataset])
def ix2date(obj: DataArrayOrDataset, dims: tuple[str] = ("date",)) -> DataArrayOrDataset:
    """Convert integer date coordinates to datetime format.
    
    Converts coordinates in YYYYMMDD integer format to datetime for specified dimensions.
    Useful when working with date coordinates stored as integers.
    
    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Input xarray object with integer date coordinates.
    dims : tuple of str, default ('date',)
        Dimension names to convert from integer to datetime.
    
    Returns
    -------
    xr.DataArray or xr.Dataset
        Object with converted datetime coordinates.
    
    Examples
    --------
    **Convert date dimension from integers:**
    
    >>> import xarray as xr
    >>> da = xr.DataArray([1, 2, 3], coords={'date': [20230101, 20230102, 20230103]}, dims=['date'])
    >>> da.ms.ix2date()
    # Date coordinates are now datetime objects
    
    See Also
    --------
    ix2int : Convert datetime coordinates to integers
    pd.to_datetime : Convert argument to datetime
    
    Notes
    -----
    - Only processes dimensions with integer dtype
    - Uses YYYYMMDD format for conversion
    - Original object is not modified (returns copy with new coords)
    """
    for dim in dims:
        if pd.api.types.is_integer_dtype(obj.get_index(dim)):
            new_coords = pd.to_datetime(obj.get_index(dim), format="%Y%m%d")
            obj = obj.assign_coords({dim: new_coords})
    return obj


@register_method([xr.DataArray, xr.Dataset])
def ix2int(obj: DataArrayOrDataset, dims: tuple[str] = ("date",)) -> DataArrayOrDataset:
    """Convert datetime coordinates to YYYYMMDD integer format.
    
    Converts datetime coordinates to integer format (YYYYMMDD) for specified dimensions.
    Useful for storage or when integer coordinates are preferred.
    
    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Input xarray object with datetime coordinates.
    dims : tuple of str, default ('date',)
        Dimension names to convert from datetime to integer.
    
    Returns
    -------
    xr.DataArray or xr.Dataset
        Object with integer date coordinates.
    
    Examples
    --------
    **Convert datetime to integers:**
    
    >>> import xarray as xr
    >>> import pandas as pd
    >>> dates = pd.date_range('2023-01-01', periods=3)
    >>> da = xr.DataArray([1, 2, 3], coords={'date': dates}, dims=['date'])
    >>> da.ms.ix2int()
    # Date coordinates are now integers: 20230101, 20230102, 20230103
    
    See Also
    --------
    ix2date : Convert integer coordinates to datetime
    
    Notes
    -----
    - Only processes dimensions with datetime dtype
    - Converts to YYYYMMDD format
    - Original object is not modified
    """
    for dim in dims:
        if pd.api.types.is_datetime64_ns_dtype(obj.get_index(dim)):
            new_coords = obj.get_index(dim).strftime("%Y%m%d").astype(int)
            obj = obj.assign_coords({dim: new_coords})
    return obj


@register_method([xr.DataArray, xr.Dataset])
def describe_index(obj: Union[xr.DataArray, xr.Dataset]) -> list[dict]:
    """Describe all dimensions and their indexes.
    
    Returns information about each dimension including its dtype and length.
    Useful for quickly understanding the structure of an xarray object.
    
    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Input xarray object.
    
    Returns
    -------
    list of dict
        List of dictionaries, each containing:
        - 'dim': dimension name
        - 'dtype': data type of the index
        - 'len': length of the dimension
    
    Examples
    --------
    **Describe dimensions:**
    
    >>> import xarray as xr
    >>> import pandas as pd
    >>> da = xr.DataArray(
    ...     np.random.rand(10, 5),
    ...     coords={'date': pd.date_range('2023-01-01', periods=10), 'stock': list('ABCDE')},
    ...     dims=['date', 'stock']
    ... )
    >>> da.ms.describe_index()
    [{'dim': 'date', 'dtype': numpy.datetime64, 'len': 10},
     {'dim': 'stock', 'dtype': numpy.object_, 'len': 5}]
    
    See Also
    --------
    xr.DataArray.dims : Dimension names
    xr.DataArray.coords : Coordinates
    """
    res = []
    for k, ix in obj.indexes.items():
        res.append({"dim": k, "dtype": ix.dtype.type, "len": len(ix)})
    return res


@register_method([xr.DataArray, xr.Dataset])
def between(obj: T, left, right, inclusive: Literal["both", "neither", "left", "right"] = "both") -> T:
    """Filter values between left and right bounds.
    
    Element-wise check if values fall between left and right bounds.
    Returns a boolean array with the same shape as input.
    
    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Input data to filter.
    left : scalar
        Left boundary.
    right : scalar
        Right boundary.
    inclusive : {'both', 'neither', 'left', 'right'}, default 'both'
        Which boundaries to include:
        
        - ``'both'``: left <= value <= right
        - ``'neither'``: left < value < right
        - ``'left'``: left <= value < right
        - ``'right'``: left < value <= right
    
    Returns
    -------
    xr.DataArray or xr.Dataset
        Boolean mask with True where values are between bounds.
    
    Examples
    --------
    **Filter values in range:**
    
    >>> import xarray as xr
    >>> da = xr.DataArray([1, 2, 3, 4, 5])
    >>> mask = da.ms.between(2, 4)
    # Returns: [False, True, True, True, False]
    
    **Exclusive bounds:**
    
    >>> mask = da.ms.between(2, 4, inclusive='neither')
    # Returns: [False, False, True, False, False]
    
    See Also
    --------
    pd.Series.between : Pandas equivalent
    
    Notes
    -----
    - Works element-wise on all values in the array/dataset
    - Returns boolean mask that can be used for filtering
    """
    if inclusive == "both":
        return (left <= obj) & (obj <= right)
    elif inclusive == "neither":
        return (left < obj) & (obj < right)
    elif inclusive == "left":
        return (left <= obj) & (obj < right)
    elif inclusive == "right":
        return (left < obj) & (obj <= right)
    else:
        raise ValueError(f"inclusive must be 'both', 'neither', 'left', or 'right', got {inclusive}")


@register_method([xr.DataArray])
def get_index_range(da: xr.DataArray, dim: str = "date") -> tuple:
    """Get the first and last values of a dimension's index.
    
    Useful for quickly checking the range of a dimension without examining all values.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    dim : str, default 'date'
        Dimension name to get range for.
    
    Returns
    -------
    tuple
        (first_value, last_value) of the dimension.
    
    Examples
    --------
    **Get date range:**
    
    >>> import xarray as xr
    >>> import pandas as pd
    >>> dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    >>> da = xr.DataArray(range(len(dates)), coords={'date': dates}, dims=['date'])
    >>> da.ms.get_index_range('date')
    (Timestamp('2023-01-01'), Timestamp('2023-12-31'))
    
    See Also
    --------
    xr.DataArray.indexes : Access dimension indexes
    """
    idx = da.get_index(dim)
    return idx[0], idx[-1]


@register_method([xr.DataArray])
def binary_logical_op_on_union(
    da1: xr.DataArray,
    da2: xr.DataArray,
    union_dim: str = "stock",
    op: str = "and",
) -> xr.DataArray:
    """Apply binary logical operation on union of coordinates.
    
    Performs element-wise binary logical operations on two boolean DataArrays
    whose coordinates may differ along one dimension. The operation is applied
    on the union of coordinate labels, treating missing values as False.
    
    Parameters
    ----------
    da1 : xr.DataArray
        First boolean DataArray.
    da2 : xr.DataArray
        Second boolean DataArray.
    union_dim : str, default 'stock'
        Dimension along which to take the union of coordinates.
    op : {'and', 'or', 'diff', 'symmetric_diff'}, default 'and'
        Binary logical operation:
        
        - ``'and'``: Logical AND (intersection)
        - ``'or'``: Logical OR (union)
        - ``'diff'``: Difference (da1 AND NOT da2)
        - ``'symmetric_diff'``: Exclusive OR (XOR)
    
    Returns
    -------
    xr.DataArray
        Result of the operation on the union of coordinates.
    
    Raises
    ------
    AssertionError
        If inputs are not boolean dtype.
    ValueError
        If union_dim not in both arrays or op is invalid.
    
    Examples
    --------
    **Intersection of two boolean arrays:**
    
    >>> import xarray as xr
    >>> da1 = xr.DataArray([True, False, True], coords={'stock': ['A', 'B', 'C']}, dims=['stock'])
    >>> da2 = xr.DataArray([True, True], coords={'stock': ['B', 'C']}, dims=['stock'])
    >>> result = binary_logical_op_on_union(da1, da2, union_dim='stock', op='and')
    # Result has union of stocks: A, B, C
    # A: False (True AND False), B: False, C: True
    
    See Also
    --------
    xr.DataArray.reindex : Conform DataArray to new index
    xr.align : Align two or more objects
    
    Notes
    -----
    - Missing values in either array are treated as False
    - Arrays are aligned on other dimensions using outer join
    - Useful for combining boolean masks from different datasets
    """
    assert da1.dtype == bool, f"{da1.dtype=}, but 'bool' expected"
    assert da2.dtype == bool, f"{da2.dtype=}, but 'bool' expected"
    
    if union_dim not in da1.dims or union_dim not in da2.dims:
        raise ValueError(f"Both inputs must contain dimension {union_dim!r}")

    # Build union of coordinates
    union_coords = np.union1d(da1[union_dim].values, da2[union_dim].values)

    # Reindex to union, treating missing as False
    da1_u = da1.reindex({union_dim: union_coords}, fill_value=False)
    da2_u = da2.reindex({union_dim: union_coords}, fill_value=False)

    # Align other dimensions
    da1_aligned, da2_aligned = xr.align(da1_u, da2_u, join="outer", fill_value=False)

    # Apply operation
    if op == "and":
        return da1_aligned & da2_aligned
    elif op == "or":
        return da1_aligned | da2_aligned
    elif op == "diff":
        return da1_aligned & ~da2_aligned
    elif op == "symmetric_diff":
        return da1_aligned ^ da2_aligned
    else:
        raise ValueError(f"Unsupported operation: {op!r}")


# Convenience methods using binary_logical_op_on_union
@register_method([xr.DataArray])
def intersection(da: xr.DataArray, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
    """Element-wise logical AND on union of coordinates.
    
    Computes the intersection of two boolean DataArrays on the union of coordinates
    along the specified dimension.
    
    Parameters
    ----------
    da : xr.DataArray
        Boolean DataArray.
    other : xr.DataArray
        Boolean DataArray to intersect with.
    union_dim : str, default 'stock'
        Dimension for coordinate union.
    
    Returns
    -------
    xr.DataArray
        Intersection result.
    
    Examples
    --------
    >>> da1 = xr.DataArray([True, True], coords={'stock': ['A', 'B']}, dims=['stock'])
    >>> da2 = xr.DataArray([True, False], coords={'stock': ['B', 'C']}, dims=['stock'])
    >>> da1.ms.intersection(da2)
    # Result: A=False, B=True, C=False
    """
    return binary_logical_op_on_union(da, other, union_dim, op="and")


@register_method([xr.DataArray])
def union(da: xr.DataArray, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
    """Element-wise logical OR on union of coordinates.
    
    Computes the union of two boolean DataArrays on the union of coordinates
    along the specified dimension.
    
    Parameters
    ----------
    da : xr.DataArray
        Boolean DataArray.
    other : xr.DataArray
        Boolean DataArray to union with.
    union_dim : str, default 'stock'
        Dimension for coordinate union.
    
    Returns
    -------
    xr.DataArray
        Union result.
    """
    return binary_logical_op_on_union(da, other, union_dim, op="or")


@register_method([xr.DataArray])
def difference(da: xr.DataArray, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
    """Element-wise difference on union of coordinates.
    
    Computes elements that are True in this array but False in the other.
    
    Parameters
    ----------
    da : xr.DataArray
        Boolean DataArray.
    other : xr.DataArray
        Boolean DataArray to compute difference with.
    union_dim : str, default 'stock'
        Dimension for coordinate union.
    
    Returns
    -------
    xr.DataArray
        Difference result (self AND NOT other).
    """
    return binary_logical_op_on_union(da, other, union_dim, op="diff")


@register_method([xr.DataArray])
def symmetric_difference(da: xr.DataArray, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
    """Element-wise symmetric difference (XOR) on union of coordinates.
    
    Computes elements that are True in exactly one of the two arrays.
    
    Parameters
    ----------
    da : xr.DataArray
        Boolean DataArray.
    other : xr.DataArray
        Boolean DataArray to compute symmetric difference with.
    union_dim : str, default 'stock'
        Dimension for coordinate union.
    
    Returns
    -------
    xr.DataArray
        Symmetric difference result (XOR).
    """
    return binary_logical_op_on_union(da, other, union_dim, op="symmetric_diff")
