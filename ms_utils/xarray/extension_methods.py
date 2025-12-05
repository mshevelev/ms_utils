import numpy as np
import pandas as pd
import xarray as xr
import functools
from typing import Union, Literal, TypeVar

T = TypeVar("T")


def xarray_ufunc(func):
    """Wraps a function that works on np.ndarray with xr.apply_ufunc.
    Only works for ufuncs and not on gufuncs, i.e. no core dimensions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return xr.apply_ufunc(func, *args, kwargs=kwargs)

    return wrapper


def convert_dates_to_ints(x: xr.DataArray, dims=("date",)) -> xr.DataArray:
    for dim in dims:
        if pd.api.types.is_datetime64_ns_dtype(x.get_index(dim)):
            new_coords = x.get_index(dim).strftime("%Y%m%d").astype(int)
            x = x.assign_coords({dim: new_coords})
    return x


def convert_ints_to_dates(x: xr.DataArray, dims=("date",)) -> xr.DataArray:
    for dim in dims:
        if pd.api.types.is_integer_dtype(x.get_index(dim)):
            new_coords = pd.to_datetime(x.get_index(dim), format="%Y%m%d")
            x = x.assign_coords({dim: new_coords})
    return x


def describe_index(x: Union[xr.Dataset, xr.DataArray]):
    res = []
    for k, ix in x.indexes.items():
        res.append({"dim": k, "dtype": ix.dtype.type, "len": len(ix)})
    return res


def between(x: T, left, right, inclusive: Literal["both", "neither", "left", "right"] = "both") -> T:
    if inclusive == "both":
        return (left <= x) & (x <= right)
    elif inclusive == "neither":
        return (left < x) & (x < right)
    elif inclusive == "left":
        return (left <= x) & (x < right)
    elif inclusive == "right":
        return (left < x) & (x <= right)
    else:
        raise ValueError(f"inclusive must be either 'both', 'neither' or 'left' or 'right'")


def binary_logical_op_on_union(
    da1: xr.DataArray,
    da2: xr.DataArray,
    union_dim: str = "stock",
    op: str = "and",
) -> xr.DataArray:
    """
    Element-wise binary logical operation on two (boolean) DataArrays whose coordinates
    may differ along one particular dimension (default: 'stock').

    The operation is applied on the union of the coordinate labels along `union_dim`.
    Missing values are treated as False.

    Parameters
    ----------
    da1, da2 : xr.DataArray
        Boolean DataArrays on which the binary operation is applied.
    union_dim : str, default "stock"
        Name of the dimension along which the union of coordinates should be taken.
    op : str, default "and"
        Binary logical operation to be applied. Supported operations are:
        - "and": Element-wise logical AND (intersection).
        - "or": Element-wise logical OR (union).
        - "diff": Difference (da1 and not da2).
        - "symmetric_diff": Exclusive OR (elements present in one but not both).

    Returns
    -------
    xr.DataArray
        Array resulting from applying the operation on the union of `union_dim`.
    """
    assert da1.dtype == bool, f"{da1.dtype=}, but 'bool' expected"
    assert da2.dtype == bool, f"{da2.dtype=}, but 'bool' expected"
    # --- 1. Build the union of coordinates along the chosen dimension ----------
    if union_dim not in da1.dims or union_dim not in da2.dims:
        raise ValueError(f"Both inputs must contain the dimension {union_dim!r}")

    union_coords = np.union1d(da1[union_dim].values, da2[union_dim].values)

    # --- 2. Re-index each array to that union, treating missing data as False ---
    da1_u = da1.reindex({union_dim: union_coords}, fill_value=False)
    da2_u = da2.reindex({union_dim: union_coords}, fill_value=False)

    # Make sure *other* dimensions are aligned as well (intersection by default)
    da1_aligned, da2_aligned = xr.align(da1_u, da2_u, join="outer", fill_value=False)

    # --- 3. Apply binary logical operation ------------------------------------
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


@xr.register_dataarray_accessor("ms")
class MshevelevAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._series = None

    def _get_series(self) -> pd.Series:
        if self._series is None:
            self._series = self._obj.to_series()
        return self._series

    def ix2date(self, dims: tuple[str] = ("date",)):
        return convert_ints_to_dates(self._obj, dims)

    def ix2int(self, dims: tuple[str] = ("date",)):
        return convert_dates_to_ints(self._obj, dims)

    def to_pandas(self, i2d=True):
        x = self._obj
        # if i2d:
        #   x = x.ms.ix2date()
        res = x.to_pandas()
        if i2d:
            res = res.ms.ix2dt()
        if isinstance(res, pd.Series):
            res = res.rename(x.name)
        return res

    def to_pd(self, i2d=True):
        """Alias for to_pandas method."""
        return self.to_pandas(i2d=i2d)

    def get_index_range(self, dim="date"):
        idx = self._obj.get_index(dim)
        return idx[0], idx[-1]

    def describe_index(self):
        return describe_index(self._obj)

    def describe_values(self, *, style=False, **kwargs):
        return self._obj.to_series().ms.describe_values(style=style, **kwargs)

    def abs(self):
        return np.abs(self._obj)

    def between(self, left, right, inclusive: Literal["both", "neither", "left", "right"] = "both") -> xr.DataArray:
        return between(self._obj, left, right, inclusive)

    def intersection(self, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
        """
        Element-wise logical AND with another DataArray on the union of coordinates.

        Parameters
        ----------
        other : xr.DataArray
            Boolean DataArray to intersect with.
        union_dim : str, default "stock"
            Name of the dimension along which the union of coordinates should be taken.

        Returns
        -------
        xr.DataArray
            Array containing the intersection (logical AND) on the union of `union_dim`.
        """
        return binary_logical_op_on_union(self._obj, other, union_dim, op="and")

    def union(self, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
        """
        Element-wise logical OR with another DataArray on the union of coordinates.

        Parameters
        ----------
        other : xr.DataArray
            Boolean DataArray to union with.
        union_dim : str, default "stock"
            Name of the dimension along which the union of coordinates should be taken.

        Returns
        -------
        xr.DataArray
            Array containing the union (logical OR) on the union of `union_dim`.
        """
        return binary_logical_op_on_union(self._obj, other, union_dim, op="or")

    def difference(self, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
        """
        Element-wise difference with another DataArray on the union of coordinates.

        Computes elements that are True in this array but False in the other.

        Parameters
        ----------
        other : xr.DataArray
            Boolean DataArray to compute difference with.
        union_dim : str, default "stock"
            Name of the dimension along which the union of coordinates should be taken.

        Returns
        -------
        xr.DataArray
            Array containing the difference (self AND NOT other) on the union of `union_dim`.
        """
        return binary_logical_op_on_union(self._obj, other, union_dim, op="diff")

    def symmetric_difference(self, other: xr.DataArray, union_dim: str = "stock") -> xr.DataArray:
        """
        Element-wise symmetric difference with another DataArray on the union of coordinates.

        Computes elements that are True in exactly one of the two arrays (exclusive OR).

        Parameters
        ----------
        other : xr.DataArray
            Boolean DataArray to compute symmetric difference with.
        union_dim : str, default "stock"
            Name of the dimension along which the union of coordinates should be taken.

        Returns
        -------
        xr.DataArray
            Array containing the symmetric difference (XOR) on the union of `union_dim`.
        """
        return binary_logical_op_on_union(self._obj, other, union_dim, op="symmetric_diff")

    @property
    def hvplot(self):
        return self._obj.ms.to_pandas().hvplot


@xr.register_dataset_accessor("ms")
class MshevelevAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._dataframe = None

    def _get_dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = self._obj.to_dataframe()
        return self._dataframe

    def to_pandas(self, i2d=True):
        x = self._obj
        #    if i2d:
        #      x = x.ms.ix2date()
        res = x.to_pandas()
        if i2d:
            res = res.ms.ix2dt()
        if isinstance(res, pd.Series):
            res = res.rename(x.name)
        return res

    def to_pd(self, i2d=True):
        """Alias for to_pandas method."""
        return self.to_pandas(i2d=i2d)

    def ix2date(self, dims: tuple[str] = ("date",)):
        return convert_ints_to_dates(self._obj, dims)

    def ix2int(self, dims: tuple[str] = ("date",)):
        return convert_dates_to_ints(self._obj, dims)

    def describe_index(self):
        return describe_index(self._obj)

    def describe_values(self, *, style=False, **kwargs):
        return self._get_dataframe().ms.describe_values(style=style, **kwargs)

    def abs(self):
        return np.abs(self._obj)

    def between(self, left, right, inclusive: Literal["both", "neither", "left", "right"] = "both") -> xr.Dataset:
        return between(self._obj, left, right, inclusive)

    @property
    def hvplot(self):
        return self._obj.ms.to_pandas(i2d=True).hvplot
