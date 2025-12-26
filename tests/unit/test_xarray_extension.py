"""Comprehensive unit tests for xarray extension methods.

Tests all 10 refactored xarray extension methods:
- ix2date, ix2int
- describe_index, between, get_index_range
- binary_logical_op_on_union
- intersection, union, difference, symmetric_difference
"""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from ms_utils.xarray.extension_methods import *


class TestIx2Date:
    """Tests for ix2date (convert integers to dates)."""

    def test_convert_integer_dates(self):
        """Test converting integer dates to datetime."""
        da = xr.DataArray([1, 2, 3], coords={"date": [20230101, 20230102, 20230103]}, dims=["date"])

        result = da.ms.ix2date()

        assert isinstance(result.coords["date"].values[0], (pd.Timestamp, np.datetime64))
        assert str(result.coords["date"].values[0])[:10] == "2023-01-01"

    def test_multiple_dimensions(self):
        """Test with multiple dimensions."""
        da = xr.DataArray(
            np.random.rand(3, 2),
            coords={"date": [20230101, 20230102, 20230103], "stock": ["A", "B"]},
            dims=["date", "stock"],
        )

        result = da.ms.ix2date(dims=("date",))

        assert pd.api.types.is_datetime64_any_dtype(result.coords["date"])
        assert not pd.api.types.is_datetime64_any_dtype(result.coords["stock"])

    def test_skip_non_integer_dims(self):
        """Test that non-integer dims are skipped."""
        dates = pd.date_range("2023-01-01", periods=3)
        da = xr.DataArray([1, 2, 3], coords={"date": dates}, dims=["date"])

        result = da.ms.ix2date(dims=("date",))

        # Should already be datetime, no change
        xr.testing.assert_equal(result, da)


class TestIx2Int:
    """Tests for ix2int (convert dates to integers)."""

    def test_convert_datetime_to_int(self):
        """Test converting datetime to integer YYYYMMDD format."""
        dates = pd.date_range("2023-01-01", periods=3)
        da = xr.DataArray([1, 2, 3], coords={"date": dates}, dims=["date"])

        result = da.ms.ix2int(dims=("date",))

        assert pd.api.types.is_integer_dtype(result.coords["date"])
        assert result.coords["date"].values[0] == 20230101
        assert result.coords["date"].values[1] == 20230102

    def test_roundtrip_conversion(self):
        """Test ix2date and ix2int are inverses."""
        da = xr.DataArray([1, 2, 3], coords={"date": [20230101, 20230102, 20230103]}, dims=["date"])

        # Convert to datetime and back
        result = da.ms.ix2date().ms.ix2int()

        xr.testing.assert_equal(result, da)


class TestDescribeIndex:
    """Tests for describe_index."""

    def test_single_dimension(self):
        """Test describe_index with single dimension."""
        da = xr.DataArray([1, 2, 3], coords={"x": [10, 20, 30]}, dims=["x"])

        result = da.ms.describe_index()

        assert len(result) == 1
        assert result[0]["dim"] == "x"
        assert result[0]["len"] == 3

    def test_multiple_dimensions(self):
        """Test with multiple dimensions."""
        da = xr.DataArray(
            np.random.rand(10, 5),
            coords={"date": pd.date_range("2023-01-01", periods=10), "stock": list("ABCDE")},
            dims=["date", "stock"],
        )

        result = da.ms.describe_index()

        assert len(result) == 2
        dims = {r["dim"] for r in result}
        assert dims == {"date", "stock"}

        # Check date dimension
        date_info = [r for r in result if r["dim"] == "date"][0]
        assert date_info["len"] == 10

        # Check stock dimension
        stock_info = [r for r in result if r["dim"] == "stock"][0]
        assert stock_info["len"] == 5


class TestBetween:
    """Tests for between."""

    def test_between_both_inclusive(self):
        """Test between with both boundaries inclusive."""
        da = xr.DataArray([1, 2, 3, 4, 5])

        result = da.ms.between(2, 4, inclusive="both")

        expected = xr.DataArray([False, True, True, True, False])
        xr.testing.assert_equal(result, expected)

    def test_between_neither_inclusive(self):
        """Test between with neither boundary inclusive."""
        da = xr.DataArray([1, 2, 3, 4, 5])

        result = da.ms.between(2, 4, inclusive="neither")

        expected = xr.DataArray([False, False, True, False, False])
        xr.testing.assert_equal(result, expected)

    def test_between_left_inclusive(self):
        """Test between with left boundary inclusive."""
        da = xr.DataArray([1, 2, 3, 4, 5])

        result = da.ms.between(2, 4, inclusive="left")

        expected = xr.DataArray([False, True, True, False, False])
        xr.testing.assert_equal(result, expected)

    def test_between_right_inclusive(self):
        """Test between with right boundary inclusive."""
        da = xr.DataArray([1, 2, 3, 4, 5])

        result = da.ms.between(2, 4, inclusive="right")

        expected = xr.DataArray([False, False, True, True, False])
        xr.testing.assert_equal(result, expected)

    def test_between_with_floats(self):
        """Test between with float values."""
        da = xr.DataArray([1.5, 2.0, 2.5, 3.0, 3.5])

        result = da.ms.between(2.0, 3.0)

        expected = xr.DataArray([False, True, True, True, False])
        xr.testing.assert_equal(result, expected)


class TestGetIndexRange:
    """Tests for get_index_range."""

    def test_get_date_range(self):
        """Test getting date range."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        da = xr.DataArray(range(len(dates)), coords={"date": dates}, dims=["date"])

        first, last = da.ms.get_index_range("date")

        assert first == pd.Timestamp("2023-01-01")
        assert last == pd.Timestamp("2023-12-31")

    def test_get_numeric_range(self):
        """Test with numeric index."""
        da = xr.DataArray([1, 2, 3], coords={"x": [10, 20, 30]}, dims=["x"])

        first, last = da.ms.get_index_range("x")

        assert first == 10
        assert last == 30


class TestBinaryLogicalOpOnUnion:
    """Tests for binary_logical_op_on_union."""

    def test_and_operation(self):
        """Test logical AND operation."""
        da1 = xr.DataArray([True, False, True], coords={"stock": ["A", "B", "C"]}, dims=["stock"])
        da2 = xr.DataArray([True, True], coords={"stock": ["B", "C"]}, dims=["stock"])

        result = binary_logical_op_on_union(da1, da2, union_dim="stock", op="and")

        # Union includes A, B, C
        # A: True AND False (missing) = False
        # B: False AND True = False
        # C: True AND True = True
        assert result.sel(stock="A").values == False
        assert result.sel(stock="B").values == False
        assert result.sel(stock="C").values == True

    def test_or_operation(self):
        """Test logical OR operation."""
        da1 = xr.DataArray([True, False], coords={"stock": ["A", "B"]}, dims=["stock"])
        da2 = xr.DataArray([False, True], coords={"stock": ["B", "C"]}, dims=["stock"])

        result = binary_logical_op_on_union(da1, da2, union_dim="stock", op="or")

        # A: True OR False = True
        # B: False OR False = False
        # C: False OR True = True
        assert result.sel(stock="A").values == True
        assert result.sel(stock="B").values == False
        assert result.sel(stock="C").values == True

    def test_diff_operation(self):
        """Test difference operation."""
        da1 = xr.DataArray([True, True], coords={"stock": ["A", "B"]}, dims=["stock"])
        da2 = xr.DataArray([True, False], coords={"stock": ["B", "C"]}, dims=["stock"])

        result = binary_logical_op_on_union(da1, da2, union_dim="stock", op="diff")

        # A: True AND NOT False = True
        # B: True AND NOT True = False
        # C: False AND NOT False = False
        assert result.sel(stock="A").values == True
        assert result.sel(stock="B").values == False
        assert result.sel(stock="C").values == False

    def test_symmetric_diff_operation(self):
        """Test symmetric difference (XOR)."""
        da1 = xr.DataArray([True, False], coords={"stock": ["A", "B"]}, dims=["stock"])
        da2 = xr.DataArray([True, True], coords={"stock": ["B", "C"]}, dims=["stock"])

        result = binary_logical_op_on_union(da1, da2, union_dim="stock", op="symmetric_diff")

        # A: True XOR False = True
        # B: False XOR True = True
        # C: False XOR True = True
        assert result.sel(stock="A").values == True
        assert result.sel(stock="B").values == True
        assert result.sel(stock="C").values == True

    def test_error_on_non_boolean(self):
        """Test error when inputs are not boolean."""
        da1 = xr.DataArray([1, 2, 3], coords={"stock": ["A", "B", "C"]}, dims=["stock"])
        da2 = xr.DataArray([4, 5], coords={"stock": ["B", "C"]}, dims=["stock"])

        with pytest.raises(AssertionError, match="bool"):
            binary_logical_op_on_union(da1, da2, union_dim="stock", op="and")

    def test_error_on_missing_dim(self):
        """Test error when union_dim not in one array."""
        da1 = xr.DataArray([True, False], coords={"x": [1, 2]}, dims=["x"])
        da2 = xr.DataArray([True], coords={"stock": ["A"]}, dims=["stock"])

        with pytest.raises(ValueError, match="must contain"):
            binary_logical_op_on_union(da1, da2, union_dim="stock", op="and")


class TestConvenienceMethods:
    """Tests for intersection, union, difference, symmetric_difference."""

    def test_intersection(self):
        """Test intersection convenience method."""
        da1 = xr.DataArray([True, True], coords={"stock": ["A", "B"]}, dims=["stock"])
        da2 = xr.DataArray([True, False], coords={"stock": ["B", "C"]}, dims=["stock"])

        result = da1.ms.intersection(da2)

        # Should be same as binary_logical_op_on_union with op='and'
        expected = binary_logical_op_on_union(da1, da2, union_dim="stock", op="and")
        xr.testing.assert_equal(result, expected)

    def test_union(self):
        """Test union convenience method."""
        da1 = xr.DataArray([True, False], coords={"stock": ["A", "B"]}, dims=["stock"])
        da2 = xr.DataArray([False, True], coords={"stock": ["B", "C"]}, dims=["stock"])

        result = da1.ms.union(da2)

        expected = binary_logical_op_on_union(da1, da2, union_dim="stock", op="or")
        xr.testing.assert_equal(result, expected)

    def test_difference(self):
        """Test difference convenience method."""
        da1 = xr.DataArray([True, True], coords={"stock": ["A", "B"]}, dims=["stock"])
        da2 = xr.DataArray([True], coords={"stock": ["B"]}, dims=["stock"])

        result = da1.ms.difference(da2)

        expected = binary_logical_op_on_union(da1, da2, union_dim="stock", op="diff")
        xr.testing.assert_equal(result, expected)

    def test_symmetric_difference(self):
        """Test symmetric_difference convenience method."""
        da1 = xr.DataArray([True], coords={"stock": ["A"]}, dims=["stock"])
        da2 = xr.DataArray([True], coords={"stock": ["B"]}, dims=["stock"])

        result = da1.ms.symmetric_difference(da2)

        expected = binary_logical_op_on_union(da1, da2, union_dim="stock", op="symmetric_diff")
        xr.testing.assert_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
