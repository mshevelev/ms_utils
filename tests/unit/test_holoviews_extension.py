"""Unit tests for holoviews extension methods."""

import pytest
import numpy as np
import pandas as pd
import holoviews as hv

# Import the extension methods
from ms_utils.holoviews.extension_methods import (
    info,
    yformat,
    rename_vdim,
    overlay_labels,
    create_avg_line,
    get_tooltips,
    update_tooltips,
    apply_colors,
    build_color_mapping,
)


class TestInfo:
    """Tests for info method."""

    def test_info_returns_element(self, capsys):
        """Test that info prints and returns element."""
        curve = hv.Curve([(1, 2), (2, 3), (3, 4)])
        result = curve.ms.info()

        # Check it returns the element
        assert isinstance(result, hv.Curve)

        # Check it printed something
        captured = capsys.readouterr()
        assert "Curve" in captured.out


class TestYFormat:
    """Tests for yformat method."""

    def test_dollar_format(self):
        """Test dollar formatting."""
        curve = hv.Curve([(1, 1000), (2, 2000)])
        result = curve.ms.yformat("$")

        # Check it returns a Curve
        assert isinstance(result, hv.Curve)

        # The formatter is applied via opts, check it's still a curve
        assert result.data.equals(curve.data)

    def test_percentage_format(self):
        """Test percentage formatting."""
        curve = hv.Curve([(1, 0.5), (2, 0.75)])
        result = curve.ms.yformat("%")

        assert isinstance(result, hv.Curve)

    def test_integer_format(self):
        """Test integer formatting."""
        curve = hv.Curve([(1, 100), (2, 200)])
        result = curve.ms.yformat("int")

        assert isinstance(result, hv.Curve)

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        curve = hv.Curve([(1, 2)])

        with pytest.raises(ValueError, match="not supported"):
            curve.ms.yformat("invalid")


class TestRenameVdim:
    """Tests for rename_vdim method."""

    def test_rename_single_vdim(self):
        """Test renaming a single vdim."""
        curve = hv.Curve([(1, 2), (2, 3)], vdims="value")
        result = curve.ms.rename_vdim("new_value")

        assert result.vdims[0].name == "new_value"

    def test_rename_with_no_vdims(self):
        """Test renaming when no vdims exist."""
        curve = hv.Curve([(1, 2), (2, 3)])
        # Should not raise error
        result = curve.ms.rename_vdim("new_name")
        assert isinstance(result, hv.Curve)


class TestOverlayLabels:
    """Tests for overlay_labels method."""

    def test_overlay_labels_basic(self):
        """Test basic label overlay."""
        bars = hv.Bars([("A", 10), ("B", 20)], kdims="Category", vdims="Value")
        result = bars.ms.overlay_labels()

        # Should return an Overlay
        assert isinstance(result, hv.Overlay)

        # Should have 2 elements (bars + labels)
        assert len(result) == 2


class TestCreateAvgLine:
    """Tests for create_avg_line method."""

    def test_create_avg_line_no_annotation(self):
        """Test creating average line without annotation."""
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 15, 25, 30]})
        curve = hv.Curve(data, kdims="x", vdims="y")

        result = curve.ms.create_avg_line()

        # Should return a Curve
        assert isinstance(result, hv.Curve)

    def test_create_avg_line_with_center_annotation(self):
        """Test creating average line with center annotation."""
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 15, 25, 30]})
        curve = hv.Curve(data, kdims="x", vdims="y")

        result = curve.ms.create_avg_line(annotation_pos="center")

        # Should return an Overlay (line + text)
        assert isinstance(result, hv.Overlay)

    def test_create_avg_line_positions(self):
        """Test all annotation positions."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 15]})
        curve = hv.Curve(data, kdims="x", vdims="y")

        for pos in ["left", "center", "right"]:
            result = curve.ms.create_avg_line(annotation_pos=pos)
            assert isinstance(result, hv.Overlay)

    def test_create_avg_line_custom_agg(self):
        """Test with custom aggregation function."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        curve = hv.Curve(data, kdims="x", vdims="y")

        result = curve.ms.create_avg_line(agg_func=np.median)

        assert isinstance(result, hv.Curve)


class TestGetTooltips:
    """Tests for get_tooltips method."""

    def test_get_tooltips_basic(self):
        """Test getting tooltips from a basic element."""
        curve = hv.Curve([(1, 2), (2, 3)])
        tooltips = curve.ms.get_tooltips()

        # Should return a list
        assert isinstance(tooltips, list)

        # May be empty if no hover tool configured
        assert isinstance(tooltips, list)


class TestUpdateTooltips:
    """Tests for update_tooltips method."""

    def test_update_tooltips_basic(self):
        """Test updating tooltips."""
        curve = hv.Curve([(1, 2), (2, 3)])
        result = curve.ms.update_tooltips({"value": "0.00"})

        # Should return an element
        assert isinstance(result, hv.Curve)


class TestApplyColors:
    """Tests for apply_colors method."""

    def test_apply_colors_to_curve(self):
        """Test applying colors to a single curve."""
        curve = hv.Curve([(1, 2), (2, 3)], label="A")
        result = curve.ms.apply_colors({"A": "red"})

        assert isinstance(result, hv.Curve)

    def test_apply_colors_to_overlay(self):
        """Test applying colors to an overlay."""
        curve1 = hv.Curve([(1, 2)], label="A")
        curve2 = hv.Curve([(1, 3)], label="B")
        overlay = curve1 * curve2

        result = overlay.ms.apply_colors({"A": "red", "B": "blue"})

        assert isinstance(result, hv.Overlay)


class TestBuildColorMapping:
    """Tests for build_color_mapping function."""

    def test_with_color_list(self):
        """Test building color mapping from list of colors."""
        curve1 = hv.Curve([1, 2, 3], label="A")
        curve2 = hv.Curve([2, 3, 4], label="B")
        overlay = curve1 * curve2

        mapping = build_color_mapping(overlay, ["red", "blue"])

        assert isinstance(mapping, dict)
        assert "A" in mapping
        assert "B" in mapping
        assert mapping["A"] == "red"
        assert mapping["B"] == "blue"

    def test_with_palette_name(self):
        """Test building color mapping from palette name."""
        curve1 = hv.Curve([1, 2, 3], label="A")
        curve2 = hv.Curve([2, 3, 4], label="B")
        overlay = curve1 * curve2

        mapping = build_color_mapping(overlay, "Category10")

        assert isinstance(mapping, dict)
        assert "A" in mapping
        assert "B" in mapping
        # Category10 colors should be hex strings
        assert mapping["A"].startswith("#")
        assert mapping["B"].startswith("#")

    def test_color_cycling(self):
        """Test that colors cycle when more labels than colors."""
        curves = [hv.Curve([i, i + 1, i + 2], label=f"Series{i}") for i in range(5)]
        overlay = curves[0]
        for curve in curves[1:]:
            overlay = overlay * curve

        mapping = build_color_mapping(overlay, ["red", "blue"])

        # Should have 5 labels
        assert len(mapping) == 5
        # Colors should cycle
        values = list(mapping.values())
        assert values[0] == "red"
        assert values[1] == "blue"
        assert values[2] == "red"
        assert values[3] == "blue"
        assert values[4] == "red"

    def test_with_ndoverlay(self):
        """Test with NdOverlay structure."""
        ndoverlay = hv.NdOverlay({"A": hv.Curve([1, 2, 3]), "B": hv.Curve([2, 3, 4]), "C": hv.Curve([3, 4, 5])})

        mapping = build_color_mapping(ndoverlay, ["red", "green", "blue"])

        assert "A" in mapping
        assert "B" in mapping
        assert "C" in mapping


class TestApplyColorsEnhanced:
    """Tests for enhanced apply_colors with palette support."""

    def test_apply_colors_with_list(self):
        """Test applying colors using a list."""
        curve1 = hv.Curve([1, 2, 3], label="A")
        curve2 = hv.Curve([2, 3, 4], label="B")
        overlay = curve1 * curve2

        result = overlay.ms.apply_colors(["red", "blue"])

        assert isinstance(result, hv.Overlay)

    def test_apply_colors_with_palette(self):
        """Test applying colors using palette name."""
        curve1 = hv.Curve([1, 2, 3], label="A")
        curve2 = hv.Curve([2, 3, 4], label="B")
        overlay = curve1 * curve2

        result = overlay.ms.apply_colors("Set1")

        assert isinstance(result, hv.Overlay)

    def test_apply_colors_with_dict_still_works(self):
        """Test that original dict behavior still works."""
        curve1 = hv.Curve([1, 2, 3], label="A")
        curve2 = hv.Curve([2, 3, 4], label="B")
        overlay = curve1 * curve2

        result = overlay.ms.apply_colors({"A": "red", "B": "blue"})

        assert isinstance(result, hv.Overlay)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
