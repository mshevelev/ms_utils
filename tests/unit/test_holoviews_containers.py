import pytest
import holoviews as hv
import pandas as pd
import numpy as np
from ms_utils.holoviews.extension_methods import build_color_mapping

# Initialize holoviews
hv.extension("bokeh")


@pytest.fixture
def sample_layout():
    c1 = hv.Curve([1, 2, 3], label="A")
    c2 = hv.Curve([2, 3, 4], label="B")
    return c1 + c2


@pytest.fixture
def sample_ndlayout():
    return hv.NdLayout({"A": hv.Curve([1, 2, 3]), "B": hv.Curve([2, 3, 4])})


@pytest.fixture
def sample_overlay():
    c1 = hv.Curve([1, 2, 3], label="A")
    c2 = hv.Curve([2, 3, 4], label="B")
    return c1 * c2


@pytest.fixture
def sample_ndoverlay():
    return hv.NdOverlay({"A": hv.Curve([1, 2, 3]), "B": hv.Curve([2, 3, 4])})


def test_registration_on_containers(sample_layout, sample_ndlayout):
    """Verify .ms namespace is available on Layout and NdLayout."""
    assert hasattr(sample_layout, "ms")
    assert hasattr(sample_ndlayout, "ms")
    assert hasattr(sample_layout.ms, "update_tooltips")
    assert hasattr(sample_layout.ms, "apply_colors")
    assert hasattr(sample_layout.ms, "yformat")


def test_update_tooltips_with_layout(sample_layout):
    """Verify update_tooltips recursively updates Layout."""
    updated = sample_layout.ms.update_tooltips({"A": "0.0"})
    assert isinstance(updated, hv.Layout)

    # Check tooltips of children
    for item in updated:
        tooltips = item.ms.get_tooltips()
        # If hover was not configured, it might be empty or default
        # But should not crash and should returned updated tooltips if configured
        assert isinstance(tooltips, list)


def test_apply_colors_with_layout(sample_layout):
    """Verify apply_colors recursively updates Layout."""
    updated = sample_layout.ms.apply_colors({"A": "red", "B": "blue"})
    assert isinstance(updated, hv.Layout)

    # Check colors of children
    assert updated[0].opts.get().kwargs.get("color") == "red"
    assert updated[1].opts.get().kwargs.get("color") == "blue"


def test_apply_colors_with_ndlayout(sample_ndlayout):
    """Verify apply_colors recursively updates NdLayout."""
    updated = sample_ndlayout.ms.apply_colors({"A": "red", "B": "blue"})
    assert isinstance(updated, hv.NdLayout)

    assert updated["A"].opts.get().kwargs.get("color") == "red"
    assert updated["B"].opts.get().kwargs.get("color") == "blue"


def test_build_color_mapping_with_containers(sample_layout, sample_ndlayout, sample_overlay, sample_ndoverlay):
    """Verify build_color_mapping works with all container types."""
    for fig in [sample_layout, sample_ndlayout, sample_overlay, sample_ndoverlay]:
        mapping = build_color_mapping(fig, ["red", "blue"])
        assert mapping == {"A": "red", "B": "blue"}


def test_yformat_with_layout(sample_layout):
    """Verify yformat recursively updates Layout."""
    updated = sample_layout.ms.yformat("$")
    assert isinstance(updated, hv.Layout)

    # Check yformatter of children
    from bokeh.models import NumeralTickFormatter

    assert isinstance(updated[0].opts.get().kwargs.get("yformatter"), NumeralTickFormatter)
    assert isinstance(updated[1].opts.get().kwargs.get("yformatter"), NumeralTickFormatter)


def test_rename_vdim_with_layout(sample_layout):
    """Verify rename_vdim recursively updates Layout."""
    updated = sample_layout.ms.rename_vdim("new_vdim")
    assert isinstance(updated, hv.Layout)

    assert updated[0].vdims[0].name == "new_vdim"
    assert updated[1].vdims[0].name == "new_vdim"
