"""Unit tests for plot_corr_matrix function."""

import pytest
import numpy as np
import pandas as pd
import holoviews as hv

from ms_utils.holoviews.extension_methods import plot_corr_matrix


class TestPlotCorrMatrix:
    """Tests for plot_corr_matrix function."""

    @pytest.fixture
    def sample_corr_matrix(self):
        """Create a sample correlation matrix for testing."""
        np.random.seed(42)
        data = np.random.randn(50, 4)
        df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
        return df.corr()

    def test_basic_plot(self, sample_corr_matrix):
        """Test basic correlation matrix plot."""
        result = plot_corr_matrix(sample_corr_matrix)

        # Should return an Overlay
        assert isinstance(result, hv.Overlay)

        # Should have 2 elements (HeatMap + Labels)
        assert len(result) == 2

        # First element should be HeatMap
        assert isinstance(result.get(0), hv.HeatMap)

        # Second element should be Labels
        assert isinstance(result.get(1), hv.Labels)

    def test_lower_triangle_only(self, sample_corr_matrix):
        """Test plotting only lower triangle."""
        result = plot_corr_matrix(sample_corr_matrix, only_lower_triangle=True)

        assert isinstance(result, hv.Overlay)

        # Verify it's still a valid plot
        heatmap = result.get(0)
        assert isinstance(heatmap, hv.HeatMap)

    def test_hide_diagonal(self, sample_corr_matrix):
        """Test hiding diagonal values."""
        result = plot_corr_matrix(sample_corr_matrix, show_diagonal=False)

        assert isinstance(result, hv.Overlay)
        assert isinstance(result.get(0), hv.HeatMap)

    def test_lower_triangle_without_diagonal(self, sample_corr_matrix):
        """Test lower triangle without diagonal."""
        result = plot_corr_matrix(sample_corr_matrix, only_lower_triangle=True, show_diagonal=False)

        assert isinstance(result, hv.Overlay)
        assert isinstance(result.get(0), hv.HeatMap)

    def test_with_small_matrix(self):
        """Test with a small 2x2 correlation matrix."""
        small_corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["X", "Y"], columns=["X", "Y"])

        result = plot_corr_matrix(small_corr)

        assert isinstance(result, hv.Overlay)
        assert len(result) == 2

    def test_with_perfect_correlations(self):
        """Test with perfect positive and negative correlations."""
        perfect_corr = pd.DataFrame(
            [[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], index=["A", "B", "C"], columns=["A", "B", "C"]
        )

        result = plot_corr_matrix(perfect_corr)

        assert isinstance(result, hv.Overlay)

    def test_with_nan_values(self):
        """Test handling of NaN values in correlation matrix."""
        corr_with_nan = pd.DataFrame(
            [[1.0, 0.5, np.nan], [0.5, 1.0, 0.3], [np.nan, 0.3, 1.0]], index=["A", "B", "C"], columns=["A", "B", "C"]
        )

        result = plot_corr_matrix(corr_with_nan)

        # Should handle NaN values gracefully
        assert isinstance(result, hv.Overlay)

    def test_return_type_annotation(self, sample_corr_matrix):
        """Test that return type matches annotation."""
        result = plot_corr_matrix(sample_corr_matrix)

        # Verify return type is hv.Overlay as annotated
        assert isinstance(result, hv.Overlay)

    def test_data_integrity(self, sample_corr_matrix):
        """Test that original dataframe is not modified."""
        original_values = sample_corr_matrix.copy()

        plot_corr_matrix(sample_corr_matrix, only_lower_triangle=True)

        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_corr_matrix, original_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
