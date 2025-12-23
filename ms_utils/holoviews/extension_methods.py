"""HoloViews extension methods for ms_utils.

Provides utility methods for HoloViews elements via the `.ms` namespace.
"""

from typing import Literal, overload, Union
import numpy as np
import pandas as pd
import bokeh.models
import scipy.stats
import holoviews as hv
import hvplot.pandas

from ms_utils.method_registration import register_method
from ms_utils.pandas.extension_methods import ecdf_transform


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def info(el: hv.core.dimension.ViewableElement) -> hv.core.dimension.ViewableElement:
    """Print information about hv.Element."""
    print(el)
    return el


def get_format_string(format_alias: str) -> tuple[str, str]:
    """Resolve format alias to format string and type.
    
    Parameters
    ----------
    format_alias : str
        Format alias like '$', 'usd', '%', 'int', 'date', etc.
    
    Returns
    -------
    tuple[str, str]
        Tuple of (format_string, format_type) where format_type is 'numeral', 'datetime', or 'printf'
    """
    # Currency formats
    if format_alias in ("$", "usd"):
        return ("$0,0.", "numeral")
    # Percentage formats
    elif format_alias in ("%", "%2"):
        return ("0.00%", "numeral")
    # Integer/number formats
    elif format_alias in ("#", "int"):
        return ("0,0.", "numeral")
    # Date formats
    elif format_alias in ("date", "yyyy-mm-dd"):
        return ("%Y-%m-%d", "datetime")
    else:
        raise ValueError(f"not supported {format_alias=!r}")


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def format_axis(el: hv.core.dimension.ViewableElement, format: str, axis: Literal["x", "y"]):
    """Apply one of the predefined formatters to x or y axis.
    
    Parameters
    ----------
    el : hv.core.dimension.ViewableElement
        The HoloViews element to format
    format : str
        Format alias like '$', 'usd', '%', 'int', 'date', etc.
    axis : Literal["x", "y"]
        Which axis to apply the formatter to
    
    Returns
    -------
    hv.core.dimension.ViewableElement
        Element with formatter applied
    """
    format_string, format_type = get_format_string(format)
    
    if format_type == "numeral":
        formatter = bokeh.models.NumeralTickFormatter(format=format_string)
    elif format_type == "datetime":
        formatter = bokeh.models.DatetimeTickFormatter()
        # Set the datetime format for all time scales
        formatter.days = [format_string]
        formatter.months = [format_string]
        formatter.years = [format_string]
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")
    
    if axis == "y":
        return el.opts(yformatter=formatter)
    elif axis == "x":
        return el.opts(xformatter=formatter)
    else:
        raise ValueError(f"Invalid axis: {axis}")


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def yformat(el: hv.core.dimension.ViewableElement, format: str):
    """Apply one of the predefined formatters to y-axis.
    
    Parameters
    ----------
    el : hv.core.dimension.ViewableElement
        The HoloViews element to format
    format : str
        Format alias like '$', 'usd', '%', 'int', 'date', etc.
    
    Returns
    -------
    hv.core.dimension.ViewableElement
        Element with y-axis formatter applied
    """
    return format_axis(el, format, axis="y")


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def xformat(el: hv.core.dimension.ViewableElement, format: str):
    """Apply one of the predefined formatters to x-axis.
    
    Parameters
    ----------
    el : hv.core.dimension.ViewableElement
        The HoloViews element to format
    format : str
        Format alias like '$', 'usd', '%', 'int', 'date', etc.
    
    Returns
    -------
    hv.core.dimension.ViewableElement
        Element with x-axis formatter applied
    """
    return format_axis(el, format, axis="x")


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def rename_vdim(fig: hv.core.dimension.ViewableElement, name: str):
    """Rename the vdim of the holoviews element.
    
    If the object is hv.Overlay or hv.NdOverlay, recursively apply renaming of vdim to children.
    Tip: useful to disable syncing on y-axis.
    """
    if isinstance(fig, (hv.NdOverlay, hv.Overlay)):
        new_items = {k: rename_vdim(v, name) for k, v in fig.items()}
        result = type(fig)(
            list(new_items.items()),
            **dict(fig.param.get_param_values())
        )
        # Preserve any existing options
        opts = fig.opts.get()
        if opts:
            result = result.opts(opts)
        return result
    
    src_vdim = fig.vdims[0].name if len(fig.vdims) else "0"
    return fig.redim(**{src_vdim: name})


@register_method(classes=[hv.element.chart.Bars], namespace="ms")
def overlay_labels(el: hv.element.chart.Bars, text_font_size="10px", **kwargs) -> hv.core.overlay.Overlay:
    """Overlay labels on top hv.Bars."""
    labels = hv.Labels(el.data, kdims=el.kdims + el.vdims[:1], vdims=el.vdims[:1]).opts(
        text_font_size=text_font_size, **kwargs
    )
    return el * labels


@register_method(classes=[hv.element.chart.Curve], namespace="ms")
def create_avg_line(curve, annotation_pos: Literal["center", "left", "right"] = None, agg_func=np.mean) -> hv.Curve:
    """Create a horizontal (dashed) line equal to the average value of the curve with optional annotation."""
    kdim = curve.kdims[0].name
    vdim = curve.vdims[0].name

    s = curve.data.set_index(kdim)[vdim].dropna().sort_index()
    value = agg_func(s)
    label = curve.label
    line = hv.Curve([(s.index[0], value), (s.index[-1], value)], label=label).opts(line_dash=[4, 8])
    if annotation_pos is None:
        return line

    # get position of annotation
    if annotation_pos == "center":
        text_pos = s.index[int(len(s) / 2)]
    elif annotation_pos == "left":
        text_pos = s.index[0]
    elif annotation_pos == "right":
        text_pos = s.index[-1]
    else:
        raise ValueError(f"{annotation_pos=!r} is not supported")

    text = hv.Text(text_pos, value, text=f"{value:.3g}", label=label, valign="top", fontsize=8)
    return line * text


@overload
def plot_ecdf(s: pd.Series, kind: str) -> Union[hv.Curve, hv.Scatter]: ...
@overload
def plot_ecdf(df: pd.DataFrame, kind: str) -> hv.NdOverlay: ...


def plot_ecdf(s_or_df: Union[pd.DataFrame, pd.Series], kind="scatter"):
    """Create ECDF plot from Series or DataFrame.
    
    Note: For new code, prefer using the ECDF element from ms_utils.holoviews.ecdf.
    """
    assert kind in ["scatter", "line"]
    if isinstance(s_or_df, pd.Series):
        v = s_or_df
        if kind == "line":
            v = v.sort_values()
        q = ecdf_transform(v)
        label = q.name or ""
        if kind == "line":
            el = hv.Curve((v, q), kdims="value", vdims="quantile", label=label)
        else:
            el = hv.Scatter((v, q), kdims="value", vdims="quantile", label=label)
        return el.opts(width=500)
    elif isinstance(s_or_df, pd.DataFrame):
        res = {col: plot_ecdf(s_or_df[col], kind=kind) for col in s_or_df.columns}
        return hv.NdOverlay(res, label='ECDF')
    else:
        raise TypeError(f"{type(s_or_df)=} is not supported")


def hvplot_qqplot(Xs):
    """Create Q-Q plot for data."""
    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(Xs, plot=None, fit=True)

    _df = pd.DataFrame({"Theoretical Quantiles": osm, "Sample Quantiles": osr})
    scatter = _df.hvplot.scatter(_df.columns[0], _df.columns[1])

    reg_line = hv.Slope(slope, intercept).opts(color='red')
    res = scatter * reg_line
    return res


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def get_tooltips(el: hv.core.dimension.ViewableElement) -> list[tuple[str, str]]:
    """Get the current tooltips from a HoloViews element.

    Returns
    -------
    list[tuple[str, str]]
        List of (label, value) tuples representing tooltips.
    """
    from bokeh.models import HoverTool

    try:
        bokeh_fig = hv.render(el, backend="bokeh")
    except Exception:
        return []

    hover_tool = None
    if hasattr(bokeh_fig, "tools"):
        for tool in bokeh_fig.tools:
            if isinstance(tool, HoverTool):
                hover_tool = tool
                break

    if hover_tool:
        return hover_tool.tooltips
    return []


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def update_tooltips(el: hv.core.dimension.ViewableElement, tooltips: dict[str, str]):
    """Update tooltip formatting for specific fields in a HoloViews element.

    Parameters
    ----------
    el : hv.core.dimension.ViewableElement
        The HoloViews element to update
    tooltips : dict[str, str]
        Mapping from field names to format strings or format aliases
        (e.g., '$', 'usd', '%', 'int', 'date', 'yyyy-mm-dd')

    Returns
    -------
    hv.core.dimension.ViewableElement
        Updated element with new tooltip configuration
    """
    import re
    from bokeh.models import HoverTool

    # Recursive handling for overlays
    if isinstance(el, (hv.Overlay, hv.NdOverlay)):
        new_items = []
        for k, v in el.items():
            new_items.append((k, update_tooltips(v, tooltips)))

        if isinstance(el, hv.NdOverlay):
            new_el = hv.NdOverlay(new_items, kdims=el.kdims, label=el.label, group=el.group)
        else:
            new_el = hv.Overlay([v for k, v in new_items], label=el.label, group=el.group)

        if hasattr(el, "opts"):
            opts = el.opts.get()
            if opts:
                new_el = new_el.opts(opts)

        return new_el

    # Render to get current tooltips
    try:
        bokeh_fig = hv.render(el, backend="bokeh")
    except Exception:
        return el

    current_tooltips = []
    current_formatters = {}
    if hasattr(bokeh_fig, "tools"):
        for tool in bokeh_fig.tools:
            if isinstance(tool, HoverTool):
                current_tooltips = tool.tooltips
                if hasattr(tool, "formatters") and tool.formatters:
                    current_formatters = dict(tool.formatters)
                break

    if not current_tooltips:
        current_tooltips = []

    new_tooltips = []
    processed_keys = set()
    formatters = dict(current_formatters)

    field_regex = re.compile(r"@(?:\{([^\}]+)\}|(\w+))")
    datetime_format_regex = re.compile(r"%[a-zA-Z]")

    def resolve_format(format_str: str) -> str:
        """Resolve format alias to actual format string."""
        # If it already contains @, it's a complete tooltip spec
        if "@" in format_str:
            return format_str
        
        # Try to resolve as format alias
        try:
            resolved_format, format_type = get_format_string(format_str)
            # Return the resolved format string
            return resolved_format
        except ValueError:
            # Not a recognized alias, treat as raw format string
            pass
        
        return format_str

    for label, value in current_tooltips:
        updated = False
        field_name = None

        match = field_regex.search(value)
        if match:
            field_name = match.group(1) or match.group(2)

        if label in tooltips:
            new_format = resolve_format(tooltips[label])
            processed_keys.add(label)
            updated = True
        elif field_name and field_name in tooltips:
            new_format = resolve_format(tooltips[field_name])
            processed_keys.add(field_name)
            updated = True

        if updated:
            if "@" in new_format:
                new_value = new_format
            else:
                if not field_name:
                    new_value = value
                else:
                    new_value = f"@{{{field_name}}}{{{new_format}}}"
                    if datetime_format_regex.search(new_format):
                        formatters[f"@{{{field_name}}}"] = "datetime"
            new_tooltips.append((label, new_value))
        else:
            new_tooltips.append((label, value))

    # Add new tooltips
    for key, format_str in tooltips.items():
        if key not in processed_keys:
            new_format = resolve_format(format_str)
            if "@" in new_format:
                new_value = new_format
            else:
                new_value = f"@{{{key}}}{{{new_format}}}"
                if datetime_format_regex.search(new_format):
                    formatters[f"@{{{key}}}"] = "datetime"
            new_tooltips.append((key, new_value))

    # Apply updated tooltips
    if formatters:
        hover = HoverTool(tooltips=new_tooltips, formatters=formatters)
    else:
        hover = HoverTool(tooltips=new_tooltips)
    return el.opts(tools=[hover])


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def apply_colors(fig: hv.core.dimension.ViewableElement, color_mapping: dict):
    """Apply colors to HoloViews elements based on dimension names.

    Parameters
    ----------
    fig : hv.core.dimension.ViewableElement
        HoloViews element (Curve, Scatter, Layout, Overlay, etc.)
    color_mapping : dict
        Mapping from dimension names to colors

    Returns
    -------
    hv.core.dimension.ViewableElement
        Element with colors applied
    """
    def _apply_color_to_element(element):
        if isinstance(element, (hv.Curve, hv.Scatter)):
            label = element.label
            if label in color_mapping:
                return element.opts(color=color_mapping[label])

            all_dims = element.kdims + element.vdims
            for dim in all_dims:
                dim_name = dim.name if hasattr(dim, "name") else str(dim)
                if dim_name in color_mapping:
                    return element.opts(color=color_mapping[dim_name])

        return element

    def _apply_colors_recursive(obj):
        if isinstance(obj, (hv.NdOverlay, hv.NdLayout)):
            new_items = {}
            for key, element in obj.items():
                key_str = str(key)
                if key_str in color_mapping:
                    color = color_mapping[key_str]
                    if isinstance(element, (hv.Curve, hv.Scatter)):
                        element = element.opts(color=color)
                    else:
                        element = _apply_colors_recursive(element)
                else:
                    element = _apply_colors_recursive(element)
                new_items[key] = element

            return type(obj)(new_items, **dict(obj.param.get_param_values()))

        elif isinstance(obj, (hv.Overlay, hv.Layout)):
            new_elements = []
            for element in obj:
                new_elements.append(_apply_colors_recursive(element))

            return type(obj)(new_elements, **dict(obj.param.get_param_values()))

        else:
            return _apply_color_to_element(obj)

    return _apply_colors_recursive(fig)





 
def plot_corr_matrix(
    df: pd.DataFrame, 
    only_lower_triangle: bool = False, 
    show_diagonal: bool = True
) -> hv.Overlay:
    """Display correlation matrix as HeatMap overlayed with formatted labels.
    
    Creates a visualization of a correlation matrix using HoloViews, with a heatmap
    showing correlation values and text labels displaying the numeric values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Correlation matrix (typically from df.corr()). Should be a square DataFrame
        with numeric values between -1 and 1.
    only_lower_triangle : bool, default False
        If True, only show the lower triangle of the correlation matrix.
        Useful for avoiding redundant information since correlation matrices are symmetric.
    show_diagonal : bool, default True
        If True, show the diagonal (self-correlations, typically 1.0).
        If False, hide diagonal values.
    
    Returns
    -------
    hv.Overlay
        HoloViews Overlay containing a HeatMap and Labels elements.
    
    Examples
    --------
    >>> corr_matrix = df.corr()
    >>> plot = plot_corr_matrix(corr_matrix)
    >>> plot = plot_corr_matrix(corr_matrix, only_lower_triangle=True, show_diagonal=False)
    """
    # Configure heatmap options
    heatmap_opts = hv.opts.HeatMap(
        width=500, 
        height=450, 
        cmap='coolwarm',
        fontsize={'xticks': 6, 'yticks': 6},
        xrotation=90, 
        colorbar=True, 
        invert_xaxis=True,
        xaxis='top', 
        xlabel="", 
        ylabel="",
        tools=['hover']
    )
    # Apply masking based on parameters (don't modify original df)
    masked_df = df.copy()
    
    if only_lower_triangle:
        # Keep only lower triangle (including or excluding diagonal based on show_diagonal)
        mask = np.tril(np.ones_like(masked_df, dtype=bool), k=0 if show_diagonal else -1)
        masked_df = masked_df.where(mask)
    elif not show_diagonal:
        # Hide only diagonal
        masked_df = masked_df.where(~np.eye(*masked_df.shape, dtype=bool))
 
    # Convert to long format for HoloViews (use future_stack=True to avoid FutureWarning)
    df_long = masked_df.stack(future_stack=True).rename("corr").reset_index()
 
    # Create heatmap and labels
    heatmap = hv.HeatMap(df_long).opts(heatmap_opts)
    labels = hv.Labels(
        df_long,
        vdims=hv.Dimension(
            'corr', 
            value_format=lambda x: f"{x:.02f}" if np.isfinite(x) else ""
        )
    ).opts(text_font_size='8pt')
    
    res = heatmap * labels
    return res
 
 
