from typing import Literal
import numpy as np
import pandas as pd
import bokeh.models
import holoviews as hv

from ms_utils.method_registration import register_method

@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def info(el: hv.core.dimension.ViewableElement) -> hv.core.dimension.ViewableElement:
  """Print information about hv.Element"""
  print(el)
  return el

@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def yformat(el: hv.core.dimension.ViewableElement, format: Literal["$", "usd", "%2", "%", "#", "int"]):
  """Apply one of the predefined yformatters"""
  if   format in ("$", "usd"):  yformatter=bokeh.models.NumeralTickFormatter(format='$0,0.')
  elif format in ("%", "%2"): yformatter=bokeh.models.NumeralTickFormatter(format='0.00%')
  elif format in ("#", "int"): yformatter=bokeh.models.NumeralTickFormatter(format='0,0.')
  else: raise ValueError(f"not supported {format=!r}")
  return el.opts(yformatter=yformatter)
 
@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def rename_vdim(fig: hv.core.dimension.ViewableElement, name: str):
  """Rename the vdim of the holoviews element.
  Tip: useful to disable syncing on y-axis
  """
  src_vdim = fig.vdims[0].name if len(fig.vdims) else "0"
  return fig.redim(**{src_vdim: name})
 
 
 
@register_method(classes=[hv.element.chart.Bars], namespace="ms")
def overlay_labels(el:hv.element.chart.Bars, text_font_size='10px', **kwargs)  -> hv.core.overlay.Overlay:
  """Overlay labels on top hv.Bars"""
  labels = hv.Labels(el.data,kdims=el.kdims+el.vdims[:1], vdims=el.vdims[:1]).opts(text_font_size=text_font_size, **kwargs)
  return el * labels
 
@register_method(classes=[hv.element.chart.Curve], namespace="ms")
def create_avg_line(curve, annotation_pos: Literal['center', 'left', 'right'] = None, agg_func=np.mean) -> hv.Curve:
  """Create a horizontal (dashed) line equal to the average value of the `curve` with optional annotation"""
  kdim = curve.kdims[0].name
  vdim = curve.vdims[0].name

  s = curve.data.set_index(kdim)[vdim].dropna().sort_index()
  value = agg_func(s)
  # _data = pd.DataFrame({"x": data.index, "y": value})
  label = curve.label
  line = hv.Curve([(s.index[0], value), (s.index[-1], value)], label=label).opts(line_dash=[4, 8])
  if annotation_pos is None:
    return line
 
  # get position of annotation
  if annotation_pos == 'center':
    text_pos = s.index[int(len(s) / 2)]
  elif annotation_pos == 'left':
    text_pos = s.index[0]
  elif annotation_pos == 'right':
    text_pos = s.index[-1]
  else:
    raise ValueError(f'{annotation_pos=!r} is not supported')

  text = hv.Text(text_pos, value, text=f"{value:.3g}", label=label, valign='top', fontsize=8)
  return line * text







# import numpy as np
#  import pandas as pd
#  import holoviews as hv
#  import bokeh.models
#  import scipy.stats
#  import hvplot.pandas
 
#  from typing import Literal, Union, overload
 
#  from .pandas import ecdf_transform
#  from .method_registration import register_method
#  from .string_formatters import get_formatter, python_format_to_c_format
 
#  @overload
#  def plot_ecdf(s: pd.Series, kind: str) -> Union[hv.Curve, hv.Scatter]: ...
#  @overload
#  def plot_ecdf(df: pd.DataFrame, kind: str) -> hv.NdOverlay: ...
 
 
#  def plot_ecdf(s_or_df: Union[pd.DataFrame, pd.Series], kind="scatter"):
#    assert kind in ["scatter", "line"]
#    if isinstance(s_or_df, pd.Series):
#      v = s_or_df
#      if kind == "line":
#        v = v.sort_values()
#      q = ecdf_transform(v)
#      label = q.name or ""
#      if kind == "line":
#        el = hv.Curve((v, q), kdims="value", vdims="quantile", label=label)
#      else:
#        el = hv.Scatter((v, q), kdims="value", vdims="quantile", label=label)
#      return el.opts(width=500)
#    elif isinstance(s_or_df, pd.DataFrame):
#      res = {col: plot_ecdf(s_or_df[col], kind=kind) for col in s_or_df.columns}
#      return hv.NdOverlay(res, label='ECDF')
#    else:
#      raise TypeError(f"{type(s_or_df)=} is not supported")
 
 
#  def hvplot_qqplot(Xs):
#    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(Xs, plot=None, fit=True)
 
#    _df = pd.DataFrame({"Theoretical Quantiles": osm, "Sample Quantiles": osr})
#    scatter = _df.hvplot.scatter(_df.columns[0], _df.columns[1])
 
#    reg_line = hv.Slope(slope, intercept).opts(color='red')
#    res = scatter * reg_line
#    return res
 
 
#  def plot_corr_matrix(df: pd.DataFrame, only_lower_triangle: bool = False, show_diagonal: bool = True):
#    """Display Corr Matrix as HeatMap overlayed with labels"""
#    hv_opts_heatmap = hv.opts.HeatMap(width=500, height=450, cmap='coolwarm', fontsize={'xticks': 6, 'yticks': 6},
#                                      xrotation=90, colorbar=True, invert_xaxis=True,
#                                      xaxis='top', xlabel="", ylabel="",
#                                      tools=['hover'])
#    if only_lower_triangle is True:
#      df = df.where(np.tril(np.ones_like(df, dtype=bool), k=0))
 
#    if show_diagonal is False:
#      df = df.where(~np.eye(*df.shape, dtype=bool))
 
#    _df0 = df.stack(dropna=False).rename("corr").reset_index()
 
#    res = (hv.HeatMap(_df0).opts(hv_opts_heatmap)
#           * hv.Labels(_df0,
#                       vdims=hv.Dimension('corr', value_format=lambda x: f"{x:.02f}" if np.isfinite(x) else "")
#                       ).opts(text_font_size='8pt')
#           )
#    return res
 
 
#  @register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
#  def yformat(
#      el: hv.core.dimension.ViewableElement,
#      format: Literal["$", "usd", "%2", "pct", "%", "#", "int", "num", "$2", "usd2"],
#      # format_hover: bool = False
#  ):
#    """Apply one of the predefined yformatters"""
#    # formatter_str = get_formatter(format)
#    # formatter_str = python_format_to_c_format(formatter_str)
#    # yformatter=bokeh.models.PrintfTickFormatter(format=formatter_str)
#    if format in ("$", "usd"):  yformatter = bokeh.models.NumeralTickFormatter(format='$0,0.')
#    elif format in ("$2", "usd2", "cents", "cent"): yformatter = bokeh.models.NumeralTickFormatter(format='$0,0.00')
#    elif format in ("pct", "%", "%2"): yformatter = bokeh.models.NumeralTickFormatter(format='0.00%')
#    elif format in ("#", "int", "num"): yformatter = bokeh.models.NumeralTickFormatter(format='0,0.')
#    else: raise ValueError(f"not supported {format=!r}")
 
 
#    el = el.opts(yformatter=yformatter)
#    # if format_hover is True:
#    #   assert isinstance(el.data, pd.DataFrame)
#    #   assert el.data.index.name == "date"
#    #   assert isinstance(el.data.index, pd.DatetimeIndex)
#    #   el = el.opts(
#    #     backend_opts={
#    #       "plot.hover.tooltips": [("date", "@date{%F}"), ("value", f"@value{{yformatter.format}}")],
#    #       "plot.hover.formatters": {"@date": "datetime"}
#    #     }
#    #   )
#    return el
 
#  @register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
#  def rename_vdim(fig: hv.core.dimension.ViewableElement, name: str):
#    """Rename the vdim of the holoviews element.
#    If the object is hv.Overlay or hv.NdOverlay recursively apply renaming of vdim to children.
#    Tip: useful to disable syncing on y-axis
#    """
#    if isinstance(fig, (hv.NdOverlay, hv.Overlay)):
#      fig = type(fig)(
#        list({k: rename_vdim(v, name) for k, v in fig.items()}.items()),
#        **dict(fig.param.get_param_values())
#      ).opts(fig.opts.get())
#      return fig
#    src_vdim = fig.vdims[0].name if len(fig.vdims) else "0"
#    return fig.redim(**{src_vdim: name})
 
 
#  @register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
#  def info(el: hv.core.dimension.ViewableElement):
#    """Print information about hv.Element"""
#    print(el)
#    return el
 
#  @register_method(classes=[hv.element.chart.Bars], namespace="ms")
#  def overlay_labels(el:hv.element.chart.Bars, text_font_size='10px', **kwargs)  -> hv.core.overlay.Overlay:
#    """Overlay labels on top hv.Bars"""
#    labels = hv.Labels(el.data,kdims=el.kdims+el.vdims[:1], vdims=el.vdims[:1]).opts(text_font_size=text_font_size, **kwargs)
#    return el * labels
 
#  @register_method(classes=[hv.element.chart.Curve], namespace="ms")
#  def create_avg_line(curve, annotation_pos: Literal['center', 'left', 'right'] = None, agg_func=np.mean) -> hv.Curve:
#    """Create a horizontal (dashed) line equal to the average value of the `curve` with optional annotation"""
#    kdim = curve.kdims[0].name
#    vdim = curve.vdims[0].name
 
#    s = curve.data.set_index(kdim)[vdim].dropna().sort_index()
#    value = agg_func(s)
#    # _data = pd.DataFrame({"x": data.index, "y": value})
#    label = curve.label
#    line = hv.Curve([(s.index[0], value), (s.index[-1], value)], label=label).opts(line_dash=[4, 8])
#    if annotation_pos is None:
#      return line
 
#    # get position of annotation
#    if annotation_pos == 'center':
#      text_pos = s.index[int(len(s) / 2)]
#    elif annotation_pos == 'left':
#      text_pos = s.index[0]
#    elif annotation_pos == 'right':
#      text_pos = s.index[-1]
#    else:
#      raise ValueError(f'{annotation_pos=!r} is not supported')
 
#    text = hv.Text(text_pos, value, text=f"{value:.3g}", label=label, valign='top', fontsize=8)
#    return line * text


@register_method(classes=[hv.core.dimension.ViewableElement], namespace='ms')
def get_tooltips(el: hv.core.dimension.ViewableElement) -> list[tuple[str, str]]:
    """
    Returns the list of tooltips for the given element.
    Returns a list of (label, value) tuples.
    """
    import holoviews as hv
    from bokeh.models import HoverTool
    # For Overlay/NdOverlay, we might want to look at the first child
    # or just render the whole thing. Rendering the whole thing is safer
    # to see what Bokeh actually produces.
    try:
        bokeh_fig = hv.render(el, backend='bokeh')
    except Exception:
        # If rendering fails (e.g. some elements can't be rendered directly), return empty
        return []

    hover_tool = None
    if hasattr(bokeh_fig, 'tools'):
        for tool in bokeh_fig.tools:
            if isinstance(tool, HoverTool):
                hover_tool = tool
                break
    
    if hover_tool:
        return hover_tool.tooltips
    return []


@register_method(classes=[hv.core.dimension.ViewableElement], namespace='ms')
def update_tooltips(el: hv.core.dimension.ViewableElement, tooltips: dict[str, str]):
    """
    Updates the tooltips of the element.
    'tooltips' is a dictionary where keys are the label OR the field name,
    and values are the new format string (e.g. '@{field}{format}').
    
    If a key matches an existing tooltip label or field name, that tooltip is updated.
    If a key does not match, a new tooltip is added.
    """
    import re
    from bokeh.models import HoverTool

    # Recursive step for Overlay and NdOverlay
    if isinstance(el, (hv.Overlay, hv.NdOverlay)):
        new_items = []
        for k, v in el.items():
            new_items.append((k, update_tooltips(v, tooltips)))
        
        # Reconstruct the container
        if isinstance(el, hv.NdOverlay):
            new_el = hv.NdOverlay(new_items, kdims=el.kdims, label=el.label, group=el.group)
        else:
            new_el = hv.Overlay([v for k, v in new_items], label=el.label, group=el.group)
        
        # Copy opts from original element
        # We need to be careful to copy only valid options or just use .opts()
        # el.opts.get() returns a dictionary of options
        if hasattr(el, 'opts'):
             # This gets the options applied to the element
             opts = el.opts.get()
             if opts:
                 new_el = new_el.opts(opts)
        
        return new_el

    # Render to Bokeh to get current tooltips
    try:
        bokeh_fig = hv.render(el, backend='bokeh')
    except Exception:
        # If rendering fails, just return original element
        return el

    current_tooltips = []
    current_formatters = {}
    if hasattr(bokeh_fig, 'tools'):
        for tool in bokeh_fig.tools:
            if isinstance(tool, HoverTool):
                current_tooltips = tool.tooltips
                # Preserve existing formatters
                if hasattr(tool, 'formatters') and tool.formatters:
                    current_formatters = dict(tool.formatters)
                break
    
    if not current_tooltips:
        # If no tooltips found, start with empty list
        current_tooltips = []

    new_tooltips = []
    processed_keys = set()
    formatters = dict(current_formatters)  # Start with existing formatters

    # Helper to extract field name from value string
    # Matches @field, @{field}, @{field}{format}
    # Group 1: field name inside {}, Group 2: field name without {}
    field_regex = re.compile(r'@(?:\{([^\}]+)\}|(\w+))')
    
    # Regex to detect datetime format codes (%, followed by letter)
    datetime_format_regex = re.compile(r'%[a-zA-Z]')

    for label, value in current_tooltips:
        updated = False
        field_name = None
        
        match = field_regex.search(value)
        if match:
            field_name = match.group(1) or match.group(2)

        # Check if we should update this tooltip
        # Match by label
        if label in tooltips:
            new_format = tooltips[label]
            processed_keys.add(label)
            updated = True
        # Match by field name
        elif field_name and field_name in tooltips:
            new_format = tooltips[field_name]
            processed_keys.add(field_name)
            updated = True
        
        if updated:
            # Construct new value string. 
            if '@' in new_format:
                 new_value = new_format
            else:
                if not field_name:
                    new_value = value 
                else:
                    new_value = f"@{{{field_name}}}{{{new_format}}}"
                    # Check if this is a datetime format
                    if datetime_format_regex.search(new_format):
                        formatters[f"@{{{field_name}}}"] = 'datetime'
            new_tooltips.append((label, new_value))
        else:
            new_tooltips.append((label, value))

    # Add new tooltips for keys that weren't matched
    for key, format_str in tooltips.items():
        if key not in processed_keys:
            # Assume key is the field name
            if '@' in format_str:
                new_value = format_str
            else:
                new_value = f"@{{{key}}}{{{format_str}}}"
                # Check if this is a datetime format
                if datetime_format_regex.search(format_str):
                    formatters[f"@{{{key}}}"] = 'datetime'
            new_tooltips.append((key, new_value))

    # Apply the updated tooltips with formatters
    if formatters:
        hover = HoverTool(tooltips=new_tooltips, formatters=formatters)
    else:
        hover = HoverTool(tooltips=new_tooltips)
    return el.opts(tools=[hover])
