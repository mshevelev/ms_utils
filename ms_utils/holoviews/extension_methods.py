from typing import Literal
import numpy as np
import bokeh.models
import holoviews as hv

from method_registration import register_method

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


@register_method(classes=[hv.core.dimension.ViewableElement], namespace="ms")
def update_tooltips(el: hv.core.dimension.ViewableElement, tooltips: dict[str, str]):
  """
  Update tooltips for the element.
  
  Args:
      el: The HoloViews element.
      tooltips: A dictionary mapping tooltip labels or field names to format strings.
                Example: {'Value': '0.00', 'date': '%Y-%m-%d'}
  """
  import re
  from bokeh.models import HoverTool

  if isinstance(el, (hv.NdOverlay, hv.Overlay)):
    new_items = []
    for k, v in el.items():
      new_v = update_tooltips(v, tooltips)
      new_items.append((k, new_v))
    
    # Reconstruct the overlay with updated items
    # We use type(el) to preserve the exact class (NdOverlay or Overlay)
    # We also need to preserve other properties/opts
    new_el = type(el)(new_items, **dict(el.param.values()))
    # Copy opts from original
    new_el = new_el.opts(el.opts.get())
    return new_el

  # Render to Bokeh to get default tooltips
  try:
    bokeh_fig = hv.render(el, backend='bokeh')
  except Exception:
    # If rendering fails (e.g. some elements might not be renderable directly), return el
    return el

  current_tooltips = None
  # Find existing HoverTool
  if hasattr(bokeh_fig, 'tools'):
    for tool in bokeh_fig.tools:
      if isinstance(tool, HoverTool):
        current_tooltips = tool.tooltips
        break
  
  if not current_tooltips:
    return el

  new_tooltips = []
  matched_keys = set()

  for label, value in current_tooltips:
    # Value format: @field or @{field} or @{field}{format}
    # Regex to extract field name. 
    # Matches @field, @{field}, @{field}{fmt}
    match = re.match(r'@(?:\{([^\}]+)\}|(\w+))', value)
    field_name = match.group(1) or match.group(2) if match else None
    
    # Check if label matches
    if label in tooltips:
      new_format = tooltips[label]
      matched_keys.add(label)
      # Reconstruct value with new format
      # We assume value starts with @... and we append {format}
      # If value is @{field}{old_fmt}, we want @{field}{new_fmt}
      # We can use the regex match to get the base part
      if match:
        base_part = match.group(0) # @{field} or @field
        if '{' not in base_part: # @field -> @{field}
             base_part = f"@{{{field_name}}}"
        new_value = f"{base_part}{{{new_format}}}"
        new_tooltips.append((label, new_value))
      else:
         # Fallback if regex didn't match (e.g. complex expression?)
         # Just append format?
         new_tooltips.append((label, value))
      
    # Check if field name matches
    elif field_name and field_name in tooltips:
      new_format = tooltips[field_name]
      matched_keys.add(field_name)
      # Construct new value
      base_part = match.group(0) # @{field} or @field
      if '{' not in base_part: # @field -> @{field}
           base_part = f"@{{{field_name}}}"
      new_value = f"{base_part}{{{new_format}}}"
      new_tooltips.append((label, new_value))
      
    else:
      new_tooltips.append((label, value))
  
  # Add new tooltips for keys that were not matched
  for key, fmt in tooltips.items():
    if key not in matched_keys:
      # We assume the key is the field name
      # Format: (key, @{key}{fmt})
      new_value = f"@{{{key}}}{{{fmt}}}"
      new_tooltips.append((key, new_value))
      
  # Create new HoverTool with updated tooltips
  hover = HoverTool(tooltips=new_tooltips)
  
  # Apply to element
  return el.opts(tools=[hover])
