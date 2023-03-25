import bokeh.models
import holoviews as hv

def apply_defaults():

  hv.opts.defaults(
    hv.opts.Curve(show_grid=True, padding=0.03,
                  default_tools=['xbox_zoom', 'box_zoom', 'pan', 'undo', 'redo', 'reset'],
                  active_tools=['xbox_zoom'],
                  toolbar='above'
                  ),
    hv.opts.Scatter(show_grid=True, padding=0.03,
                   default_tools=['box_zoom', 'pan', 'undo', 'redo', 'reset'],
                   toolbar='above'
                   ),
    hv.opts.BoxWhisker(show_grid=True, toolbar='above'),
    hv.opts.Bars(show_grid=True, xrotation=90, padding=0.03, toolbar='above'),
    hv.opts.Histogram(show_grid=True, toolbar='above'),
    hv.opts.Distribution(show_grid=True, toolbar='above'),
    hv.opts.Overlay(click_policy='hide', toolbar='above', legend_position='right'),
    hv.opts.NdOverlay(click_policy='hide', toolbar='above', legend_position='right'),
    hv.opts.Layout(merge_tools=False),
  )
