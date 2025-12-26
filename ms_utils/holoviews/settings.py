import bokeh.models
import holoviews as hv


def apply_defaults():
    hv.opts.defaults(
        hv.opts.Curve(
            show_grid=True,
            padding=0.03,
            default_tools=["auto_box_zoom", "crosshair", "pan", "undo", "redo", "reset"],
            active_tools=["box_zoom"],
            toolbar="above",
        ),
        hv.opts.Scatter(
            show_grid=True,
            padding=0.03,
            default_tools=["auto_box_zoom", "crosshair", "pan", "undo", "redo", "reset"],
            toolbar="above",
        ),
        hv.opts.BoxWhisker(show_grid=True, toolbar="above"),
        hv.opts.Bars(
            show_grid=True,
            xrotation=90,
            padding=0.03,
            default_tools=["auto_box_zoom", "pan", "wheel_zoom", "undo", "redo", "reset"],
            active_tools=["pan"],
            toolbar="above",
        ),
        hv.opts.Histogram(
            show_grid=True,
            default_tools=["auto_box_zoom", "pan", "wheel_zoom", "undo", "redo", "reset"],
            active_tools=["pan"],
            toolbar="above",
        ),
        hv.opts.Distribution(
            show_grid=True,
            default_tools=["auto_box_zoom", "pan", "wheel_zoom", "undo", "redo", "reset"],
            active_tools=["pan"],
            toolbar="above",
        ),
        hv.opts.HeatMap(
            frame_width=450,
            frame_height=450,
            xrotation=90,
            padding=0,
            cmap="fire",
            colorbar=True,
            active_tools=[],
            toolbar="above",
        ),
        hv.opts.Contours(
            show_grid=True,
            padding=0.01,
            default_tools=["auto_box_zoom", "crosshair", "pan", "undo", "redo", "reset"],
            active_tools=["box_zoom"],
            toolbar="above",
        ),
        hv.opts.Image(
            show_grid=True,
            padding=0.01,
            default_tools=["auto_box_zoom", "wheel_zoom", "pan", "undo", "redo", "reset"],
            active_tools=["box_zoom"],
            toolbar="above",
        ),
        hv.opts.Overlay(click_policy="hide", toolbar="above", legend_position="right"),
        hv.opts.HLine(line_dash=[4, 8], line_width=2),
        hv.opts.VLine(line_dash=[4, 8], line_width=2),
        hv.opts.NdOverlay(click_policy="hide", toolbar="above", legend_position="right"),
        hv.opts.Layout(merge_tools=False),
    )
