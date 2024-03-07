import numpy as np
from statsmodels import api as sm
import panel as pn
import pandas as pd
import holoviews as hv
import colorcet as cc
import hvplot.pandas
import holoviews.operation.datashader as hd
import param


class ScatterPlot(param.Parameterized):
    df = param.DataFrame()

    def __init__(self, **params):
        super(ScatterPlot, self).__init__(**params)
        df = self.df
        self.select_x = pn.widgets.Select(options=list(df.columns), value=df.columns[0], name="x_axis")
        self.select_y = pn.widgets.Select(options=list(df.columns), value=df.columns[1], name="y_axis")
        self.w_fit_reg_line = pn.widgets.Checkbox(value=False, name="fit_reg_line")
        self.w_color_select = pn.widgets.Select(options=list(palette[0] for palette in cc.palette_n.items() if "_" not in palette[0]), 
                                                     value="fire", name="point_color")
        self.w_point_select = pn.widgets.Select(options=["Tap", "Hover"], value="Tap", name="selection_method")
        self.w_point_spread = pn.widgets.IntInput(value=3, start=1, end=20, name="point_spread")
        
#        self.info_pane = pn.pane.HTML("")
        self.to_update = False
        self.x_range = None
        self.y_range = None
        self.last_x_value = None
        self.last_y_value = None
        self.count = 0

    def return_controls(self):
        return pn.Column(
            self.select_x,
            self.select_y,
            self.w_fit_reg_line,
            self.w_color_select,
            self.w_point_select,
            self.w_point_spread
            )

    @param.depends('select_x.value', 'select_y.value', 'w_fit_reg_line.value', 'w_color_select.value', 'w_point_select.value', 'w_point_spread.value')
    def plot_scatter(self):
        col_x = self.select_x.value
        col_y = self.select_y.value
        fit_reg_line = self.w_fit_reg_line.value
        color_select= self.w_color_select.value
        point_select = self.w_point_select.value

        if self.last_x_value == col_x and self.last_y_value == col_y:
            self.to_update = True
        else:
            self.x_range = None
            self.y_range = None
        self.last_x_value = col_x
        self.last_y_value = col_y

        if col_x != col_y:
            scatter_plot = self.df.hvplot.scatter(x=col_x, y=col_y, rasterize=True).opts(frame_width=400, frame_height=400)
        else:
            df_new = self.df.copy()
            df_new[col_x+"_2"] = df_new[col_x]
            scatter_plot = df_new.hvplot.scatter(x=col_x, y=col_x+"_2", rasterize=True).opts(frame_width=400, frame_height=400)

        raster = hd.spread(scatter_plot, px=self.w_point_spread.value)

        def range_hook(plot, element):
            if self.x_range is not None and self.y_range is not None and self.to_update:
                plot.state.x_range.start = self.x_range[0]
                plot.state.x_range.end = self.x_range[1]
                plot.state.y_range.start = self.y_range[0]
                plot.state.y_range.end = self.y_range[1]
                self.to_update = False

        def transform_points(df):
 #           self.info_pane.object = df.head().to_html()
            return df

        highlighter = hd.inspect_points.instance(streams=[hv.streams.Tap if point_select=="Tap" else hv.streams.PointerXY], transform=transform_points)
        highlight = highlighter(raster).opts(color="blue", tools=["hover"], marker="o", size=10, fill_alpha=0, hooks=[range_hook])

        range_stream = hv.streams.RangeXY(highlight)
        def stream_points(points, x_range, y_range):
            if x_range is None:
                self.count = 0
            else:
                self.count += 1
            if x_range is not None and self.count > 2:
                self.x_range = x_range
                self.y_range = y_range

            return points

        scatter = raster.apply(stream_points, streams=[range_stream]).opts(cmap=color_select) * highlight
        if fit_reg_line is True:
            df = self.df.dropna(how='any')
            ols_fit = sm.OLS(df[col_y], sm.add_constant(df[[col_x]])).fit()
            reg_line = hv.Slope(ols_fit.params[1], ols_fit.params[0]).opts(color='red', alpha=0.3, line_width=1)
            scatter = scatter * reg_line

        res = pn.Column(
        scatter,
#        self.info_pane,
        )
        if fit_reg_line is True:
            res = pn.Row(res, pn.pane.Str(ols_fit.summary2()))
        return res