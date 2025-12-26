import pandas as pd
import scipy.stats
import holoviews as hv
import hvplot.pandas


def hv_qqplot(Xs):
    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(Xs, plot=None, fit=True)

    _df = pd.DataFrame({"Theoretical Quantiles": osm, "Sample Quantiles": osr})
    scatter = _df.hvplot.scatter(_df.columns[0], _df.columns[1])

    reg_line = hv.Slope(slope, intercept).opts(color="red")
    res = scatter * reg_line
    return res
