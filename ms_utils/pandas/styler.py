import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import panel as pn
from typing import Sequence

from ms_utils.method_registration import register_method
from ms_utils.string_formatters import get_formatter


@register_method([Styler], namespace="ms")
def style_describe_table(styler: Styler, formatter=None):
    df = styler.data
    subset_pct = pd.IndexSlice[df.index.str.startswith("pct_"), :]
    subset_int = pd.IndexSlice[df.index.str.startswith("count"), :]
    res = styler.format("{:.01e}").format("{:.02%}", subset=subset_pct).format("{:,.0f}", subset=subset_int)
    if formatter is not None:
        formatter = get_formatter(formatter)
        subset = pd.IndexSlice[df.index.str.fullmatch("(mean|std|min|max|[0-9]+%)"), :]
        res = res.format(formatter, subset=subset)

    res = res.bar(subset=subset_pct, vmin=0, vmax=1, align="left", color="red")
    res = res.applymap(lambda v: f"text-shadow: 1px 1px 3px red" if v == 0 else None, subset=subset_pct)
    return res


@register_method([Styler], namespace="ms")
def style_corr_matrix(
    styler: Styler,
    show_upper_triangle=False,
    show_diagonal: bool = False,
    *,
    vmin=0.0,
    vmax=1.0,
    format="{:.02%}",
    cmap="coolwarm",
):
    res = styler.background_gradient(cmap=cmap, vmin=vmin, vmax=vmax).format(format)

    if show_upper_triangle is False:
        func_hide_upper_triangle = lambda data: np.where(
            np.triu(np.ones_like(data), k=int(bool(show_diagonal))), "background-color: #f1f1f1; color: #f1f1f1;", ""
        )
        res = res.apply(func_hide_upper_triangle, axis=None)

    return res


@register_method([Styler], namespace="ms")
def panel(styler: Styler, **kwargs):
    """Wrap an object into pn.panel for display in panel layouts"""
    return pn.panel(styler, **kwargs)


@register_method([Styler], namespace="ms")
def format(
    styler: Styler,
    formatter: "ExtFormatter | None" = None,
    subset: "Subset | None" = None,
    include_dtypes: Sequence | None = None,
    na_rep: "str | None" = None,
    **kwargs,
):
    """Utility method to format dataframe.

    Similar pandas.io.formats.style.Styler.format, but with some extra convenience.

    :param formatter: formatter to be applied. See `msu.string_formatters.get_formatter` for details.
    :param subset: list of regexps to match columns names against
    :param include_dtypes: list of dtypes to apply formatting to
    :param na_rep: representation for missing values
    :param kwargs: other parameters passed to `pandas.io.formats.style.Styler.format`
    """
    formatter = get_formatter(formatter)
    cols = styler.data.columns
    if subset is None:
        subset = set(cols)
    else:
        subset = set.union(*[set(cols[cols.str.fullmatch(regexp)]) for regexp in subset])

    if include_dtypes is not None:
        subset2 = set(styler.data.select_dtypes(include=include_dtypes).columns)
        subset = subset & subset2

    return styler.format(formatter, subset=list(subset), na_rep=na_rep, **kwargs)

    # subset=styler.data.select_dtypes(include=int).columns
    # .format("{:.3g}", subset=s.data.select_dtypes(include=float).columns


@register_method([Styler], namespace="ms")
def left_align_index(styler: Styler):
    """Left-align index columns in the styled DataFrame.

    By default, pandas Styler right-aligns index columns. This method applies
    CSS to left-align them for better readability, especially for text-based indices.

    Parameters
    ----------
    styler : Styler
        The Styler object to modify.

    Returns
    -------
    Styler
        The modified Styler with left-aligned index columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]},
    ...                   index=['Row 1', 'Row 2', 'Row 3'])
    >>> df.style.ms.left_align_index()

    Notes
    -----
    This method uses CSS selectors to target index `th` elements with class `row_heading`.
    Works with both single and MultiIndex.
    """
    # Use set_table_styles to target the index cells specifically
    return styler.set_table_styles([{"selector": "th.row_heading", "props": [("text-align", "left")]}], overwrite=False)
