import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import panel as pn
from typing import Sequence, Literal

from ms_utils.method_registration import register_method
from ms_utils.string_formatters import get_formatter


@register_method([Styler])
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


@register_method([Styler])
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


@register_method([Styler])
def panel(styler: Styler, **kwargs):
    """Wrap an object into pn.panel for display in panel layouts"""
    return pn.panel(styler, **kwargs)


@register_method([Styler])
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


@register_method([Styler])
def align_index(
    styler: Styler,
    align: Literal["left", "right", "center"] = "left",
    *,
    align_header: bool = True,
    align_index_values: bool = True,
):
    """Align index columns (headers and/or values) in the styled DataFrame.

    By default, pandas Styler right-aligns index columns. This method allows you to
    control the alignment of both index headers and index values independently.

    Parameters
    ----------
    styler : Styler
        The Styler object to modify.
    align : {'left', 'right', 'center'}, default 'left'
        Text alignment to apply.
    align_header : bool, default True
        Whether to align index header cells (the index name labels).
    align_index_values : bool, default True
        Whether to align index value cells (the actual index values).

    Returns
    -------
    Styler
        The modified Styler with aligned index columns.

    Examples
    --------
    >>> import pandas as pd

    **Left-align index (default):**

    >>> df = pd.DataFrame({'A': [1, 2]}, index=['Row 1', 'Row 2'])
    >>> df.style.ms.align_index()

    **Center-align index:**

    >>> df.style.ms.align_index('center')

    **Only align values, not headers:**

    >>> df.style.ms.align_index('left', align_header=False)

    **With MultiIndex:**

    >>> arrays = [['A', 'A', 'B'], ['X', 'Y', 'X']]
    >>> idx = pd.MultiIndex.from_arrays(arrays, names=['First', 'Second'])
    >>> df = pd.DataFrame({'Value': [1, 2, 3]}, index=idx)
    >>> df.style.ms.align_index('left')  # Aligns both levels

    Notes
    -----
    - Uses CSS selectors to target index elements:
      - `th.index_name`: Index header cells (e.g., "Date", "Stock")
      - `th.row_heading`: Index value cells (e.g., "2023-01-01", "AAPL")
    - Works with both single index and MultiIndex
    - Setting `align_header=False` and `align_index_values=False` returns unchanged styler
    """
    styles = []

    if align_index_values:
        # Target the index value cells (th.row_heading)
        styles.append({"selector": "th.row_heading", "props": [("text-align", align)]})

    if align_header:
        # Target the index header cells (th.index_name)
        styles.append({"selector": "th.index_name", "props": [("text-align", align)]})

    if styles:
        return styler.set_table_styles(styles, overwrite=False)
    else:
        return styler
