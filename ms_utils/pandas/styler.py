import re
from typing import Sequence
from pandas.io.formats.style import Styler
import panel as pn

from method_registration import register_method


def get_formatter(formatter_str: str):
   if formatter_str.startswith("{") and formatter_str.endswith("}"):
     return formatter_str
   match = re.match(r'(?P<prefix>[^0-9]*)(?P<digits>\d*)$', formatter_str)
   prefix = match.group("prefix")
   digits = match.group("digits")
   if prefix in ("usd", "$"):
     return "${:,.0" + (digits or "0") + "f}"
   elif prefix in ("%", "pct"):
     return "{:.0" + (digits or "2") + "%}"
   elif prefix in ("#", "num", "count") and digits == '':
     return "{:,d}"
   else:
     raise ValueError(f"cannot parse formatter string {formatter_str!r}" )


@register_method(class_=Styler, namespace="ms")
def format(styler: Styler,
  formatter: str = None,
  subset: Sequence = None,
  include_dtypes: Sequence = None,
  **kwargs) -> Styler:
  """
  :param subset: list of regexps to match columns names against
  """
  formatter = get_formatter(formatter)
  cols = styler.data.columns
  if subset is None:
    subset = set(cols)
  else:
    subset = set.union(*[set(cols[cols.str.match(regexp)]) for regexp in subset])
 
  if include_dtypes is not None:
    subset2 = set(styler.data.select_dtypes(include=include_dtypes).columns)
    subset = subset & subset2
 
  return styler.format(formatter, subset=list(subset), **kwargs)


@register_method(class_=Styler, namespace="ms")
def panel(styler: Styler, **kwargs):
  return pn.panel(styler, **kwargs)

