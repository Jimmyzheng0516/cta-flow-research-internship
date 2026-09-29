"""CTA gamma: the Table 3 reaction curve as a daily, per-market measure of mechanical
managed-money flow per unit price move, and the tests of what it predicts.

Modules
  config    paths and fixed settings
  data      loaders and the return / timing conventions
  measures  the curve measures (gamma per 1%, per sigma, forced flow at 2 sigma, ...)
  panel     the estimation panel: measures + controls + forward targets
  stats     panel regressions, out-of-sample forecast comparison, sorts
  overlay   the value test: volatility-targeted books sized with and without the measures
  product   the daily index file and the cross-market snapshot
  plots     figure helpers (matplotlib, plain style)
"""
from . import config, data, measures, panel, stats, overlay, product, plots  # noqa: F401
