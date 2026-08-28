# Homework CSVs (Jackovich / ML4T)

This folder is a **tiny excerpt** of the Udacity/Georgia Tech ML4T Yahoo-style
CSVs mirrored at
https://github.com/JeffreyJackovich/machine-learning-for-trading
(commit `08763f80649604158040dd1559925b3b90c97d32`).

Included symbols: `SPY`, `FAKE1`, `FAKE2`.

Date windows (enough for tests, not a full history):

- 2010-06-10 .. 2010-06-18
- 2010-12-30 .. 2011-01-05

`FAKE2` has no bars from 2010-06-15 through 2010-12-31 in the original dump
(three holes; this excerpt keeps the start of the 2010 gap and the 2011
resume). `FAKE1` matches SPY dates in this window. Files are still newest-first
with columns `Date, Open, High, Low, Close, Volume, Adj Close`.

This is stale homework data (FAKE names, dead tickers in the full dump, prices
that predate later splits). It is **not** a live market vendor.
