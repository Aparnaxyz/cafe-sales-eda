"""Cleaning steps for the café sales data, so the notebook and any later script use the same logic."""
import numpy as np
import pandas as pd

BAD_TOKENS = ["", "ERROR", "UNKNOWN"]
LINKED = ["Item", "Quantity", "Price Per Unit", "Total Spent"]


def load_raw(path):
    """Read every column as text so the bad tokens stay visible."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def to_missing_and_typed(raw):
    """Turn bad tokens into real NaN and give each column its proper type."""
    df = raw.replace(BAD_TOKENS, np.nan)
    for col in ["Quantity", "Price Per Unit", "Total Spent"]:
        df[col] = pd.to_numeric(df[col])
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    return df


def learn_menu(df):
    """Each item sells at exactly one price. Fail loudly if that ever stops being true."""
    known = df.dropna(subset=["Item", "Price Per Unit"])
    prices_per_item = known.groupby("Item")["Price Per Unit"].nunique()
    assert (prices_per_item == 1).all(), "An item has more than one price"
    return known.groupby("Item")["Price Per Unit"].first().to_dict()


def fill_linked_columns(df, menu, passes=3):
    """Rebuild Item / Quantity / Price / Total from each other where the math allows it.

    Total = Quantity x Price, and each item has one price. A price that belongs to only one
    item also tells us the item. Each fix can unlock another, so the rules run a few passes.
    """
    df = df.copy()
    unique_price = {p: i for i, p in menu.items() if list(menu.values()).count(p) == 1}
    q, p, t, i = "Quantity", "Price Per Unit", "Total Spent", "Item"
    for _ in range(passes):
        m = df[p].isna() & df[i].notna()
        df.loc[m, p] = df.loc[m, i].map(menu)
        m = df[p].isna() & df[t].notna() & df[q].notna()
        df.loc[m, p] = df.loc[m, t] / df.loc[m, q]
        m = df[i].isna() & df[p].notna()
        df.loc[m, i] = df.loc[m, p].map(unique_price)
        m = df[q].isna() & df[t].notna() & df[p].notna()
        df.loc[m, q] = df.loc[m, t] / df.loc[m, p]
        m = df[t].isna() & df[q].notna() & df[p].notna()
        df.loc[m, t] = df.loc[m, q] * df.loc[m, p]
    return df


def clean(path):
    raw = load_raw(path)
    df = to_missing_and_typed(raw)
    menu = learn_menu(df)
    df = fill_linked_columns(df, menu)
    for col in ["Payment Method", "Location"]:
        df[col] = df[col].fillna("Not recorded")
    return df
