"""Turn the raw DataCo CSV into a clean reference table.

One row = one order line that was actually shipped (cancelled orders removed).
We keep only the columns that matter for procurement / fulfilment and drop
everything that identifies people (names, emails, passwords, addresses).
"""

from pathlib import Path

import pandas as pd

CATEGORICAL = ["shipping_mode", "market", "customer_segment", "department", "payment_type"]
NUMERIC = ["quantity", "unit_price", "discount_rate", "order_month", "order_dow",
           "days_scheduled", "days_real"]
TARGET = "late"
COLUMNS = CATEGORICAL + NUMERIC + [TARGET]

PERSONAL_FIELDS = ["Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
                   "Customer Street", "Customer Zipcode", "Customer City", "Customer State",
                   "Latitude", "Longitude"]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the CSV as published (it is latin-1 encoded, not UTF-8)."""
    return pd.read_csv(csv_path, encoding="latin-1")


def load_reference(csv_path: str | Path) -> pd.DataFrame:
    raw = load_raw(csv_path).drop(columns=PERSONAL_FIELDS)
    raw = raw[raw["Delivery Status"] != "Shipping canceled"]  # never shipped -> no delivery to model

    order_date = pd.to_datetime(raw["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    df = pd.DataFrame({
        "order_date": order_date,
        "shipping_mode": raw["Shipping Mode"],
        "market": raw["Market"],
        "customer_segment": raw["Customer Segment"],
        "department": raw["Department Name"].str.strip(),  # "Health and Beauty " has a trailing space
        "payment_type": raw["Type"],
        "quantity": raw["Order Item Quantity"],
        "unit_price": raw["Order Item Product Price"].round(2),
        "discount_rate": raw["Order Item Discount Rate"].round(2),
        "order_month": order_date.dt.month,
        "order_dow": order_date.dt.dayofweek,
        "days_scheduled": raw["Days for shipment (scheduled)"],
        "days_real": raw["Days for shipping (real)"],
        TARGET: raw["Late_delivery_risk"],
    })
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, holdout_frac: float = 0.3, seed: int = 42):
    """Train part (what generators may see) and holdout part (never seen, used to judge them)."""
    holdout = df.sample(frac=holdout_frac, random_state=seed)
    train = df.drop(holdout.index)
    return train.reset_index(drop=True), holdout.reset_index(drop=True)