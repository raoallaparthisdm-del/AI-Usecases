from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


from models import Fact


AGE_BINS = [18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def build_knowledge_base(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Date"].dt.year.astype(str)

    monthly_sales = df.groupby("Month")["Sales"].sum().sort_index()
    quarterly_sales = df.groupby("Quarter")["Sales"].sum().sort_index()
    yearly_sales = df.groupby("Year")["Sales"].sum().sort_index()

    product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    product_region_sales = (
        df.groupby(["Product", "Region"])["Sales"].sum().sort_values(ascending=False)
    )

    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    segment = (
        df.groupby(["Age_Group", "Customer_Gender"])  # type: ignore[call-arg]
        .agg(
            count=("Sales", "count"),
            avg_sales=("Sales", "mean"),
            avg_satisfaction=("Customer_Satisfaction", "mean"),
        )
        .sort_values("count", ascending=False)
    )

    stats = {
        "sales_mean": df["Sales"].mean(),
        "sales_median": df["Sales"].median(),
        "sales_std": df["Sales"].std(),
        "sales_min": df["Sales"].min(),
        "sales_max": df["Sales"].max(),
        "age_mean": df["Customer_Age"].mean(),
        "age_median": df["Customer_Age"].median(),
        "age_std": df["Customer_Age"].std(),
        "satisfaction_mean": df["Customer_Satisfaction"].mean(),
        "satisfaction_median": df["Customer_Satisfaction"].median(),
        "satisfaction_std": df["Customer_Satisfaction"].std(),
    }

    return {
        "meta": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "start_date": df["Date"].min().date().isoformat(),
            "end_date": df["Date"].max().date().isoformat(),
        },
        "sales_by_period": {
            "monthly": monthly_sales,
            "quarterly": quarterly_sales,
            "yearly": yearly_sales,
        },
        "product_sales": product_sales,
        "region_sales": region_sales,
        "product_region_sales": product_region_sales,
        "customer_segment": segment,
        "stats": stats,
    }


def build_fact_list(kb: Dict[str, Any]) -> List[Fact]:
    facts: List[Fact] = []

    meta = kb["meta"]
    facts.append(
        Fact(
            text=(
                "Dataset metadata: rows={rows}, date_range={start} to {end}."
            ).format(rows=meta["rows"], start=meta["start_date"], end=meta["end_date"]),
            tags=["meta", "rows", "date", "range"],
        )
    )

    stats = kb["stats"]
    facts.append(
        Fact(
            text=(
                "Sales stats: mean={sales_mean:.2f}, median={sales_median:.2f}, "
                "std={sales_std:.2f}, min={sales_min:.2f}, max={sales_max:.2f}."
            ).format(**stats),
            tags=["sales", "stats", "mean", "median", "std", "min", "max"],
        )
    )
    facts.append(
        Fact(
            text=(
                "Customer age stats: mean={age_mean:.2f}, median={age_median:.2f}, "
                "std={age_std:.2f}."
            ).format(**stats),
            tags=["age", "customer", "stats", "mean", "median", "std"],
        )
    )
    facts.append(
        Fact(
            text=(
                "Customer satisfaction stats: mean={satisfaction_mean:.2f}, "
                "median={satisfaction_median:.2f}, std={satisfaction_std:.2f}."
            ).format(**stats),
            tags=["satisfaction", "customer", "stats", "mean", "median", "std"],
        )
    )

    for period, series in kb["sales_by_period"].items():
        for label, value in series.items():
            facts.append(
                Fact(
                    text=f"Sales in {period} {label}: total={value:.2f}.",
                    tags=["sales", "time", period, str(label)],
                )
            )

    for product, value in kb["product_sales"].items():
        facts.append(
            Fact(
                text=f"Product sales total for {product}: {value:.2f}.",
                tags=["sales", "product", product.lower()],
            )
        )

    for region, value in kb["region_sales"].items():
        facts.append(
            Fact(
                text=f"Region sales total for {region}: {value:.2f}.",
                tags=["sales", "region", region.lower()],
            )
        )

    for (product, region), value in kb["product_region_sales"].items():
        facts.append(
            Fact(
                text=(
                    f"Sales for {product} in {region}: {value:.2f}."
                ),
                tags=["sales", "product", "region", product.lower(), region.lower()],
            )
        )

    for (age_group, gender), row in kb["customer_segment"].iterrows():
        facts.append(
            Fact(
                text=(
                    f"Segment {age_group} {gender}: count={row['count']}, "
                    f"avg_sales={row['avg_sales']:.2f}, "
                    f"avg_satisfaction={row['avg_satisfaction']:.2f}."
                ),
                tags=["segment", "customer", "age", str(age_group).lower(), str(gender).lower()],
            )
        )

    return facts
