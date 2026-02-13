def calculate_kpis(df):
    return {
        "Total Revenue": df["revenue"].sum(),
        "Total Profit": df["profit"].sum(),
        "Average Daily Revenue": df.groupby("date")["revenue"].sum().mean()
    }
