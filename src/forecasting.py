from sklearn.linear_model import LinearRegression
import numpy as np

def forecast_revenue(df, days=7):
    daily = df.groupby("date")["revenue"].sum().reset_index()
    X = np.arange(len(daily)).reshape(-1, 1)
    y = daily["revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(daily), len(daily) + days).reshape(-1, 1)
    return model.predict(future_X)
