import numpy as np
import pandas as pd

FREQ_FACTORS = {"Q": 0.25, "SA": 0.5, "A": 1}


def coupons_matrix(
    time_to_maturities: np.ndarray, annual_coupons: np.ndarray, frequencies: np.ndarray
) -> np.ndarray:
    n = len(time_to_maturities)
    coupon_values = annual_coupons * frequencies
    coupons = np.zeros((n, n))
    j = 0
    for i in range(n):
        step = time_to_maturities / frequencies[i]
        time = 0
        f = frequencies[i]
        for j, t in enumerate(step):
            if t.is_integer() and time <= time_to_maturities[i]:
                coupons[i, j] = coupon_values[i]
            time = time + f
    return coupons


def principal_matrix(
    time_to_maturities: np.ndarray, bond_principal: np.ndarray
) -> np.ndarray:
    n = len(time_to_maturities)
    return np.diag(bond_principal) * np.identity(n)


def solve_zero_rates(coupons, princ, time_to_maturities, bond_prices):
    X = coupons + princ
    R = np.linalg.solve(X, bond_prices)
    return -np.log(R) / time_to_maturities


def main():
    header = [
        "bond_principal",
        "time_to_maturity",
        "annual_coupon",
        "bond_price",
        "frequency",
    ]
    raw_data = [
        [100, 0.25, 0, 99.6, 0.25],
        [100, 0.5, 0, 99.0, 0.5],
        [100, 1.0, 0, 97.8, 1.0],
        [100, 1.5, 4, 102.5, 0.5],
        [100, 2.0, 5, 105.0, 0.5],
    ]
    data = pd.DataFrame(raw_data, columns=header)
    tmt = data["time_to_maturity"].to_numpy()
    annual_coupons = data["annual_coupon"].to_numpy()
    frequencies = data["frequency"].to_numpy()
    principals = data["bond_principal"].to_numpy()
    prices = data["bond_price"].to_numpy()
    coupons = coupons_matrix(tmt, annual_coupons, frequencies)
    princ = principal_matrix(tmt, principals)
    zero_rates = solve_zero_rates(coupons, princ, tmt, prices)
    df_zero_rates = pd.DataFrame({"maturity": tmt, "zero_rate": zero_rates})
    print(df_zero_rates)


if __name__ == "__main__":
    main()
