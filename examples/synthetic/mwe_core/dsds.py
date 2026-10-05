# -*- coding: utf-8 -*-
"""Portable DSDS core adapted from ``building_sampling/house_mc_sampling.py``.

The conditional Poisson GLMs, lognormal OLS, truncation, and count caps are
unchanged. Changes: use the supplied generator for Poisson draws; expose a
separate wall RNG; omit PIT plotting helpers and the local-data demo entrypoint.
All fitting in this example uses generated synthetic data.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def _zscore_arr(a: np.ndarray, mean_: float, std_: float) -> np.ndarray:
    return (a - mean_) / (std_ + 1e-9)


@dataclass
class HouseMCModel:
    col_candidates: Dict[str, list] = field(default_factory=lambda: {
        "W": ["tot_num_window", "windows", "num_windows"],
        "D": ["num_int_door_openings", "internal_doors", "num_doors"],
        "R": ["number_of_rooms", "rooms"],
        "L": ["total_inner_wall_length", "inner_wall_length"],
        "T": ["gross_area", "area", "total_area"],
        "P": ["tot_perimeter", "perimeter"],
        "F": ["floor", "floors", "num_floors"],
    })
    valid_ratio: float = 0.2
    random_state: int = 42

    # 拟合结果
    res_R: Optional[sm.GLM] = None
    res_W: Optional[sm.GLM] = None
    res_D: Optional[sm.GLM] = None
    ols_L: Optional[sm.OLS] = None
    sigma_L: Optional[float] = None

    # 训练统计量/缓存
    scalers_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    train_idx_: Optional[np.ndarray] = None
    valid_idx_: Optional[np.ndarray] = None
    df_: Optional[pd.DataFrame] = None
    Xb_train_: Optional[pd.DataFrame] = None
    Xb_valid_: Optional[pd.DataFrame] = None
    Xr_train_: Optional[pd.DataFrame] = None
    Xr_valid_: Optional[pd.DataFrame] = None
    XL_train_: Optional[pd.DataFrame] = None
    XL_valid_: Optional[pd.DataFrame] = None

    # ---------- Private API ----------
    def _auto_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for k, cand_list in self.col_candidates.items():
            for c in cand_list:
                if c in df.columns:
                    rename_map[c] = k
                    break
        df2 = df.rename(columns=rename_map)
        required = ["W", "D", "R", "L", "T", "P", "F"]
        missing = [c for c in required if c not in df2.columns]
        if missing:
            raise ValueError(f"Essential columns are  missing: {missing}. Available columns: {list(df.columns)}")
        return df2[required].copy()

    def _fit_poisson(self, y: pd.Series, X: pd.DataFrame):
        Xc = sm.add_constant(X)
        model = sm.GLM(y, Xc, family=sm.families.Poisson())
        res = model.fit()
        return res

    def _fit_lognormal(self, L: pd.Series, X: pd.DataFrame):
        # add constant with explicit names/order
        Xc = sm.add_constant(X, has_constant='add')
        Xc = pd.DataFrame(Xc, columns=['const'] + list(X.columns))

        y_ln = np.log(L.clip(lower=1e-9).values)
        ols = sm.OLS(y_ln, Xc).fit()

        # store the exact design columns used at fit-time
        self.L_exog_names = list(Xc.columns)  # e.g. ['const','T_z','F_z','P_z','R_z','D_z']

        sigma = float(np.std(ols.resid, ddof=Xc.shape[1]))
        return ols, sigma

    def _build_scalers(self, df: pd.DataFrame):
        self.scalers_.clear()
        for col in ["T", "F", "P", "R", "D"]:
            arr = df[col].astype(float).values
            self.scalers_[col] = (float(np.mean(arr)), float(np.std(arr, ddof=0)))

    def _z(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        out = {}
        for c in cols:
            mean_, std_ = self.scalers_[c]
            out[f"{c}_z"] = _zscore_arr(df[c].astype(float).values, mean_, std_)
        return pd.DataFrame(out, index=df.index)

    # ---------- Fitting ----------
    def fit(self, raw_df: pd.DataFrame):
        df = self._auto_rename(raw_df).dropna().reset_index(drop=True)
        for c in ["W", "D", "R", "F"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
        for c in ["L", "T", "P"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        split = int((1 - self.valid_ratio) * len(df))
        self.train_idx_, self.valid_idx_ = idx[:split], idx[split:]
        df_tr, df_va = df.iloc[self.train_idx_], df.iloc[self.valid_idx_]
        self.df_ = df

        self._build_scalers(df_tr)

        Xb_tr = self._z(df_tr, ["T", "F", "P"])
        Xb_va = self._z(df_va, ["T", "F", "P"])

        Rz_tr = self._z(df_tr, ["R"])
        Rz_va = self._z(df_va, ["R"])
        Xr_tr = pd.concat([Xb_tr, Rz_tr], axis=1)
        Xr_va = pd.concat([Xb_va, Rz_va], axis=1)

        XL_tr = pd.concat([Xb_tr, self._z(df_tr, ["R", "D"])], axis=1)
        XL_va = pd.concat([Xb_va, self._z(df_va, ["R", "D"])], axis=1)

        self.res_R = self._fit_poisson(df_tr["R"], Xb_tr)
        self.res_W = self._fit_poisson(df_tr["W"], Xr_tr)
        self.res_D = self._fit_poisson(df_tr["D"], Xr_tr)
        self.ols_L, self.sigma_L = self._fit_lognormal(df_tr["L"], XL_tr)

        self.Xb_train_, self.Xb_valid_ = Xb_tr, Xb_va
        self.Xr_train_, self.Xr_valid_ = Xr_tr, Xr_va
        self.XL_train_, self.XL_valid_ = XL_tr, XL_va
        return self

    # ---------- Condition Average and Quantiles ----------
    def _predict_poisson_mu(self, res, X: pd.DataFrame) -> np.ndarray:
        Xc = sm.add_constant(X.reset_index(drop=True), has_constant='add')
        mu = res.predict(Xc)
        return np.asarray(mu, dtype=float)  # drop index → positional-safe

    def predict_R_mean(self, new_df: pd.DataFrame) -> np.ndarray:
        return self._predict_poisson_mu(self.res_R, self._z(new_df, ["T", "F", "P"]))

    def predict_W_mean(self, new_df: pd.DataFrame) -> np.ndarray:
        Xb = self._z(new_df, ["T", "F", "P"])
        Rz = self._z(new_df, ["R"])
        return self._predict_poisson_mu(self.res_W, pd.concat([Xb, Rz], axis=1))

    def predict_D_mean(self, new_df: pd.DataFrame) -> np.ndarray:
        Xb = self._z(new_df, ["T", "F", "P"])
        Rz = self._z(new_df, ["R"])
        return self._predict_poisson_mu(self.res_D, pd.concat([Xb, Rz], axis=1))

    def _predict_mu_lnL(self, new_df: pd.DataFrame) -> np.ndarray:
        # build the SAME features, same order as training
        Xb = self._z(new_df, ["T", "F", "P"])  # -> T_z, F_z, P_z
        RzDz = self._z(new_df, ["R", "D"])  # -> R_z, D_z
        XL = pd.concat([Xb, RzDz], axis=1)  # columns: T_z, F_z, P_z, R_z, D_z

        # add constant, then align to training exog names
        Xc = sm.add_constant(XL, has_constant='add')
        Xc = pd.DataFrame(Xc, columns=['const'] + list(XL.columns))
        Xc = Xc.reindex(columns=self.L_exog_names, fill_value=0.0)

        # use numpy arrays for matmul to avoid pandas' dot quirks
        mu_ln = (Xc.values @ self.ols_L.params.values).astype(float)
        return mu_ln

    def predict_L(self, new_df: pd.DataFrame, kind: str = "mean", q: float = 0.5) -> np.ndarray:
        mu_ln = self._predict_mu_lnL(new_df)
        if kind == "median": return np.exp(mu_ln)
        if kind == "mean":   return np.exp(mu_ln + 0.5 * (self.sigma_L ** 2))
        if kind == "quantile":
            z = stats.norm.ppf(q)
            return np.exp(mu_ln + self.sigma_L * z)
        raise ValueError("kind must be 'mean', 'median', or 'quantile'")

    # ---------- Conditioning Lower/Upper Bounds ----------
    def predict_bounds_R(self, new_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        mu = self.predict_R_mean(new_df)
        return pd.DataFrame({
            "R_min": stats.poisson.ppf(alpha, mu).astype(int),
            "R_max": stats.poisson.ppf(1 - alpha, mu).astype(int),
        })

    def predict_bounds_W(self, new_df: pd.DataFrame, alpha: float = 0.05, w_floor: int = 2) -> pd.DataFrame:
        mu = self.predict_W_mean(new_df)
        lo = stats.poisson.ppf(alpha, mu).astype(int)
        hi = stats.poisson.ppf(1 - alpha, mu).astype(int)
        lo = np.maximum(lo, w_floor)  # a house at least have two windows
        return pd.DataFrame({"W_min": lo, "W_max": hi})

    def predict_bounds_D(self, new_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        mu = self.predict_D_mean(new_df)
        return pd.DataFrame({
            "D_min": stats.poisson.ppf(alpha, mu).astype(int),
            "D_max": stats.poisson.ppf(1 - alpha, mu).astype(int),
        })

    def predict_bounds_L(self, new_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        mu_ln = self._predict_mu_lnL(new_df)
        z_lo, z_hi = stats.norm.ppf(alpha), stats.norm.ppf(1 - alpha)
        lo = np.exp(mu_ln + self.sigma_L * z_lo)
        hi = np.exp(mu_ln + self.sigma_L * z_hi)
        return pd.DataFrame({"L_min": lo, "L_max": hi})

    # ---------- Sampling with  Truncation ----------
    @staticmethod
    def _trunc_poisson_sample(mu: float, lo: int, hi: int, rng: np.random.Generator) -> int:
        if hi < lo: hi = lo
        cdf_lo_minus = stats.poisson.cdf(max(lo - 1, -1), mu) if lo > 0 else 0.0
        cdf_hi = stats.poisson.cdf(hi, mu)
        u_adj = cdf_lo_minus + rng.random() * max(cdf_hi - cdf_lo_minus, 1e-12)
        return int(np.clip(stats.poisson.ppf(u_adj, mu), lo, hi))

    @staticmethod
    def _trunc_lognormal_sample(mu_ln: float, sigma: float, lo: float, hi: float, rng: np.random.Generator) -> float:
        a = (np.log(max(lo, 1e-9)) - mu_ln) / sigma
        b = (np.log(max(hi, lo + 1e-9)) - mu_ln) / sigma
        Fa, Fb = stats.norm.cdf(a), stats.norm.cdf(b)
        z = stats.norm.ppf(Fa + rng.random() * max(Fb - Fa, 1e-12))
        return float(np.exp(mu_ln + sigma * z))

    # ---------- Sequential Condition Sampling with Truncation ----------
    def sample_sequential(self, new_df: pd.DataFrame, n_samples: int = 1,
                          alpha: float = 0.05, w_floor: int = 2,
                          rng: Optional[np.random.Generator] = None,
                          wall_rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """Draw counts sequentially and then conditional internal-wall lengths.

        ``rng`` controls count draws. If supplied, ``wall_rng`` controls only the
        lognormal wall-length quantile. Reset that stream to the same seed in
        separate one-building calls to preserve a shared dimensional factor.
        Omitting it uses ``rng`` for all draws.
        """
        if rng is None: rng = np.random.default_rng(self.random_state + 1)
        if wall_rng is None: wall_rng = rng
        n = len(new_df)
        R_s = np.zeros((n, n_samples), dtype=int)
        W_s = np.zeros((n, n_samples), dtype=int)
        D_s = np.zeros((n, n_samples), dtype=int)
        L_s = np.zeros((n, n_samples), dtype=float)

        for i in range(n):
            base_row = new_df.iloc[[i]].copy()
            # R
            mu_R = float(self.predict_R_mean(base_row)[0])
            rb = self.predict_bounds_R(base_row, alpha=alpha).iloc[0]
            R_lo, R_hi = int(rb["R_min"]), int(rb["R_max"])

            for s in range(n_samples):
                R_draw = self._trunc_poisson_sample(mu_R, R_lo, R_hi, rng)
                R_s[i, s] = R_draw

                # W | x,R
                row_RW = base_row.copy()
                row_RW["R"] = R_draw
                mu_W = float(self.predict_W_mean(row_RW)[0])
                wb = self.predict_bounds_W(row_RW, alpha=alpha, w_floor=w_floor).iloc[0]
                W_draw = self._trunc_poisson_sample(mu_W, int(wb["W_min"]), int(wb["W_max"]), rng)
                if W_draw > 42:
                    W_draw = 42
                W_s[i, s] = W_draw

                # D | x,R
                mu_D = float(self.predict_D_mean(row_RW)[0])
                db = self.predict_bounds_D(row_RW, alpha=alpha).iloc[0]
                D_draw = self._trunc_poisson_sample(mu_D, int(db["D_min"]), int(db["D_max"]), rng)
                if D_draw > 32:
                    D_draw = 32
                D_s[i, s] = D_draw

                # L | x,R,D
                row_L = row_RW.copy()
                row_L["D"] = D_draw
                mu_ln = float(self._predict_mu_lnL(row_L)[0])
                lb = self.predict_bounds_L(row_L, alpha=alpha).iloc[0]
                L_draw = self._trunc_lognormal_sample(mu_ln, self.sigma_L, float(lb["L_min"]), float(lb["L_max"]), wall_rng)
                L_s[i, s] = L_draw
        return {"R": R_s, "W": W_s, "D": D_s, "L": L_s}

