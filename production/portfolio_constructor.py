"""
portfolio_constructor.py
Gap 9:  Portfolio Construction
Gap 16: Capital Allocation Cap

Production module for constructing a filtered, risk-aware stock portfolio
from a list of signal-ranked picks. Applies sector caps, correlation filters,
position limits, and capital deployment caps before finalising the portfolio.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Sector mapping
# ---------------------------------------------------------------------------

SECTOR_MAPPING: Dict[str, List[str]] = {
    "BANKING": [
        "HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB", "BANDHAN",
        "FEDERAL", "RBL", "YES BANK", "CANARA", "UNION", "INDUSIND",
    ],
    "IT": [
        "TCS", "INFOSYS", "WIPRO", "HCL", "TECH MAHINDRA", "MPHASIS",
        "COFORGE", "PERSISTENT", "LTI",
    ],
    "PHARMA": [
        "SUN PHARMA", "CIPLA", "DR REDDY", "LUPIN", "AUROBINDO",
        "GLENMARK", "ALKEM", "BIOCON", "DIVIS",
    ],
    "AUTO": [
        "MARUTI", "TATA MOTORS", "MAHINDRA", "BAJAJ AUTO", "HERO", "EICHER",
    ],
    "FMCG": [
        "HINDUSTAN UNILEVER", "ITC", "NESTLE", "BRITANNIA", "DABUR",
        "MARICO", "COLGATE", "GODREJ CONSUMER",
    ],
    "METALS": [
        "TATA STEEL", "JSW STEEL", "SAIL", "HINDALCO", "VEDANTA",
        "NMDC", "NATIONAL ALUMINIUM",
    ],
    "ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "PETRONET", "GAIL",
        "ADANI GREEN", "TATA POWER", "POWER GRID", "NTPC",
    ],
    "REALTY": [
        "DLF", "GODREJ PROPERTIES", "OBEROI", "PRESTIGE", "BRIGADE",
    ],
    "TELECOM": [
        "BHARTI AIRTEL", "IDEA",
    ],
    "INFRA": [
        "L&T", "ADANI PORTS", "CONCOR", "IRB",
    ],
    "NBFC": [
        "BAJAJ FINANCE", "BAJAJ FINSERV", "MUTHOOT", "SHRIRAM",
        "MANAPPURAM", "CHOLA", "L&T FINANCE",
    ],
    "OTHERS": [],  # Catch-all for unclassified stocks
}


# ---------------------------------------------------------------------------
# 2. PortfolioConstructor  (Gap 9 & Gap 16)
# ---------------------------------------------------------------------------

class PortfolioConstructor:
    """
    Constructs a risk-filtered stock portfolio from a ranked picks DataFrame.

    Filters applied in order inside construct_portfolio():
      1. Minimum probability threshold
      2. Sector cap (max N stocks per sector)
      3. Correlation filter (drop highly correlated lower-ranked picks)
      4. Position limit (max total open positions)
      5. Capital deployment cap (max % of capital deployed)

    Parameters
    ----------
    max_positions : int
        Maximum total open positions across the portfolio.
    max_sector_positions : int
        Maximum picks allowed from any single sector.
    max_capital_deployed : float
        Maximum fraction of total capital that may be deployed (e.g., 0.70 = 70%).
    max_correlation : float
        Pearson correlation threshold above which a lower-ranked pick is dropped.
    min_probability : float
        Minimum signal probability for a pick to be included.
    """

    def __init__(
        self,
        max_positions: int = 15,
        max_sector_positions: int = 2,
        max_capital_deployed: float = 0.70,
        max_correlation: float = 0.70,
        min_probability: float = 0.60,
    ) -> None:
        self.max_positions = max_positions
        self.max_sector_positions = max_sector_positions
        self.max_capital_deployed = max_capital_deployed
        self.max_correlation = max_correlation
        self.min_probability = min_probability

    # ------------------------------------------------------------------
    # Sector classification
    # ------------------------------------------------------------------

    def classify_sector(self, sc_name: str) -> str:
        """
        Classify a stock into a sector using SECTOR_MAPPING keywords.

        The check is case-insensitive and looks for substring matches.
        If no keyword matches, returns 'OTHERS'.

        Parameters
        ----------
        sc_name : str
            Stock name (SC_NAME field from BSE/NSE bhav copy or master file).

        Returns
        -------
        str : sector name
        """
        if not sc_name:
            return "OTHERS"
        name_upper = sc_name.upper()
        for sector, keywords in SECTOR_MAPPING.items():
            if sector == "OTHERS":
                continue
            for kw in keywords:
                if kw.upper() in name_upper:
                    return sector
        return "OTHERS"

    # ------------------------------------------------------------------
    # Filter: minimum probability
    # ------------------------------------------------------------------

    def _filter_by_probability(self, picks_df: pd.DataFrame) -> pd.DataFrame:
        """Drop picks below the minimum probability threshold."""
        if "probability" not in picks_df.columns:
            return picks_df
        return picks_df[picks_df["probability"] >= self.min_probability].copy()

    # ------------------------------------------------------------------
    # Filter: sector cap
    # ------------------------------------------------------------------

    def apply_sector_cap(
        self,
        picks_df: pd.DataFrame,
        current_positions_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Limit picks to max_sector_positions per sector, accounting for
        any already-open positions in that sector.

        Parameters
        ----------
        picks_df : pd.DataFrame
            New picks. Must contain 'sc_name' and optionally 'probability'
            and 'sector' columns. Picks are assumed to be sorted by
            descending probability/rank (best picks first).
        current_positions_df : pd.DataFrame, optional
            Currently open positions. Must contain 'sc_name' column.

        Returns
        -------
        pd.DataFrame : picks filtered to sector cap.
        """
        df = picks_df.copy()

        # Ensure sector column exists
        if "sector" not in df.columns:
            df["sector"] = df["sc_name"].apply(self.classify_sector)

        # Count already-open positions per sector
        sector_counts: Dict[str, int] = {}
        if current_positions_df is not None and not current_positions_df.empty:
            if "sector" not in current_positions_df.columns:
                current_positions_df = current_positions_df.copy()
                current_positions_df["sector"] = current_positions_df["sc_name"].apply(
                    self.classify_sector
                )
            for _, row in current_positions_df.iterrows():
                s = row.get("sector", "OTHERS")
                sector_counts[s] = sector_counts.get(s, 0) + 1

        # Walk through picks in rank order; accept up to the cap per sector
        remaining_capacity: Dict[str, int] = {}
        for sector in SECTOR_MAPPING:
            already_open = sector_counts.get(sector, 0)
            remaining_capacity[sector] = max(
                0, self.max_sector_positions - already_open
            )

        keep_indices = []
        for idx, row in df.iterrows():
            sector = row.get("sector", "OTHERS")
            cap = remaining_capacity.get(sector, self.max_sector_positions)
            if cap > 0:
                keep_indices.append(idx)
                remaining_capacity[sector] = cap - 1

        return df.loc[keep_indices].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Filter: correlation
    # ------------------------------------------------------------------

    def compute_correlation_matrix(
        self,
        bhav_df: pd.DataFrame,
        sc_codes: List[str],
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """
        Compute pairwise Pearson correlation matrix of daily returns.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Historical price data. Must contain columns:
            'SC_CODE' (or 'sc_code'), 'TRDDT' (or 'date'), 'CLOSE' (or 'close').
        sc_codes : list of str
            List of stock codes to include.
        lookback_days : int
            Number of trading days of history to use.

        Returns
        -------
        pd.DataFrame : correlation matrix indexed and columned by sc_code.
        """
        # Normalise column names
        col_map = {c.upper(): c for c in bhav_df.columns}
        code_col = col_map.get("SC_CODE", col_map.get("SCCODE", None))
        date_col = col_map.get("TRDDT", col_map.get("DATE", None))
        close_col = col_map.get("CLOSE", None)

        if any(c is None for c in (code_col, date_col, close_col)):
            warnings.warn(
                "bhav_df must have SC_CODE/TRDDT/CLOSE columns. "
                "Returning empty correlation matrix."
            )
            return pd.DataFrame(index=sc_codes, columns=sc_codes, dtype=float)

        df = bhav_df[bhav_df[code_col].isin(sc_codes)].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Keep only the last `lookback_days` trading dates
        recent_dates = df[date_col].unique()
        if len(recent_dates) > lookback_days:
            cutoff = sorted(recent_dates)[-lookback_days]
            df = df[df[date_col] >= cutoff]

        pivot = df.pivot_table(index=date_col, columns=code_col, values=close_col)
        returns = pivot.pct_change().dropna(how="all")

        if returns.empty:
            return pd.DataFrame(index=sc_codes, columns=sc_codes, dtype=float)

        corr_matrix = returns.corr()
        # Reindex to ensure all requested codes appear
        corr_matrix = corr_matrix.reindex(index=sc_codes, columns=sc_codes)
        return corr_matrix

    def filter_by_correlation(
        self,
        picks_df: pd.DataFrame,
        bhav_df: pd.DataFrame,
        existing_positions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Remove lower-ranked picks that are highly correlated with an
        already-selected pick or existing position.

        Greedy forward selection: iterate picks in rank order; accept a pick
        only if it is not correlated above max_correlation with any
        already-accepted pick or existing position.

        Parameters
        ----------
        picks_df : pd.DataFrame
            New picks sorted by descending rank/probability.
            Must contain 'sc_code' column.
        bhav_df : pd.DataFrame
            Historical bhav data (see compute_correlation_matrix).
        existing_positions : list of str, optional
            SC_CODEs of currently open positions (treated as anchors).

        Returns
        -------
        pd.DataFrame : de-correlated picks.
        """
        if picks_df.empty:
            return picks_df

        code_col = "sc_code" if "sc_code" in picks_df.columns else "SC_CODE"
        all_codes = picks_df[code_col].tolist()
        if existing_positions:
            all_codes = list(set(all_codes) | set(existing_positions))

        corr_matrix = self.compute_correlation_matrix(bhav_df, all_codes)

        # Anchors are existing positions (always kept)
        anchors: List[str] = list(existing_positions) if existing_positions else []
        accepted: List[str] = []

        for _, row in picks_df.iterrows():
            code = row[code_col]
            conflict = False
            for anchor in anchors + accepted:
                try:
                    c = corr_matrix.at[code, anchor]
                    if pd.notna(c) and abs(c) >= self.max_correlation:
                        conflict = True
                        break
                except KeyError:
                    pass
            if not conflict:
                accepted.append(code)

        return picks_df[picks_df[code_col].isin(accepted)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Filter: position limit
    # ------------------------------------------------------------------

    def apply_position_limit(
        self,
        picks_df: pd.DataFrame,
        current_open_positions: int = 0,
    ) -> pd.DataFrame:
        """
        Trim picks so that total open positions do not exceed max_positions.

        Parameters
        ----------
        picks_df : pd.DataFrame
            Filtered picks (sorted by descending rank).
        current_open_positions : int
            Number of positions currently open.

        Returns
        -------
        pd.DataFrame : at most (max_positions - current_open_positions) picks.
        """
        slots_available = max(0, self.max_positions - current_open_positions)
        return picks_df.head(slots_available).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Filter: capital cap  (Gap 16)
    # ------------------------------------------------------------------

    def apply_capital_cap(
        self,
        picks_df: pd.DataFrame,
        current_deployed_capital: float,
        total_capital: float,
    ) -> pd.DataFrame:
        """
        Filter picks so that total capital deployed stays within
        max_capital_deployed * total_capital.

        Each pick is assumed to consume 'position_size' capital if that
        column is present; otherwise an equal split of available slack
        is assumed.

        Parameters
        ----------
        picks_df : pd.DataFrame
            Picks, optionally with a 'position_size' column (Rs.).
        current_deployed_capital : float
            Capital already deployed in open positions (Rs.).
        total_capital : float
            Total trading capital (Rs.).

        Returns
        -------
        pd.DataFrame : picks that can be added within the capital cap.
        """
        max_deployable = self.max_capital_deployed * total_capital
        slack = max_deployable - current_deployed_capital

        if slack <= 0:
            return picks_df.iloc[0:0].reset_index(drop=True)  # empty

        if "position_size" not in picks_df.columns:
            # Assume equal allocation across remaining picks
            num_picks = len(picks_df)
            if num_picks == 0:
                return picks_df
            per_pick = slack / num_picks
            picks_df = picks_df.copy()
            picks_df["position_size"] = per_pick

        keep_indices = []
        running_capital = 0.0
        for idx, row in picks_df.iterrows():
            size = row["position_size"]
            if running_capital + size <= slack + 1e-6:  # small float tolerance
                keep_indices.append(idx)
                running_capital += size

        return picks_df.loc[keep_indices].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Master pipeline
    # ------------------------------------------------------------------

    def construct_portfolio(
        self,
        picks_df: pd.DataFrame,
        bhav_df: pd.DataFrame,
        current_positions_df: Optional[pd.DataFrame] = None,
        current_deployed_capital: float = 0.0,
        total_capital: float = 1_000_000,
    ) -> pd.DataFrame:
        """
        Run the full portfolio construction pipeline.

        Pipeline stages
        ---------------
        1. Minimum probability filter
        2. Sector classification (adds 'sector' column if missing)
        3. Sector cap filter
        4. Correlation filter
        5. Position limit filter
        6. Capital deployment cap filter

        Parameters
        ----------
        picks_df : pd.DataFrame
            Signal picks. Expected columns: 'sc_code', 'sc_name',
            'probability' (optional), 'position_size' (optional).
            Should be sorted by descending signal strength.
        bhav_df : pd.DataFrame
            Historical bhav data for correlation computation.
        current_positions_df : pd.DataFrame, optional
            Currently open positions (sc_code, sc_name columns).
        current_deployed_capital : float
            Already-deployed capital in Rs..
        total_capital : float
            Total available trading capital in Rs..

        Returns
        -------
        pd.DataFrame : final accepted picks ready for order entry.
        """
        df = picks_df.copy()

        # Ensure sector column
        if "sector" not in df.columns:
            df["sector"] = df["sc_name"].apply(self.classify_sector)

        # --- Stage 1: probability filter ---
        df = self._filter_by_probability(df)
        if df.empty:
            return df

        # --- Stage 2: sector cap ---
        df = self.apply_sector_cap(df, current_positions_df)
        if df.empty:
            return df

        # --- Stage 3: correlation filter ---
        existing_codes: Optional[List[str]] = None
        code_col = "sc_code" if "sc_code" in df.columns else "SC_CODE"
        if current_positions_df is not None and not current_positions_df.empty:
            pos_code_col = (
                "sc_code" if "sc_code" in current_positions_df.columns else "SC_CODE"
            )
            if pos_code_col in current_positions_df.columns:
                existing_codes = current_positions_df[pos_code_col].tolist()

        df = self.filter_by_correlation(df, bhav_df, existing_codes)
        if df.empty:
            return df

        # --- Stage 4: position limit ---
        current_open = (
            len(current_positions_df) if current_positions_df is not None else 0
        )
        df = self.apply_position_limit(df, current_open)
        if df.empty:
            return df

        # --- Stage 5: capital cap ---
        df = self.apply_capital_cap(df, current_deployed_capital, total_capital)
        return df

    # ------------------------------------------------------------------
    # Portfolio statistics
    # ------------------------------------------------------------------

    def get_portfolio_stats(
        self,
        portfolio_df: pd.DataFrame,
        bhav_df: pd.DataFrame,
    ) -> dict:
        """
        Compute summary statistics for the constructed portfolio.

        Returns
        -------
        dict with keys:
            sector_distribution  - {sector: count}
            correlation_stats    - {mean, max, min} pairwise correlations
            total_capital_deployed - Rs. (sum of position_size if available)
            num_positions        - int
        """
        if portfolio_df.empty:
            return {
                "sector_distribution": {},
                "correlation_stats": {"mean": None, "max": None, "min": None},
                "total_capital_deployed": 0.0,
                "num_positions": 0,
            }

        # Sector distribution
        if "sector" not in portfolio_df.columns:
            portfolio_df = portfolio_df.copy()
            portfolio_df["sector"] = portfolio_df["sc_name"].apply(self.classify_sector)

        sector_dist = portfolio_df["sector"].value_counts().to_dict()

        # Capital deployed
        total_deployed = (
            portfolio_df["position_size"].sum()
            if "position_size" in portfolio_df.columns
            else 0.0
        )

        # Correlation stats
        code_col = "sc_code" if "sc_code" in portfolio_df.columns else "SC_CODE"
        sc_codes = portfolio_df[code_col].tolist()
        corr_stats: Dict[str, Optional[float]] = {"mean": None, "max": None, "min": None}
        if len(sc_codes) >= 2:
            corr_matrix = self.compute_correlation_matrix(bhav_df, sc_codes)
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
            upper = corr_matrix.values[mask]
            valid = upper[~np.isnan(upper)]
            if len(valid) > 0:
                corr_stats = {
                    "mean": float(np.mean(valid)),
                    "max": float(np.max(valid)),
                    "min": float(np.min(valid)),
                }

        return {
            "sector_distribution": sector_dist,
            "correlation_stats": corr_stats,
            "total_capital_deployed": float(total_deployed),
            "num_positions": len(portfolio_df),
        }


# ---------------------------------------------------------------------------
# 3. CorrelationMonitor
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """
    Standalone correlation analysis tool for a stock universe.

    Used for monitoring portfolio risk concentration and identifying
    potentially over-correlated baskets.
    """

    def __init__(self, bhav_df: pd.DataFrame, lookback_days: int = 60) -> None:
        """
        Parameters
        ----------
        bhav_df : pd.DataFrame
            Historical price data with columns SC_CODE, TRDDT, CLOSE.
        lookback_days : int
            Number of trading days to use for return computation.
        """
        self.bhav_df = bhav_df
        self.lookback_days = lookback_days
        self._constructor = PortfolioConstructor()

    # ------------------------------------------------------------------
    # Return matrix
    # ------------------------------------------------------------------

    def compute_returns_matrix(
        self,
        sc_codes: List[str],
        reference_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a DataFrame of daily returns per stock.

        Parameters
        ----------
        sc_codes : list of str
            Stock codes to include.
        reference_date : str, optional
            ISO date string. If provided, only data up to this date is used.

        Returns
        -------
        pd.DataFrame : rows = dates, columns = sc_codes, values = daily returns.
        """
        col_map = {c.upper(): c for c in self.bhav_df.columns}
        code_col = col_map.get("SC_CODE", col_map.get("SCCODE", None))
        date_col = col_map.get("TRDDT", col_map.get("DATE", None))
        close_col = col_map.get("CLOSE", None)

        if any(c is None for c in (code_col, date_col, close_col)):
            warnings.warn("bhav_df must have SC_CODE/TRDDT/CLOSE columns.")
            return pd.DataFrame()

        df = self.bhav_df[self.bhav_df[code_col].isin(sc_codes)].copy()
        df[date_col] = pd.to_datetime(df[date_col])

        if reference_date:
            df = df[df[date_col] <= pd.to_datetime(reference_date)]

        df = df.sort_values(date_col)

        # Use the last lookback_days distinct dates
        unique_dates = sorted(df[date_col].unique())
        if len(unique_dates) > self.lookback_days:
            cutoff = unique_dates[-self.lookback_days]
            df = df[df[date_col] >= cutoff]

        pivot = df.pivot_table(index=date_col, columns=code_col, values=close_col)
        returns = pivot.pct_change().dropna(how="all")
        return returns

    # ------------------------------------------------------------------
    # Correlation heatmap data
    # ------------------------------------------------------------------

    def get_correlation_heatmap_data(self, sc_codes: List[str]) -> pd.DataFrame:
        """
        Return the full pairwise correlation matrix for a list of stock codes.

        Parameters
        ----------
        sc_codes : list of str

        Returns
        -------
        pd.DataFrame : symmetric correlation matrix.
        """
        returns = self.compute_returns_matrix(sc_codes)
        if returns.empty:
            return pd.DataFrame(index=sc_codes, columns=sc_codes, dtype=float)
        return returns.corr()

    # ------------------------------------------------------------------
    # Highly correlated pairs
    # ------------------------------------------------------------------

    def find_highly_correlated_pairs(
        self,
        sc_codes: List[str],
        threshold: float = 0.70,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of stocks with correlation above the threshold.

        Parameters
        ----------
        sc_codes : list of str
        threshold : float
            Correlation threshold (absolute value).

        Returns
        -------
        list of (sc_code1, sc_code2, correlation) tuples, sorted by
        descending absolute correlation.
        """
        corr_matrix = self.get_correlation_heatmap_data(sc_codes)
        if corr_matrix.empty:
            return []

        pairs: List[Tuple[str, str, float]] = []
        codes = list(corr_matrix.columns)
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                c = corr_matrix.iloc[i, j]
                if pd.notna(c) and abs(c) >= threshold:
                    pairs.append((codes[i], codes[j], round(float(c), 4)))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import random

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    INR = "Rs."  # ASCII-safe rupee label for console output

    print("=" * 65)
    print("PORTFOLIO CONSTRUCTOR DEMO")
    print("=" * 65)

    # --- Synthetic picks: 20 stocks across various sectors ---
    random.seed(42)
    np.random.seed(42)

    SYNTHETIC_STOCKS = [
        ("500180", "HDFC BANK LTD"),
        ("532174", "ICICI BANK LTD"),
        ("500112", "SBI LTD"),
        ("532215", "AXIS BANK LTD"),
        ("500696", "HDFC LTD"),          # Will map to BANKING via HDFC
        ("532540", "TCS LTD"),
        ("500209", "INFOSYS LTD"),
        ("507685", "WIPRO LTD"),
        ("532281", "HCL TECHNOLOGIES"),
        ("500124", "DR REDDY LABS"),
        ("500087", "CIPLA LTD"),
        ("524715", "SUN PHARMA"),
        ("532488", "MARUTI SUZUKI"),
        ("500570", "TATA MOTORS LTD"),
        ("500182", "HERO MOTOCORP"),
        ("500696", "RELIANCE INDUSTRIES"),
        ("500312", "ONGC LTD"),
        ("500400", "TATA STEEL LTD"),
        ("500440", "HINDALCO LTD"),
        ("532977", "BAJAJ FINANCE LTD"),
    ]

    picks_data = []
    for sc_code, sc_name in SYNTHETIC_STOCKS:
        picks_data.append({
            "sc_code": sc_code,
            "sc_name": sc_name,
            "probability": round(random.uniform(0.55, 0.92), 4),
            "position_size": round(random.uniform(40_000, 80_000), 0),
        })

    picks_df = pd.DataFrame(picks_data)
    picks_df = picks_df.sort_values("probability", ascending=False).reset_index(drop=True)

    print(f"\nInput: {len(picks_df)} picks (sorted by probability)")
    print(picks_df[["sc_code", "sc_name", "probability", "position_size"]].to_string(index=False))

    # --- Classify sectors ---
    constructor = PortfolioConstructor(
        max_positions=10,
        max_sector_positions=2,
        max_capital_deployed=0.70,
        max_correlation=0.70,
        min_probability=0.60,
    )

    picks_df["sector"] = picks_df["sc_name"].apply(constructor.classify_sector)

    print("\n--- Sector classification ---")
    print(
        picks_df[["sc_name", "sector"]].to_string(index=False)
    )

    # --- Synthetic bhav data for correlation (random walk prices) ---
    dates = pd.date_range("2025-10-01", periods=80, freq="B")
    bhav_records = []
    for sc_code, sc_name in SYNTHETIC_STOCKS:
        price = random.uniform(200, 3000)
        for d in dates:
            pct_chg = np.random.normal(0, 0.015)
            price = max(10, price * (1 + pct_chg))
            bhav_records.append({
                "SC_CODE": sc_code,
                "TRDDT": d,
                "CLOSE": round(price, 2),
            })

    bhav_df = pd.DataFrame(bhav_records)

    # --- Run full portfolio construction ---
    total_capital = 1_000_000  # Rs.10 lakh
    portfolio = constructor.construct_portfolio(
        picks_df=picks_df,
        bhav_df=bhav_df,
        current_positions_df=None,
        current_deployed_capital=0.0,
        total_capital=total_capital,
    )

    print(f"\n--- Final portfolio ({len(portfolio)} positions) ---")
    if not portfolio.empty:
        display_cols = [c for c in ["sc_code", "sc_name", "sector", "probability", "position_size"] if c in portfolio.columns]
        print(portfolio[display_cols].to_string(index=False))

    # --- Portfolio statistics ---
    stats = constructor.get_portfolio_stats(portfolio, bhav_df)
    print("\n--- Portfolio statistics ---")
    print(f"  Number of positions    : {stats['num_positions']}")
    print(f"  Total capital deployed : {INR}{stats['total_capital_deployed']:,.0f}  "
          f"({stats['total_capital_deployed']/total_capital*100:.1f}% of {INR}{total_capital:,.0f})")
    print(f"  Capital deployment cap : {constructor.max_capital_deployed*100:.0f}%")
    print("\n  Sector distribution:")
    for sector, count in sorted(stats["sector_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {sector:20s}: {count} position(s)")
    cstats = stats["correlation_stats"]
    if cstats["mean"] is not None:
        print(f"\n  Correlation (pairwise):")
        print(f"    Mean : {cstats['mean']:.4f}")
        print(f"    Max  : {cstats['max']:.4f}")
        print(f"    Min  : {cstats['min']:.4f}")

    # --- Correlation monitor ---
    print("\n--- Correlation Monitor: highly correlated pairs (threshold=0.70) ---")
    monitor = CorrelationMonitor(bhav_df, lookback_days=60)
    all_codes = picks_df["sc_code"].tolist()
    pairs = monitor.find_highly_correlated_pairs(all_codes, threshold=0.70)
    if pairs:
        for code1, code2, corr in pairs[:10]:
            print(f"  {code1} <-> {code2}  :  {corr:.4f}")
    else:
        print("  No pairs above the threshold (expected with random synthetic data).")

    print("\nDemo complete.")
