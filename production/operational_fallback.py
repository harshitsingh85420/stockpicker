"""
Operational Fallback Module — Gap 17
Disaster recovery logic: use last cached model when data source fails,
skip trading day with alert rather than falling back to stale signals.
"""

import os
import json
import pickle
import logging
import smtplib
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert channels
# ---------------------------------------------------------------------------

class AlertManager:
    """Send alerts via multiple channels (console always; email/file optionally)."""

    def __init__(
        self,
        alert_log: str = "stock_picker_data/alerts.log",
        smtp_host: str = None,
        smtp_port: int = 587,
        smtp_user: str = None,
        smtp_password: str = None,
        alert_email: str = None,
    ):
        self.alert_log = Path(alert_log)
        self.alert_log.parent.mkdir(parents=True, exist_ok=True)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.alert_email = alert_email

    def send(self, subject: str, body: str, level: str = "WARNING"):
        """Send alert to all configured channels."""
        timestamp = datetime.now().isoformat()
        msg = f"[{timestamp}] [{level}] {subject}: {body}"

        # Console
        if level == "CRITICAL":
            logger.critical(msg)
        elif level == "ERROR":
            logger.error(msg)
        else:
            logger.warning(msg)

        # File log
        try:
            with open(self.alert_log, "a") as fh:
                fh.write(msg + "\n")
        except Exception as e:
            logger.debug("Alert log write failed: %s", e)

        # Email (optional)
        if self.smtp_host and self.alert_email:
            self._send_email(subject, body)

    def _send_email(self, subject: str, body: str):
        try:
            from email.mime.text import MIMEText
            msg = MIMEText(body)
            msg["Subject"] = f"[StockPicker Alert] {subject}"
            msg["From"] = self.smtp_user
            msg["To"] = self.alert_email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info("Alert email sent to %s", self.alert_email)
        except Exception as e:
            logger.debug("Email send failed: %s", e)


# ---------------------------------------------------------------------------
# Data source health check
# ---------------------------------------------------------------------------

class DataSourceHealthChecker:
    """
    Verify that today's market data is available and fresh before running signals.
    """

    def __init__(
        self,
        cache_dir: str = "stock_picker_data/cache/bse",
        max_stale_hours: int = 4,
        min_stocks_expected: int = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_stale_hours = max_stale_hours
        self.min_stocks_expected = min_stocks_expected

    def check_bhav_data(self, bhav_df: Optional[pd.DataFrame], reference_date: str) -> dict:
        """
        Validate that BhavCopy data is available, recent, and has enough stocks.

        Returns:
            dict with keys: is_healthy, issues, n_stocks, latest_date
        """
        issues = []

        if bhav_df is None or bhav_df.empty:
            return {"is_healthy": False, "issues": ["BhavCopy DataFrame is empty or None"],
                    "n_stocks": 0, "latest_date": None}

        # Check date column
        date_col = "DATE" if "DATE" in bhav_df.columns else "date"
        if date_col not in bhav_df.columns:
            issues.append("No DATE column found in BhavCopy data.")
            return {"is_healthy": False, "issues": issues, "n_stocks": 0, "latest_date": None}

        bhav_df[date_col] = pd.to_datetime(bhav_df[date_col])
        latest_date = bhav_df[date_col].max()

        # Check freshness
        ref_dt = pd.to_datetime(reference_date)
        delta_days = (ref_dt - latest_date).days
        if delta_days > 5:  # more than a week stale (allowing for holidays)
            issues.append(
                f"Data is stale: latest date {latest_date.date()} vs reference {reference_date} "
                f"({delta_days} calendar days gap)."
            )

        # Check stock count
        code_col = "SC_CODE" if "SC_CODE" in bhav_df.columns else "sc_code"
        n_stocks = bhav_df[bhav_df[date_col] == latest_date][code_col].nunique() if code_col in bhav_df.columns else 0
        if n_stocks < self.min_stocks_expected:
            issues.append(
                f"Only {n_stocks} stocks in latest date — expected ≥ {self.min_stocks_expected}."
            )

        return {
            "is_healthy": len(issues) == 0,
            "issues": issues,
            "n_stocks": n_stocks,
            "latest_date": str(latest_date.date()),
        }

    def check_cache_files(self, reference_date: str) -> dict:
        """Check if cached data files exist for the reference date."""
        if not self.cache_dir.exists():
            return {"is_healthy": False, "issues": ["Cache directory does not exist."]}

        pkl_files = list(self.cache_dir.glob("*.pkl"))
        if not pkl_files:
            return {"is_healthy": False, "issues": ["No pickle cache files found."]}

        # Check modification times
        stale_threshold = datetime.now() - timedelta(hours=self.max_stale_hours * 4)
        recent = [f for f in pkl_files if datetime.fromtimestamp(f.stat().st_mtime) > stale_threshold]

        issues = []
        if not recent:
            issues.append(
                f"All cache files are older than {self.max_stale_hours * 4}h. "
                "Data may not reflect today's market."
            )

        return {"is_healthy": len(issues) == 0, "issues": issues, "n_cache_files": len(pkl_files)}


# ---------------------------------------------------------------------------
# Model cache manager
# ---------------------------------------------------------------------------

class ModelCacheManager:
    """
    Manage the last-known-good model so we can fall back to it when the
    current day's model fails to train or load.
    """

    def __init__(self, models_dir: str = "stock_picker_data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.last_good_path = self.models_dir / "last_known_good.pkl"
        self.metadata_path = self.models_dir / "last_known_good_meta.json"

    # ------------------------------------------------------------------
    def save_good_model(self, model, metadata: dict):
        """Persist a successfully trained model as the last-known-good."""
        with open(self.last_good_path, "wb") as fh:
            pickle.dump(model, fh)
        meta = {
            "saved_at": datetime.now().isoformat(),
            "trained_on": metadata.get("trained_on", ""),
            "cv_auc": metadata.get("cv_auc", None),
            "feature_count": metadata.get("feature_count", None),
            "model_type": metadata.get("model_type", "unknown"),
            "checksum": self._file_checksum(self.last_good_path),
        }
        with open(self.metadata_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Last-known-good model saved (AUC=%s).", meta.get("cv_auc"))

    # ------------------------------------------------------------------
    def load_last_good_model(self):
        """
        Load the last-known-good model.

        Returns:
            (model, metadata) or (None, None) if no good model exists.
        """
        if not self.last_good_path.exists():
            logger.warning("No last-known-good model found at %s.", self.last_good_path)
            return None, None

        try:
            # Verify checksum
            if self.metadata_path.exists():
                with open(self.metadata_path) as fh:
                    meta = json.load(fh)
                stored_checksum = meta.get("checksum", "")
                actual_checksum = self._file_checksum(self.last_good_path)
                if stored_checksum and stored_checksum != actual_checksum:
                    logger.error("Checksum mismatch on last-known-good model — refusing to load.")
                    return None, None
            else:
                meta = {}

            with open(self.last_good_path, "rb") as fh:
                model = pickle.load(fh)

            logger.info(
                "Loaded last-known-good model (saved %s, AUC=%s).",
                meta.get("saved_at", "unknown"),
                meta.get("cv_auc", "unknown"),
            )
            return model, meta
        except Exception as e:
            logger.error("Failed to load last-known-good model: %s", e)
            return None, None

    # ------------------------------------------------------------------
    def is_model_stale(self, max_age_days: int = 7) -> bool:
        """Return True if the cached model is older than max_age_days."""
        if not self.metadata_path.exists():
            return True
        with open(self.metadata_path) as fh:
            meta = json.load(fh)
        saved_at = datetime.fromisoformat(meta.get("saved_at", "2000-01-01"))
        age = (datetime.now() - saved_at).days
        if age > max_age_days:
            logger.warning("Last-known-good model is %d days old (max=%d).", age, max_age_days)
            return True
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _file_checksum(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Main fallback orchestrator
# ---------------------------------------------------------------------------

class OperationalFallback:
    """
    High-level coordinator that decides what to do when something goes wrong:
      - Data source failure -> skip trading day with alert
      - Model failure -> load last-known-good model
      - Both fail -> emergency skip with human notification
    """

    SKIP_TODAY_FLAG = Path("stock_picker_data/SKIP_TODAY")

    def __init__(
        self,
        alert_manager: AlertManager = None,
        data_checker: DataSourceHealthChecker = None,
        model_cache: ModelCacheManager = None,
        max_model_age_days: int = 7,
    ):
        self.alert = alert_manager or AlertManager()
        self.data_checker = data_checker or DataSourceHealthChecker()
        self.model_cache = model_cache or ModelCacheManager()
        self.max_model_age_days = max_model_age_days

    # ------------------------------------------------------------------
    def handle_data_failure(self, reference_date: str, issues: list) -> dict:
        """
        Called when fresh market data cannot be obtained.

        Policy: skip the trading day entirely (never trade on stale data).

        Returns:
            dict with action='SKIP_DAY', reason, reference_date
        """
        reason = "; ".join(issues)
        self.alert.send(
            subject="Data Source Failure — Trading Day Skipped",
            body=(
                f"Date: {reference_date}\n"
                f"Issues:\n" + "\n".join(f"  • {i}" for i in issues) + "\n\n"
                "Action: No trades will be generated for today. "
                "System will retry on the next trading session."
            ),
            level="ERROR",
        )
        self._write_skip_flag(reference_date, reason)
        logger.error("SKIP_DAY: %s — %s", reference_date, reason)
        return {"action": "SKIP_DAY", "reason": reason, "reference_date": reference_date}

    # ------------------------------------------------------------------
    def handle_model_failure(self, error: Exception, reference_date: str) -> tuple:
        """
        Called when the freshly-trained model fails to load or predict.

        Policy: fall back to last-known-good model if it is not stale.

        Returns:
            (model, metadata) or (None, None) if fallback also unavailable.
        """
        self.alert.send(
            subject="Model Failure — Attempting Fallback",
            body=(
                f"Date: {reference_date}\n"
                f"Error: {error}\n\n"
                "Attempting to load last-known-good model."
            ),
            level="WARNING",
        )

        if self.model_cache.is_model_stale(self.max_model_age_days):
            self.alert.send(
                subject="Fallback Model Too Stale — Trading Day Skipped",
                body=(
                    f"Date: {reference_date}\n"
                    f"The last-known-good model is older than {self.max_model_age_days} days. "
                    "Refusing to trade with a stale model."
                ),
                level="CRITICAL",
            )
            return None, None

        model, meta = self.model_cache.load_last_good_model()
        if model is None:
            self.alert.send(
                subject="No Fallback Model Available — Trading Day Skipped",
                body=f"Date: {reference_date}\nNo last-known-good model exists.",
                level="CRITICAL",
            )
            return None, None

        self.alert.send(
            subject="Fallback Model Activated",
            body=(
                f"Date: {reference_date}\n"
                f"Using model saved on {meta.get('saved_at', 'unknown')} "
                f"(AUC={meta.get('cv_auc', 'N/A')}).\n"
                "WARNING: Signals may not reflect the latest market conditions."
            ),
            level="WARNING",
        )
        return model, meta

    # ------------------------------------------------------------------
    def pre_run_health_check(
        self, bhav_df: Optional[pd.DataFrame], reference_date: str
    ) -> dict:
        """
        Run all health checks before beginning the daily pipeline.

        Returns:
            dict with is_healthy, action ('PROCEED' / 'SKIP_DAY'), details
        """
        health = self.data_checker.check_bhav_data(bhav_df, reference_date)

        if not health["is_healthy"]:
            result = self.handle_data_failure(reference_date, health["issues"])
            return {"is_healthy": False, "action": "SKIP_DAY", "details": result}

        cache_health = self.data_checker.check_cache_files(reference_date)
        if not cache_health["is_healthy"]:
            # Cache issues are warnings, not blockers — log and continue
            self.alert.send(
                subject="Cache Health Warning",
                body="; ".join(cache_health["issues"]),
                level="WARNING",
            )

        logger.info(
            "Pre-run health check PASSED for %s (%d stocks).",
            reference_date, health["n_stocks"]
        )
        return {"is_healthy": True, "action": "PROCEED", "details": health}

    # ------------------------------------------------------------------
    def should_skip_today(self) -> bool:
        """Return True if the SKIP_TODAY flag file is present."""
        return self.SKIP_TODAY_FLAG.exists()

    def clear_skip_flag(self):
        """Remove the SKIP_TODAY flag (call at start of new session)."""
        if self.SKIP_TODAY_FLAG.exists():
            self.SKIP_TODAY_FLAG.unlink()
            logger.info("SKIP_TODAY flag cleared.")

    def _write_skip_flag(self, date: str, reason: str):
        self.SKIP_TODAY_FLAG.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SKIP_TODAY_FLAG, "w") as fh:
            json.dump({"date": date, "reason": reason, "written_at": datetime.now().isoformat()}, fh)


# ---------------------------------------------------------------------------
# P06 — Multi-source BhavCopy failover
# ---------------------------------------------------------------------------

class DataSourceFailover:
    """
    Attempt to obtain a full BhavCopy DataFrame from multiple data sources,
    trying each in priority order.  If all sources fail the caller should
    trigger SKIP_DAY.

    Priority:
      1. BSE BhavCopy  (existing DataLoader / BSEDataFetcher)
      2. NSE BhavCopy  (CSV zip download from NSE archives)
      3. yfinance      (bulk download for top Nifty 500 stocks)
      4. None          -> caller must set SKIP_DAY = True
    """

    # NSE BhavCopy archive template
    _NSE_BHAV_URL = (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES"
        "/{year}/{mon}/cm{date_str}bhav.csv.zip"
    )
    _NSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://www.nseindia.com/",
        "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(
        self,
        cache_dir: str = "stock_picker_data/cache/bse",
        lookback_days: int = 730,
        alert_manager: "AlertManager" = None,
    ):
        self.cache_dir    = Path(cache_dir)
        self.lookback_days = lookback_days
        self.alert        = alert_manager or AlertManager()

    # ------------------------------------------------------------------
    def fetch_bhav(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        Return a standardised BhavCopy DataFrame for the period ending
        on *trade_date*, trying each source in order.

        Columns guaranteed in result:
            SC_CODE, DATE, Open, High, Low, Close, Volume

        Returns None only if ALL sources fail.
        """
        sources = [
            ("BSE BhavCopy",  self._fetch_bse),
            ("NSE BhavCopy",  self._fetch_nse),
            ("yfinance",      self._fetch_yfinance),
        ]
        last_error = None
        for source_name, fn in sources:
            try:
                logger.info("DataSourceFailover: trying %s ...", source_name)
                df = fn(trade_date)
                if df is not None and not df.empty and len(df) > 100:
                    logger.info(
                        "DataSourceFailover: %s succeeded — %d rows, %d stocks.",
                        source_name, len(df), df["SC_CODE"].nunique(),
                    )
                    if source_name != "BSE BhavCopy":
                        self.alert.send(
                            subject=f"Data Failover: using {source_name}",
                            body=(
                                f"BSE BhavCopy was unavailable. "
                                f"Loaded {len(df)} rows from {source_name}.\n"
                                f"Date: {trade_date}"
                            ),
                            level="WARNING",
                        )
                    return df
                logger.warning("DataSourceFailover: %s returned empty data.", source_name)
            except Exception as exc:
                last_error = exc
                logger.warning("DataSourceFailover: %s failed — %s", source_name, exc)

        # All sources failed
        self.alert.send(
            subject="ALL data sources failed — SKIP_DAY",
            body=(
                f"Date: {trade_date}\n"
                f"BSE BhavCopy, NSE BhavCopy, and yfinance all failed.\n"
                f"Last error: {last_error}\n\n"
                "Action: SKIP_DAY.  No stale T-1 data will be used."
            ),
            level="CRITICAL",
        )
        logger.critical(
            "DataSourceFailover: all sources exhausted for %s.  SKIP_DAY.",
            trade_date,
        )
        return None

    # ------------------------------------------------------------------
    def _fetch_bse(self, trade_date: str) -> Optional[pd.DataFrame]:
        """Primary: BSE BhavCopy via existing DataLoader."""
        from production.data_loader import DataLoader
        loader = DataLoader(cache_dir=str(self.cache_dir))
        df = loader.load(lookback_days=self.lookback_days, end=trade_date)
        if df is None or df.empty:
            raise RuntimeError("DataLoader returned empty DataFrame")
        return df

    # ------------------------------------------------------------------
    def _fetch_nse(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        Fallback 1: Download NSE BhavCopy CSV zip for *trade_date* and the
        preceding lookback_days, then convert to BSE-compatible schema.
        """
        import io
        import zipfile
        import calendar
        import requests

        td  = pd.to_datetime(trade_date)
        start = td - pd.Timedelta(days=self.lookback_days + 60)

        frames = []
        current = start
        while current <= td:
            if current.weekday() >= 5:     # skip weekends
                current += pd.Timedelta(days=1)
                continue

            year    = current.strftime("%Y")
            mon     = current.strftime("%b").upper()       # e.g. JAN
            date_str = current.strftime("%d%b%Y").upper()  # e.g. 01JAN2024
            url = self._NSE_BHAV_URL.format(year=year, mon=mon, date_str=date_str)

            try:
                r = requests.get(url, headers=self._NSE_HEADERS, timeout=15)
                if not r.ok:
                    current += pd.Timedelta(days=1)
                    continue
                zf  = zipfile.ZipFile(io.BytesIO(r.content))
                csv = zf.open(zf.namelist()[0])
                day = pd.read_csv(csv)
                day.columns = day.columns.str.strip()

                # NSE columns: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, ...
                day = day[day.get("SERIES", day.get("Series", pd.Series(["EQ"]*len(day)))) == "EQ"].copy()
                day["DATE"]    = current.date()
                day["SC_CODE"] = day.get("SYMBOL", day.get("Symbol", ""))
                day["Open"]    = pd.to_numeric(day.get("OPEN",  day.get("Open",  0)), errors="coerce")
                day["High"]    = pd.to_numeric(day.get("HIGH",  day.get("High",  0)), errors="coerce")
                day["Low"]     = pd.to_numeric(day.get("LOW",   day.get("Low",   0)), errors="coerce")
                day["Close"]   = pd.to_numeric(day.get("CLOSE", day.get("Close", 0)), errors="coerce")
                day["Volume"]  = pd.to_numeric(
                    day.get("TOTTRDQTY", day.get("TotTrdQty", day.get("Volume", 0))),
                    errors="coerce",
                ).fillna(0)
                day["SC_NAME"] = day["SC_CODE"]
                frames.append(day[["SC_CODE", "SC_NAME", "DATE", "Open", "High", "Low", "Close", "Volume"]])
            except Exception:
                pass   # holiday or error — skip day

            current += pd.Timedelta(days=1)

        if not frames:
            raise RuntimeError("No NSE BhavCopy files could be downloaded")

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    def _fetch_yfinance(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        Fallback 2: yfinance for the top Nifty 500 stocks.
        Uses NSE symbol suffixes (.NS); maps back to numeric SC_CODE proxies.
        """
        import yfinance as yf

        # Representative Nifty 100 symbols (reduces download time)
        NIFTY100_NS = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
            "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ONGC.NS",
            "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS", "TECHM.NS",
            "HCLTECH.NS", "TATAMOTORS.NS", "COALINDIA.NS", "DIVISLAB.NS", "BAJAJFINSV.NS",
            "ADANIPORTS.NS", "DRREDDY.NS", "EICHERMOT.NS", "CIPLA.NS", "HEROMOTOCO.NS",
            "GRASIM.NS", "SHREECEM.NS", "BPCL.NS", "BRITANNIA.NS", "INDUSINDBK.NS",
            "TATACONSUM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "UPL.NS", "ADANIENT.NS",
            "PIDILITIND.NS", "AMBUJACEM.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "HAVELLS.NS",
        ]

        end   = pd.to_datetime(trade_date)
        start = end - pd.Timedelta(days=self.lookback_days + 30)

        frames = []
        for sym in NIFTY100_NS:
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(start=start.strftime("%Y-%m-%d"),
                                  end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                  auto_adjust=False)
                if hist.empty:
                    continue
                hist = hist.reset_index()
                hist["SC_CODE"] = sym.replace(".NS", "")
                hist["SC_NAME"] = sym.replace(".NS", "")
                hist["DATE"]    = pd.to_datetime(hist["Date"]).dt.date
                hist["Open"]    = hist["Open"]
                hist["High"]    = hist["High"]
                hist["Low"]     = hist["Low"]
                hist["Close"]   = hist["Close"]
                hist["Volume"]  = hist["Volume"]
                frames.append(hist[["SC_CODE", "SC_NAME", "DATE",
                                    "Open", "High", "Low", "Close", "Volume"]])
            except Exception:
                continue

        if not frames:
            raise RuntimeError("yfinance returned no data")

        df = pd.concat(frames, ignore_index=True)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_fallback_system(
    models_dir: str = "stock_picker_data/models",
    cache_dir: str = "stock_picker_data/cache/bse",
    alert_log: str = "stock_picker_data/alerts.log",
) -> OperationalFallback:
    """Build an OperationalFallback with default settings."""
    return OperationalFallback(
        alert_manager=AlertManager(alert_log=alert_log),
        data_checker=DataSourceHealthChecker(cache_dir=cache_dir),
        model_cache=ModelCacheManager(models_dir=models_dir),
    )


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fb = create_fallback_system()

    print("=== Simulating empty data failure ===")
    result = fb.pre_run_health_check(None, "2026-04-23")
    print(result)

    print("\n=== Simulating healthy data ===")
    import numpy as np
    dates = pd.bdate_range("2026-04-01", "2026-04-23")
    n = len(dates) * 100
    bhav = pd.DataFrame({
        "DATE": np.repeat(dates, 100),
        "SC_CODE": list(range(100)) * len(dates),
        "Close": np.random.uniform(100, 500, n),
        "Volume": np.random.randint(10000, 500000, n),
    })
    result = fb.pre_run_health_check(bhav, "2026-04-23")
    print(result)

    print("\n* OperationalFallback smoke test passed.")
