"""
ETF Portfolio Backtest Engine
순수 계산 로직만 담당 — UI 의존성 없음
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252
REBALANCE_FREQ = {
    "None": None,
    "Monthly": "ME",
    "Quarterly": "QE",
    "Semi-Annual": "2QE",
    "Annual": "YE",
}


@dataclass
class BacktestResult:
    prices: pd.DataFrame                   # 각 ETF 종가 (정렬됨)
    returns: pd.DataFrame                  # 각 ETF 일별 수익률
    portfolio_returns: pd.Series           # 포트폴리오 일별 수익률
    portfolio_equity: pd.Series            # $ 기준 누적 자산 곡선
    benchmark_equity: Optional[pd.Series]  # 벤치마크 자산 곡선
    asset_equity: pd.DataFrame             # 각 ETF 개별 자산 곡선
    weights: dict[str, float]              # 목표 비중
    initial_capital: float
    rebalance_freq: Optional[str]
    tickers: list[str] = field(default_factory=list)
    benchmark_ticker: Optional[str] = None


def download_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """yfinance로 종가 다운로드. auto_adjust=True라 Close가 이미 수정주가."""
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0).copy()
    else:
        # 단일 티커인 경우
        col = "Close" if "Close" in data.columns else data.columns[0]
        prices = data[[col]].copy()
        prices.columns = [tickers[0]]

    # 티커 순서 맞추고 결측 드롭
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available].dropna(how="all")
    return prices


def _resolve_rebalance_dates(index: pd.DatetimeIndex, freq_label: str) -> set:
    """리밸런싱 일자(각 기간의 마지막 거래일) 집합 반환."""
    freq = REBALANCE_FREQ.get(freq_label)
    if freq is None or len(index) == 0:
        return set()

    # 각 기간의 마지막 거래일을 구한다
    series = pd.Series(index=index, data=np.arange(len(index)))
    grouped = series.groupby(pd.Grouper(freq=freq)).tail(1)
    # 첫 거래일은 리밸런싱 불필요 (초기 비중으로 시작)
    dates = set(grouped.index.tolist())
    if len(index) > 0:
        dates.discard(index[0])
    return dates


def run_backtest(
    tickers: list[str],
    weights: list[float],
    start,
    end,
    initial_capital: float = 10_000.0,
    rebalance_freq: str = "Quarterly",
    benchmark: Optional[str] = "SPY",
) -> BacktestResult:
    """
    포트폴리오 백테스트 실행.

    weights 는 합이 1.0(또는 100)이 되는 리스트. 자동으로 정규화함.
    """
    if len(tickers) != len(weights):
        raise ValueError("tickers와 weights의 길이가 일치해야 합니다.")
    if not tickers:
        raise ValueError("최소 1개 이상의 티커가 필요합니다.")

    # 정규화
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("비중 합계가 0보다 커야 합니다.")
    w = w / w.sum()

    # 벤치마크까지 한 번에 받아서 같은 거래일 축으로 정렬
    all_tickers = list(dict.fromkeys(tickers + ([benchmark] if benchmark else [])))
    all_prices = download_prices(all_tickers, start, end)

    if all_prices.empty:
        raise ValueError("가격 데이터를 불러오지 못했습니다. 티커/기간을 확인하세요.")

    # 구성 종목은 모두 존재해야 함 (없으면 에러)
    missing = [t for t in tickers if t not in all_prices.columns]
    if missing:
        raise ValueError(f"다음 티커의 데이터를 불러올 수 없습니다: {missing}")

    prices = all_prices[tickers].dropna()
    if prices.empty:
        raise ValueError("공통 거래일이 없습니다. 기간을 더 넓게 설정하세요.")

    returns = prices.pct_change().fillna(0.0)

    # --- 포트폴리오 시뮬레이션 ---
    # 각 ETF를 "주식 수"로 들고 있다고 가정: 리밸런싱 시점에만 비중을 재조정
    rebal_dates = _resolve_rebalance_dates(prices.index, rebalance_freq)

    holdings = np.zeros(len(tickers))  # 각 종목의 보유 주수
    equity_curve = []
    first = True
    for dt, row in prices.iterrows():
        px = row.values
        if first:
            # 초기 매수
            alloc = initial_capital * w
            holdings = np.where(px > 0, alloc / px, 0.0)
            first = False
            equity_curve.append(initial_capital)
            continue

        equity = float(np.dot(holdings, px))
        if dt in rebal_dates and equity > 0:
            alloc = equity * w
            holdings = np.where(px > 0, alloc / px, 0.0)
            equity = float(np.dot(holdings, px))
        equity_curve.append(equity)

    portfolio_equity = pd.Series(equity_curve, index=prices.index, name="Portfolio")
    portfolio_returns = portfolio_equity.pct_change().fillna(0.0)

    # 개별 ETF의 $10,000 가정 성장 곡선
    asset_equity = prices.div(prices.iloc[0]).mul(initial_capital)

    # 벤치마크
    benchmark_equity = None
    if benchmark and benchmark in all_prices.columns:
        bench_px = all_prices[benchmark].reindex(prices.index).dropna()
        if not bench_px.empty:
            benchmark_equity = (bench_px / bench_px.iloc[0]) * initial_capital
            benchmark_equity.name = benchmark

    return BacktestResult(
        prices=prices,
        returns=returns,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        benchmark_equity=benchmark_equity,
        asset_equity=asset_equity,
        weights={t: float(wi) for t, wi in zip(tickers, w)},
        initial_capital=initial_capital,
        rebalance_freq=rebalance_freq,
        tickers=tickers,
        benchmark_ticker=benchmark,
    )


# ---------------------------------------------------------------------------
# 성과 지표
# ---------------------------------------------------------------------------

def cagr(equity: pd.Series) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def total_return(equity: pd.Series) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    return equity.iloc[-1] / equity.iloc[0] - 1


def volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    vol = volatility(daily_returns)
    if vol == 0:
        return 0.0
    ann_ret = daily_returns.mean() * TRADING_DAYS
    return (ann_ret - rf) / vol


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    downside = daily_returns[daily_returns < 0]
    dd_std = downside.std() * np.sqrt(TRADING_DAYS)
    if dd_std == 0 or np.isnan(dd_std):
        return 0.0
    ann_ret = daily_returns.mean() * TRADING_DAYS
    return (ann_ret - rf) / dd_std


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return equity / roll_max - 1


def beta(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return np.nan
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    return float(cov / var)


def performance_table(result: BacktestResult, rf: float = 0.0) -> pd.DataFrame:
    """각 ETF + 포트폴리오 + 벤치마크의 주요 지표 테이블."""
    rows = []

    # 벤치마크 수익률 (베타 계산용)
    bench_returns = None
    if result.benchmark_equity is not None:
        bench_returns = result.benchmark_equity.pct_change().fillna(0.0)

    # 개별 ETF
    for t in result.tickers:
        eq = result.asset_equity[t]
        r = result.returns[t]
        rows.append({
            "Ticker": t,
            "Total Return": total_return(eq),
            "CAGR": cagr(eq),
            "Volatility": volatility(r),
            "Sharpe": sharpe_ratio(r, rf),
            "Sortino": sortino_ratio(r, rf),
            "Max Drawdown": max_drawdown(eq),
            "Beta": beta(r, bench_returns) if bench_returns is not None else np.nan,
        })

    # 포트폴리오
    rows.append({
        "Ticker": "Portfolio",
        "Total Return": total_return(result.portfolio_equity),
        "CAGR": cagr(result.portfolio_equity),
        "Volatility": volatility(result.portfolio_returns),
        "Sharpe": sharpe_ratio(result.portfolio_returns, rf),
        "Sortino": sortino_ratio(result.portfolio_returns, rf),
        "Max Drawdown": max_drawdown(result.portfolio_equity),
        "Beta": beta(result.portfolio_returns, bench_returns) if bench_returns is not None else np.nan,
    })

    # 벤치마크
    if result.benchmark_equity is not None and bench_returns is not None:
        rows.append({
            "Ticker": result.benchmark_ticker,
            "Total Return": total_return(result.benchmark_equity),
            "CAGR": cagr(result.benchmark_equity),
            "Volatility": volatility(bench_returns),
            "Sharpe": sharpe_ratio(bench_returns, rf),
            "Sortino": sortino_ratio(bench_returns, rf),
            "Max Drawdown": max_drawdown(result.benchmark_equity),
            "Beta": 1.0,
        })

    return pd.DataFrame(rows).set_index("Ticker")


def yearly_returns_table(result: BacktestResult) -> pd.DataFrame:
    """각 자산 + 포트폴리오 + 벤치마크의 연도별 수익률 매트릭스."""
    frames = {}
    for t in result.tickers:
        eq = result.asset_equity[t]
        frames[t] = eq.resample("YE").last().pct_change()
        # 첫 해는 시작가 대비
        first_year = eq.resample("YE").last().index[0]
        frames[t].loc[first_year] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1

    pe = result.portfolio_equity
    yr = pe.resample("YE").last().pct_change()
    first_year = pe.resample("YE").last().index[0]
    yr.loc[first_year] = pe.resample("YE").last().iloc[0] / pe.iloc[0] - 1
    frames["Portfolio"] = yr

    if result.benchmark_equity is not None:
        be = result.benchmark_equity
        br = be.resample("YE").last().pct_change()
        first_year = be.resample("YE").last().index[0]
        br.loc[first_year] = be.resample("YE").last().iloc[0] / be.iloc[0] - 1
        frames[result.benchmark_ticker] = br

    df = pd.DataFrame(frames)
    df.index = df.index.year
    df.index.name = "Year"
    return df.sort_index()


def monthly_returns_matrix(equity: pd.Series) -> pd.DataFrame:
    """포트폴리오의 월별 수익률 (행: 연도, 열: 월)."""
    monthly = equity.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    return df.pivot(index="Year", columns="Month", values="Return").sort_index()


def correlation_matrix(result: BacktestResult) -> pd.DataFrame:
    return result.returns.corr()


# ---------------------------------------------------------------------------
# 배당
# ---------------------------------------------------------------------------

def fetch_dividends(tickers: list[str], start, end) -> pd.DataFrame:
    """각 티커별 배당 이력. 컬럼=티커, 인덱스=지급일."""
    out = {}
    for t in tickers:
        try:
            div = yf.Ticker(t).dividends
            if div is None or div.empty:
                continue
            # 타임존 제거 후 기간 필터
            if div.index.tz is not None:
                div.index = div.index.tz_localize(None)
            div = div[(div.index >= pd.Timestamp(start)) & (div.index <= pd.Timestamp(end))]
            if not div.empty:
                out[t] = div
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def annual_dividends(div_df: pd.DataFrame) -> pd.DataFrame:
    if div_df.empty:
        return div_df
    annual = div_df.resample("YE").sum()
    annual.index = annual.index.year
    annual.index.name = "Year"
    return annual


# ---------------------------------------------------------------------------
# 몬테카를로
# ---------------------------------------------------------------------------

def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_years: int = 10,
    n_sims: int = 1000,
    initial: float = 10_000.0,
    seed: int = 42,
) -> np.ndarray:
    """
    과거 일별 수익률에서 복원추출 부트스트랩하여 (days+1, n_sims) 자산경로 반환.
    index 0 은 모두 initial.
    """
    hist = daily_returns.dropna().values
    if len(hist) == 0:
        raise ValueError("부트스트랩할 히스토리 수익률이 없습니다.")

    rng = np.random.default_rng(seed)
    n_days = int(TRADING_DAYS * n_years)
    idx = rng.integers(0, len(hist), size=(n_days, n_sims))
    sampled = hist[idx]
    growth = np.cumprod(1 + sampled, axis=0)
    paths = initial * growth
    # index 0 = 시작 자산
    paths = np.vstack([np.full((1, n_sims), initial), paths])
    return paths


def monte_carlo_percentiles(paths: np.ndarray) -> pd.DataFrame:
    """시점별 5/25/50/75/95 퍼센타일."""
    pcts = [5, 25, 50, 75, 95]
    data = {f"p{p}": np.percentile(paths, p, axis=1) for p in pcts}
    df = pd.DataFrame(data)
    df.index.name = "Day"
    return df


def monte_carlo_summary(paths: np.ndarray, initial: float) -> dict:
    final = paths[-1]
    return {
        "median": float(np.median(final)),
        "mean": float(np.mean(final)),
        "p5": float(np.percentile(final, 5)),
        "p25": float(np.percentile(final, 25)),
        "p75": float(np.percentile(final, 75)),
        "p95": float(np.percentile(final, 95)),
        "prob_loss": float(np.mean(final < initial)),
        "prob_double": float(np.mean(final >= 2 * initial)),
    }


# ---------------------------------------------------------------------------
# 레버리지 ETF 분석
# ---------------------------------------------------------------------------

def volatility_drag_analysis(
    lev_ticker: str,
    base_ticker: str,
    leverage: float,
    start,
    end,
    initial: float = 10_000.0,
) -> pd.DataFrame:
    """
    레버리지 ETF 실제 수익 vs '기초 × leverage' 이론 수익 비교.

    반환 컬럼: base, theoretical, actual, drag_pct
    - drag_pct = (theoretical - actual) / theoretical * 100  (누적 기준)
    """
    px = download_prices([lev_ticker, base_ticker], start, end).dropna()
    if px.empty or lev_ticker not in px.columns or base_ticker not in px.columns:
        raise ValueError(f"{lev_ticker} 또는 {base_ticker} 데이터를 불러올 수 없습니다.")

    lev_r = px[lev_ticker].pct_change().fillna(0.0)
    base_r = px[base_ticker].pct_change().fillna(0.0)
    theo_r = leverage * base_r

    base_eq = initial * (1 + base_r).cumprod()
    theo_eq = initial * (1 + theo_r).cumprod()
    actual_eq = initial * (1 + lev_r).cumprod()

    drag_pct = (theo_eq - actual_eq) / theo_eq * 100

    out = pd.DataFrame({
        "base": base_eq,
        "theoretical": theo_eq,
        "actual": actual_eq,
        "drag_pct": drag_pct,
    })
    out.attrs["lev_ticker"] = lev_ticker
    out.attrs["base_ticker"] = base_ticker
    out.attrs["leverage"] = leverage
    return out


def rolling_correlations(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    모든 ETF 페어의 롤링 상관계수. 컬럼명 형식: 'A-B'.
    """
    cols = list(returns.columns)
    out = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            out[f"{a}-{b}"] = returns[a].rolling(window).corr(returns[b])
    return pd.DataFrame(out).dropna(how="all")


def risk_contribution(returns: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """
    각 ETF의 포트폴리오 분산 기여도.

    σ_p² = wᵀ Σ w = Σᵢ wᵢ · (Σ w)ᵢ
    리스크 기여도(분산 기준) = wᵢ · (Σ w)ᵢ / σ_p²
    """
    tickers = [t for t in weights.keys() if t in returns.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    cov = returns[tickers].cov() * TRADING_DAYS  # 연율화
    cov_vals = cov.values
    port_var = float(w @ cov_vals @ w)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0

    marginal = cov_vals @ w                         # ∂σ_p²/∂w (dimless)
    contrib_var = w * marginal                      # (N,)
    pct = contrib_var / port_var if port_var > 0 else np.zeros_like(w)
    vol_contrib = contrib_var / port_vol if port_vol > 0 else np.zeros_like(w)

    return pd.DataFrame({
        "Weight": w,
        "VolContribution": vol_contrib,
        "RiskContribPct": pct,
    }, index=tickers)
