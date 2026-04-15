"""
ETF Portfolio Backtester — Streamlit UI
실행: streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from backtest import (
    REBALANCE_FREQ,
    annual_dividends,
    cagr,
    correlation_matrix,
    drawdown_series,
    fetch_dividends,
    max_drawdown,
    monte_carlo_percentiles,
    monte_carlo_simulation,
    monte_carlo_summary,
    monthly_returns_matrix,
    performance_table,
    risk_contribution,
    rolling_correlations,
    run_backtest,
    sharpe_ratio,
    total_return,
    volatility_drag_analysis,
    yearly_returns_table,
)
from report import generate_pdf_report
from utils import (
    correlation_heatmap,
    dividend_bar_chart,
    drawdown_chart,
    equity_chart,
    fmt_money,
    fmt_num,
    fmt_pct,
    monte_carlo_fan_chart,
    monte_carlo_histogram,
    monthly_heatmap,
    parse_tickers,
    risk_contribution_pie,
    rolling_correlation_chart,
    style_performance_table,
    volatility_drag_area_chart,
    volatility_drag_chart,
    yearly_heatmap,
)

st.set_page_config(
    page_title="ETF Portfolio Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, int]] = {
    "공격형":  {"QLD": 60, "SCHD": 20, "ALLW": 20},
    "균형형":  {"QLD": 40, "SCHD": 30, "ALLW": 30},
    "방어형":  {"QLD": 20, "SCHD": 40, "ALLW": 40},
    "커스텀":  {},
}
PRESET_NAMES = list(PRESETS.keys())


def _apply_preset_to_state(name: str) -> None:
    """선택된 프리셋을 session_state에 심어둔다 (다음 렌더에 반영)."""
    if name == "커스텀":
        return
    preset = PRESETS[name]
    st.session_state["tickers_raw"] = ",".join(preset.keys())
    for t, v in preset.items():
        st.session_state[f"w_{t}"] = int(v)


def _on_preset_change() -> None:
    _apply_preset_to_state(st.session_state["preset"])


# 최초 진입 시 기본 프리셋 주입
if "initialized" not in st.session_state:
    st.session_state["preset"] = "공격형"
    _apply_preset_to_state("공격형")
    st.session_state["initialized"] = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ 포트폴리오 설정")

st.sidebar.selectbox(
    "프리셋",
    options=PRESET_NAMES,
    key="preset",
    on_change=_on_preset_change,
    help="프리셋을 고르면 티커와 비중이 자동 입력됩니다.",
)

st.sidebar.text_input(
    "ETF 티커 (쉼표 구분)",
    key="tickers_raw",
    help="예: QLD,SCHD,ALLW",
)
tickers = parse_tickers(st.session_state["tickers_raw"])

st.sidebar.markdown("**비중 (%)**")
weights: list[float] = []
for t in tickers:
    key = f"w_{t}"
    if key not in st.session_state:
        st.session_state[key] = int(round(100 / max(1, len(tickers))))
    st.sidebar.slider(f"{t}", min_value=0, max_value=100, step=1, key=key)
    weights.append(st.session_state[key])

total_w = sum(weights)
if total_w == 0:
    st.sidebar.error("비중 합계가 0입니다.")
elif total_w != 100:
    st.sidebar.warning(f"비중 합계 = {total_w}% (자동 정규화됨)")
else:
    st.sidebar.success(f"비중 합계 = {total_w}%")

initial_capital = st.sidebar.number_input(
    "초기 투자금 ($)",
    min_value=100.0, max_value=10_000_000.0,
    value=10_000.0, step=1_000.0,
)

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 5)
col_s, col_e = st.sidebar.columns(2)
with col_s:
    start_date = st.date_input("시작일", value=default_start, max_value=today)
with col_e:
    end_date = st.date_input("종료일", value=today, max_value=today)

rebalance = st.sidebar.selectbox(
    "리밸런싱 주기",
    options=list(REBALANCE_FREQ.keys()),
    index=2,
)

benchmark = st.sidebar.text_input("벤치마크 티커", value="SPY").strip().upper() or None

run_clicked = st.sidebar.button("🚀 백테스트 실행", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("📈 ETF Portfolio Backtester")
st.caption("개인용 포트폴리오 백테스트 · 거래비용 무시 · yfinance 데이터")

# 캐싱: 같은 입력이면 재계산 생략
@st.cache_data(show_spinner=False)
def _cached_run(tickers, weights, start, end, initial, rebal, bench):
    return run_backtest(
        tickers=list(tickers),
        weights=list(weights),
        start=start,
        end=end,
        initial_capital=float(initial),
        rebalance_freq=rebal,
        benchmark=bench,
    )


# 실행 여부 관리 — 한 번 실행하면 세션 동안 결과 유지
if run_clicked:
    st.session_state["run_token"] = True

if not st.session_state.get("run_token"):
    st.info("사이드바에서 설정을 조정한 뒤 **백테스트 실행** 버튼을 눌러주세요.")
    st.stop()

# --- 입력 검증 ---
if not tickers:
    st.error("최소 1개 이상의 티커를 입력하세요.")
    st.stop()
if start_date >= end_date:
    st.error("종료일은 시작일보다 뒤여야 합니다.")
    st.stop()
if total_w == 0:
    st.error("비중 합계가 0입니다. 슬라이더를 조정하세요.")
    st.stop()

with st.spinner("데이터 다운로드 및 백테스트 중..."):
    try:
        result = _cached_run(
            tuple(tickers), tuple(weights),
            start_date, end_date, initial_capital, rebalance, benchmark,
        )
    except Exception as e:
        st.error(f"백테스트 실패: {e}")
        st.stop()

# --- 메트릭 카드 ---
tr = total_return(result.portfolio_equity)
cg = cagr(result.portfolio_equity)
sh = sharpe_ratio(result.portfolio_returns)
mdd = max_drawdown(result.portfolio_equity)

m1, m2, m3, m4 = st.columns(4)
m1.metric("총 수익률", fmt_pct(tr))
m2.metric("CAGR", fmt_pct(cg))
m3.metric("샤프 비율", fmt_num(sh))
m4.metric("최대 낙폭 (MDD)", fmt_pct(mdd))

# --- PDF 다운로드 버튼 ---
perf_df = performance_table(result)
try:
    pdf_bytes = generate_pdf_report(result, perf_df)
    st.download_button(
        "📄 PDF 리포트 다운로드",
        data=pdf_bytes,
        file_name=f"backtest_report_{dt.date.today().isoformat()}.pdf",
        mime="application/pdf",
        use_container_width=False,
    )
except Exception as e:
    st.warning(f"PDF 생성 실패: {e}")

st.divider()

# --- 탭 ---
tab_perf, tab_risk, tab_stats, tab_div, tab_mc, tab_lev = st.tabs([
    "📊 성과",
    "⚠️ 리스크",
    "📋 통계",
    "💰 배당",
    "🎲 몬테카를로",
    "🔥 레버리지 분석",
])

# ===== 성과 =====
with tab_perf:
    st.subheader("누적 수익률")
    st.plotly_chart(
        equity_chart(
            portfolio=result.portfolio_equity,
            benchmark=result.benchmark_equity,
            assets=result.asset_equity,
            initial=result.initial_capital,
        ),
        use_container_width=True,
    )

    st.subheader("연도별 수익률")
    yearly_df = yearly_returns_table(result)
    st.plotly_chart(yearly_heatmap(yearly_df), use_container_width=True)

    st.subheader("월별 수익률 (포트폴리오)")
    monthly_df = monthly_returns_matrix(result.portfolio_equity)
    st.plotly_chart(monthly_heatmap(monthly_df), use_container_width=True)

# ===== 리스크 =====
with tab_risk:
    st.subheader("드로우다운")
    dd = drawdown_series(result.portfolio_equity)
    st.plotly_chart(drawdown_chart(dd), use_container_width=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("상관관계 매트릭스")
        corr = correlation_matrix(result)
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    with c2:
        st.subheader("리스크 기여도")
        rc_df = risk_contribution(result.returns, result.weights)
        st.plotly_chart(risk_contribution_pie(rc_df), use_container_width=True)
        display_rc = rc_df.copy()
        display_rc["Weight"] = display_rc["Weight"].map(lambda v: f"{v * 100:.1f}%")
        display_rc["VolContribution"] = display_rc["VolContribution"].map(lambda v: f"{v * 100:.2f}%")
        display_rc["RiskContribPct"] = display_rc["RiskContribPct"].map(lambda v: f"{v * 100:.1f}%")
        st.dataframe(display_rc, use_container_width=True)

    st.subheader("롤링 상관계수")
    if len(result.tickers) < 2:
        st.info("롤링 상관계수는 2개 이상의 ETF가 필요합니다.")
    else:
        window = st.radio(
            "윈도우",
            options=[60, 120],
            horizontal=True,
            format_func=lambda x: f"{x}일",
            key="roll_window",
        )
        roll_df = rolling_correlations(result.returns, window=window)
        if roll_df.empty:
            st.info("롤링 상관계수를 계산하기에 기간이 너무 짧습니다.")
        else:
            st.plotly_chart(rolling_correlation_chart(roll_df, window), use_container_width=True)

# ===== 통계 =====
with tab_stats:
    st.subheader("종합 성과 지표")
    st.dataframe(style_performance_table(perf_df), use_container_width=True)

    with st.expander("📥 Raw 데이터 다운로드"):
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "포트폴리오 자산곡선 CSV",
                data=result.portfolio_equity.to_csv().encode("utf-8"),
                file_name="portfolio_equity.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "성과 지표 CSV",
                data=perf_df.to_csv().encode("utf-8"),
                file_name="performance.csv",
                mime="text/csv",
            )

    st.markdown("**현재 목표 비중**")
    weights_df = pd.DataFrame({
        "Ticker": list(result.weights.keys()),
        "Weight": [f"{v * 100:.1f}%" for v in result.weights.values()],
    })
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

# ===== 배당 =====
with tab_div:
    st.subheader("배당 이력")
    with st.spinner("배당 데이터 가져오는 중..."):
        div_df = fetch_dividends(result.tickers, start_date, end_date)

    if div_df.empty:
        st.info("기간 내 배당 데이터가 없습니다.")
    else:
        annual = annual_dividends(div_df)
        st.plotly_chart(dividend_bar_chart(annual), use_container_width=True)

        st.markdown("**연간 배당 합계 (주당, $)**")
        st.dataframe(annual.style.format("{:.4f}"), use_container_width=True)

        st.markdown("**개별 배당 지급 이력**")
        display = div_df.copy()
        display.index = display.index.strftime("%Y-%m-%d")
        st.dataframe(display.style.format("{:.4f}"), use_container_width=True)

# ===== 몬테카를로 =====
with tab_mc:
    st.subheader("몬테카를로 시뮬레이션")
    st.caption("과거 포트폴리오 일별 수익률에서 부트스트랩 복원추출 (1000회).")

    c1, c2, c3 = st.columns(3)
    with c1:
        mc_years = st.slider("시뮬레이션 기간 (년)", 1, 30, 10, key="mc_years")
    with c2:
        mc_sims = st.slider("시뮬레이션 횟수", 200, 5000, 1000, step=100, key="mc_sims")
    with c3:
        mc_seed = st.number_input("랜덤 시드", value=42, step=1, key="mc_seed")

    with st.spinner("몬테카를로 실행 중..."):
        try:
            paths = monte_carlo_simulation(
                daily_returns=result.portfolio_returns,
                n_years=int(mc_years),
                n_sims=int(mc_sims),
                initial=result.initial_capital,
                seed=int(mc_seed),
            )
            pct_df = monte_carlo_percentiles(paths)
            summary = monte_carlo_summary(paths, result.initial_capital)
        except Exception as e:
            st.error(f"시뮬레이션 실패: {e}")
            paths = None

    if paths is not None:
        st.plotly_chart(
            monte_carlo_fan_chart(pct_df, result.initial_capital),
            use_container_width=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("중위 최종값", fmt_money(summary["median"]))
        m2.metric("5% 분위",    fmt_money(summary["p5"]))
        m3.metric("95% 분위",   fmt_money(summary["p95"]))
        m4.metric("원금 손실 확률", fmt_pct(summary["prob_loss"]))

        st.plotly_chart(
            monte_carlo_histogram(paths[-1], result.initial_capital),
            use_container_width=True,
        )

        st.caption(
            f"2배 달성 확률: {fmt_pct(summary['prob_double'])} · "
            f"평균 최종값: {fmt_money(summary['mean'])}"
        )

# ===== 레버리지 분석 =====
with tab_lev:
    st.subheader("변동성 드래그 분석")
    st.caption("레버리지 ETF의 일일 리밸런싱 때문에 발생하는 복리 경로 손실 측정.")

    default_lev = next((t for t in result.tickers if t in ("QLD", "SSO", "TQQQ", "UPRO", "SOXL")), "QLD")

    c1, c2, c3 = st.columns(3)
    with c1:
        lev_ticker = st.text_input("레버리지 ETF", value=default_lev, key="lev_ticker").strip().upper()
    with c2:
        base_ticker = st.text_input("기초지수 ETF", value="QQQ", key="base_ticker").strip().upper()
    with c3:
        lev_mult = st.number_input("레버리지 배수", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="lev_mult")

    if st.button("변동성 드래그 계산", key="calc_drag"):
        with st.spinner("데이터 다운로드 및 계산 중..."):
            try:
                drag_df = volatility_drag_analysis(
                    lev_ticker=lev_ticker,
                    base_ticker=base_ticker,
                    leverage=float(lev_mult),
                    start=start_date,
                    end=end_date,
                    initial=result.initial_capital,
                )
                st.session_state["drag_df"] = drag_df
            except Exception as e:
                st.error(f"계산 실패: {e}")
                st.session_state.pop("drag_df", None)

    drag_df = st.session_state.get("drag_df")
    if drag_df is not None and not drag_df.empty:
        final_theo = drag_df["theoretical"].iloc[-1]
        final_actual = drag_df["actual"].iloc[-1]
        drag_final = (final_theo - final_actual) / final_theo
        annualized_drag = (final_theo / final_actual) ** (
            365.25 / max(1, (drag_df.index[-1] - drag_df.index[0]).days)
        ) - 1

        m1, m2, m3 = st.columns(3)
        m1.metric(f"실제 최종값", fmt_money(final_actual))
        m2.metric(f"이론 최종값", fmt_money(final_theo))
        m3.metric("누적 드래그", fmt_pct(drag_final),
                  delta=f"연율 ~{annualized_drag * 100:.2f}%")

        st.plotly_chart(volatility_drag_chart(drag_df), use_container_width=True)
        st.plotly_chart(volatility_drag_area_chart(drag_df), use_container_width=True)

        with st.expander("원시 데이터"):
            st.dataframe(drag_df.tail(20), use_container_width=True)
    else:
        st.info("**변동성 드래그 계산** 버튼을 눌러 분석을 시작하세요.")

st.divider()
st.caption(
    f"Backtest Period: {result.portfolio_equity.index[0].date()} → "
    f"{result.portfolio_equity.index[-1].date()} · "
    f"Rebalance: {result.rebalance_freq} · "
    f"Benchmark: {result.benchmark_ticker or '—'}"
)
