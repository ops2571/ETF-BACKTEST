"""
UI/포매팅 유틸리티.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DARK_TEMPLATE = "plotly_dark"


def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def equity_chart(
    portfolio: pd.Series,
    benchmark: pd.Series | None,
    assets: pd.DataFrame | None,
    initial: float,
) -> go.Figure:
    fig = go.Figure()

    if assets is not None:
        for col in assets.columns:
            fig.add_trace(go.Scatter(
                x=assets.index, y=assets[col],
                mode="lines", name=col,
                line=dict(width=1.2),
                opacity=0.55,
                hovertemplate="%{x|%Y-%m-%d}<br>" + col + ": $%{y:,.0f}<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=portfolio.index, y=portfolio.values,
        mode="lines", name="Portfolio",
        line=dict(width=3, color="#f5d547"),
        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>",
    ))

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values,
            mode="lines", name=str(benchmark.name or "Benchmark"),
            line=dict(width=2, color="#6ec1e4", dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Benchmark: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"누적 수익률 (초기 ${initial:,.0f} 기준)",
        xaxis_title="",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def drawdown_chart(dd: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#ef4444", width=1),
        fillcolor="rgba(239,68,68,0.35)",
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="드로우다운 (고점 대비)",
        yaxis_title="Drawdown (%)",
        xaxis_title="",
        height=350,
    )
    return fig


def yearly_heatmap(yearly_df: pd.DataFrame) -> go.Figure:
    z = yearly_df.values * 100  # %
    text = np.vectorize(lambda v: f"{v:.1f}%" if not np.isnan(v) else "")(z)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=yearly_df.columns.tolist(),
        y=yearly_df.index.astype(str).tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=text,
        texttemplate="%{text}",
        hovertemplate="%{y} · %{x}<br>%{z:.2f}%<extra></extra>",
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="연도별 수익률",
        height=max(300, 40 * len(yearly_df.index) + 120),
        xaxis=dict(side="top"),
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    z = corr.values
    text = np.vectorize(lambda v: f"{v:.2f}")(z)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=text,
        texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="상관관계 (일별 수익률)",
        height=420,
    )
    return fig


def monthly_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # 누락된 월은 NaN으로 채우고 1~12 순서 보장
    df = monthly_df.reindex(columns=range(1, 13))
    z = df.values * 100
    text = np.vectorize(lambda v: f"{v:.1f}%" if not np.isnan(v) else "")(z)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=month_labels,
        y=df.index.astype(str).tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=text,
        texttemplate="%{text}",
        hovertemplate="%{y}년 %{x}<br>%{z:.2f}%<extra></extra>",
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="월별 수익률 (포트폴리오)",
        height=max(300, 38 * len(df.index) + 120),
        xaxis=dict(side="top"),
    )
    return fig


def style_performance_table(df: pd.DataFrame):
    """pandas Styler로 퍼센트/숫자 포맷팅."""
    pct_cols = ["Total Return", "CAGR", "Volatility", "Max Drawdown"]
    num_cols = ["Sharpe", "Sortino", "Beta"]

    fmt = {}
    for c in pct_cols:
        if c in df.columns:
            fmt[c] = lambda v: "—" if pd.isna(v) else f"{v * 100:.2f}%"
    for c in num_cols:
        if c in df.columns:
            fmt[c] = lambda v: "—" if pd.isna(v) else f"{v:.2f}"

    styler = df.style.format(fmt)
    # 가장 눈에 띄게: Max Drawdown은 빨강, CAGR은 초록 그라데이션
    if "CAGR" in df.columns:
        styler = styler.background_gradient(subset=["CAGR"], cmap="Greens")
    if "Max Drawdown" in df.columns:
        styler = styler.background_gradient(subset=["Max Drawdown"], cmap="Reds_r")
    return styler


def monte_carlo_fan_chart(percentiles: pd.DataFrame, initial: float) -> go.Figure:
    """5/25/50/75/95 퍼센타일 밴드."""
    x = percentiles.index
    fig = go.Figure()

    # 5-95 밴드
    fig.add_trace(go.Scatter(
        x=x, y=percentiles["p95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=percentiles["p5"], mode="lines",
        line=dict(width=0),
        fill="tonexty", fillcolor="rgba(110,193,228,0.15)",
        name="5–95%",
        hovertemplate="Day %{x}<br>p5: $%{y:,.0f}<extra></extra>",
    ))

    # 25-75 밴드
    fig.add_trace(go.Scatter(
        x=x, y=percentiles["p75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=percentiles["p25"], mode="lines",
        line=dict(width=0),
        fill="tonexty", fillcolor="rgba(110,193,228,0.35)",
        name="25–75%",
        hovertemplate="Day %{x}<br>p25: $%{y:,.0f}<extra></extra>",
    ))

    # 중위값
    fig.add_trace(go.Scatter(
        x=x, y=percentiles["p50"], mode="lines",
        line=dict(color="#f5d547", width=2.5),
        name="Median",
        hovertemplate="Day %{x}<br>Median: $%{y:,.0f}<extra></extra>",
    ))

    # 초기값 기준선
    fig.add_hline(
        y=initial, line_dash="dot", line_color="#9ca3af",
        annotation_text=f"Initial ${initial:,.0f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="몬테카를로 자산 경로 (부트스트랩)",
        xaxis_title="Trading Days",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def monte_carlo_histogram(final_values: np.ndarray, initial: float) -> go.Figure:
    median = float(np.median(final_values))
    fig = go.Figure(data=[
        go.Histogram(
            x=final_values,
            nbinsx=60,
            marker_color="#6ec1e4",
            opacity=0.85,
            hovertemplate="$%{x:,.0f}<br>Count: %{y}<extra></extra>",
        )
    ])
    fig.add_vline(
        x=initial, line_dash="dot", line_color="#9ca3af",
        annotation_text="Initial", annotation_position="top",
    )
    fig.add_vline(
        x=median, line_dash="dash", line_color="#f5d547",
        annotation_text=f"Median ${median:,.0f}",
        annotation_position="top",
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="최종 자산 분포",
        xaxis_title="Final Equity ($)",
        yaxis_title="Frequency",
        height=380,
        bargap=0.02,
    )
    return fig


def volatility_drag_chart(df: pd.DataFrame) -> go.Figure:
    lev = df.attrs.get("lev_ticker", "Leveraged")
    base = df.attrs.get("base_ticker", "Base")
    lev_mult = df.attrs.get("leverage", 2.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["base"], mode="lines",
        name=f"{base} (1x)",
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>" + base + ": $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["theoretical"], mode="lines",
        name=f"이론 {lev_mult:g}×{base}",
        line=dict(color="#6ec1e4", width=2, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Theory: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["actual"], mode="lines",
        name=f"실제 {lev}",
        line=dict(color="#f5d547", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>" + lev + ": $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{lev} 실제 vs 이론 {lev_mult:g}× {base}",
        xaxis_title="",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        height=460,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def volatility_drag_area_chart(df: pd.DataFrame) -> go.Figure:
    lev = df.attrs.get("lev_ticker", "Leveraged")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["drag_pct"], mode="lines",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.3)",
        name="Drag",
        hovertemplate="%{x|%Y-%m-%d}<br>Drag: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"누적 변동성 드래그 (이론 − 실제) / 이론",
        xaxis_title="",
        yaxis_title="Drag (%)",
        height=320,
    )
    return fig


def rolling_correlation_chart(df: pd.DataFrame, window: int) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines",
            name=col, line=dict(width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>" + col + ": %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{window}일 롤링 상관계수",
        xaxis_title="",
        yaxis_title="ρ",
        yaxis=dict(range=[-1, 1]),
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def risk_contribution_pie(df: pd.DataFrame) -> go.Figure:
    pct = df["RiskContribPct"].clip(lower=0)  # 음수 기여도는 파이에서 0으로
    fig = go.Figure(data=[
        go.Pie(
            labels=df.index.tolist(),
            values=pct.values,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}<br>리스크 기여: %{percent}<extra></extra>",
        )
    ])
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="포트폴리오 변동성 기여도",
        height=420,
    )
    return fig


def dividend_bar_chart(annual_div: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in annual_div.columns:
        fig.add_trace(go.Bar(
            x=annual_div.index.astype(str),
            y=annual_div[col],
            name=col,
            hovertemplate="%{x}<br>" + col + ": $%{y:.3f}/share<extra></extra>",
        ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="연간 배당금 (주당)",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="$ / share",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
