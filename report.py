"""
PDF 리포트 생성 (reportlab + matplotlib).

주의: 한글 폰트 이슈를 피하기 위해 리포트는 영문으로 생성합니다.
"""
from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # 헤드리스
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from backtest import (
    BacktestResult,
    cagr,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    volatility,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#888",
    "axes.grid": True,
    "grid.color": "#eee",
    "font.size": 9,
})


def _fig_to_png_buffer(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _equity_png(result: BacktestResult) -> BytesIO:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(result.portfolio_equity.index, result.portfolio_equity.values,
            color="#d4a017", linewidth=2.2, label="Portfolio")
    if result.benchmark_equity is not None:
        ax.plot(result.benchmark_equity.index, result.benchmark_equity.values,
                color="#2a7fd1", linewidth=1.5, linestyle="--",
                label=f"Benchmark ({result.benchmark_ticker})")
    for col in result.asset_equity.columns:
        ax.plot(result.asset_equity.index, result.asset_equity[col],
                linewidth=0.9, alpha=0.55, label=col)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    return _fig_to_png_buffer(fig)


def _drawdown_png(result: BacktestResult) -> BytesIO:
    dd = drawdown_series(result.portfolio_equity) * 100
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.fill_between(dd.index, dd.values, 0, color="#ef4444", alpha=0.35)
    ax.plot(dd.index, dd.values, color="#b91c1c", linewidth=1)
    ax.set_title("Drawdown")
    ax.set_ylabel("DD (%)")
    return _fig_to_png_buffer(fig)


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    return f"{v * 100:.2f}%"


def _fmt_num(v):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    return f"{v:.2f}"


def generate_pdf_report(result: BacktestResult, perf_df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        title="ETF Backtest Report",
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["Normal"],
        textColor=colors.grey, fontSize=8,
    ))

    story: list = []
    story.append(Paragraph("ETF Portfolio Backtest Report", styles["Title"]))
    story.append(Spacer(1, 6))

    period = (f"{result.portfolio_equity.index[0].date()} "
              f"to {result.portfolio_equity.index[-1].date()}")
    meta = [
        f"<b>Period:</b> {period}",
        f"<b>Initial Capital:</b> ${result.initial_capital:,.0f}",
        f"<b>Rebalance:</b> {result.rebalance_freq or 'None'}",
        f"<b>Benchmark:</b> {result.benchmark_ticker or '—'}",
    ]
    for line in meta:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 10))

    # --- Key metrics ---
    tr = total_return(result.portfolio_equity)
    cg = cagr(result.portfolio_equity)
    sh = sharpe_ratio(result.portfolio_returns)
    so = sortino_ratio(result.portfolio_returns)
    mdd = max_drawdown(result.portfolio_equity)
    vol = volatility(result.portfolio_returns)

    metrics_data = [
        ["Total Return", "CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown"],
        [_fmt_pct(tr), _fmt_pct(cg), _fmt_pct(vol),
         _fmt_num(sh), _fmt_num(so), _fmt_pct(mdd)],
    ]
    mt = Table(metrics_data, colWidths=[2.6 * cm] * 6)
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3f5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f7fb")]),
    ]))
    story.append(mt)
    story.append(Spacer(1, 12))

    # --- Weights ---
    story.append(Paragraph("<b>Portfolio Weights</b>", styles["Heading3"]))
    wdata = [["Ticker", "Weight"]]
    for t, w in result.weights.items():
        wdata.append([t, f"{w * 100:.1f}%"])
    wt = Table(wdata, colWidths=[4 * cm, 3 * cm])
    wt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3f5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(wt)
    story.append(Spacer(1, 14))

    # --- Equity chart ---
    story.append(Image(_equity_png(result), width=17 * cm, height=7.2 * cm))
    story.append(Spacer(1, 6))

    # --- Drawdown chart ---
    story.append(Image(_drawdown_png(result), width=17 * cm, height=4.8 * cm))
    story.append(Spacer(1, 12))

    # --- Performance table ---
    story.append(Paragraph("<b>Performance Metrics</b>", styles["Heading3"]))
    pct_cols = {"Total Return", "CAGR", "Volatility", "Max Drawdown"}
    header = ["Ticker"] + list(perf_df.columns)
    pdata = [header]
    for idx, row in perf_df.iterrows():
        line = [str(idx)]
        for c in perf_df.columns:
            v = row[c]
            if c in pct_cols:
                line.append(_fmt_pct(v))
            else:
                line.append(_fmt_num(v))
        pdata.append(line)

    pt = Table(pdata, repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3f5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.HexColor("#f5f7fb")]),
    ]))
    story.append(pt)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Generated by ETF Portfolio Backtester · data: yfinance",
        styles["Caption"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
