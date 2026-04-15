# 📈 ETF Portfolio Backtester

Streamlit으로 만든 개인용 ETF 포트폴리오 백테스트 웹앱.
yfinance 데이터를 받아 비중, 리밸런싱 주기, 기간을 자유롭게 바꿔가며
포트폴리오의 누적 수익률·리스크·배당을 한눈에 분석합니다.

## ✨ 주요 기능

- **포트폴리오 시뮬레이션** — 원하는 ETF 조합과 비중으로 실제 주식수 기반 시뮬레이션
- **리밸런싱** — None / Monthly / Quarterly / Semi-Annual / Annual
- **성과 지표** — Total Return, CAGR, Sharpe, Sortino, MDD, Beta, Volatility
- **인터랙티브 차트** (Plotly)
  - 누적 수익률 (포트폴리오 / 벤치마크 / 개별 ETF)
  - 드로우다운 영역 차트
  - 연도별 · 월별 수익률 히트맵
  - 상관관계 매트릭스
- **배당 분석** — 연간 주당 배당금 바차트 + 지급 이력 테이블
- **CSV 다운로드** — 자산곡선, 성과지표 내보내기

## 🚀 설치 & 실행

```bash
cd etf-backtest
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열리면 (보통 http://localhost:8501) 사이드바에서
티커/비중/기간을 설정하고 **백테스트 실행** 버튼을 누르세요.

## 📁 프로젝트 구조

```
etf-backtest/
├── app.py           # Streamlit UI
├── backtest.py      # 백테스트 엔진 (순수 계산 로직)
├── utils.py         # 포매팅 & Plotly 차트 헬퍼
├── requirements.txt
└── README.md
```

- `backtest.py` 는 UI 의존성이 없어 단독 테스트/노트북 사용이 가능합니다.
- `utils.py` 는 Plotly 차트 생성과 포매팅 함수 모음.

## 🧮 계산 로직 요약

1. `yf.download(..., auto_adjust=True)` 로 수정종가 일괄 다운로드
2. 초기 투자금을 목표 비중대로 배분해 **주식수**로 보유
3. 리밸런싱 시점마다 당일 평가액을 다시 목표 비중으로 재분배
4. 일별 평가액 변화로 일 수익률을 산출하고 지표 계산
   - `CAGR = (V_end / V_start) ^ (1/years) - 1`
   - `Volatility = daily_std * √252`
   - `Sharpe = (ann_return - rf) / vol`
   - `Sortino = (ann_return - rf) / downside_std`
   - `MDD = min(equity / cummax - 1)`
   - `Beta = cov(r_p, r_b) / var(r_b)`

## ⚠️ 알려진 제약

- 거래비용·슬리피지·세금 반영 없음 (개인 참고용)
- yfinance가 제공하지 않는 ETF는 조회 불가
- 무위험 수익률은 기본 0% (사이드바에서 노출하지 않음)

## 📝 기본 포트폴리오

기본값은 `QLD,SCHD,ALLW` / 균등 비중 / 분기 리밸런싱 / 벤치마크 SPY입니다.
사이드바에서 자유롭게 바꿔보세요.
