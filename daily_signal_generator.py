import yfinance as yf
import numpy as np
import pandas as pd
import sys

# --- [1. '전략 1.80' 파라미터 설정] ---
BASE_WEIGHTS = {
    'QQQ': 0.45,
    'GLD': 0.20,
    'Tactical_Bond': 0.35
}
N_BAND = 0.03 # 3% 이격도
MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0} # 시나리오 A
RATE_MA_WINDOW = 200
BOND_RISING_RATE = 'IEF'
BOND_FALLING_RATE = 'TLT'

# 분석할 티커 목록
core_tickers = ['QQQ', 'GLD']
bond_tickers = [BOND_RISING_RATE, BOND_FALLING_RATE]
rate_ticker = ['^TNX']
all_tickers = core_tickers + bond_tickers + rate_ticker

# --- [2. 일일 신호 계산 함수] ---

def get_daily_signals_and_report():
    
    print("... 최신 시장 데이터 다운로드 중 ...")
    # MA 계산 및 상태 확인을 위해 400일(200MA + 버퍼) 데이터 다운로드
    data_full = yf.download(all_tickers, period="400d", progress=False)
    
    if data_full.empty:
        raise ValueError("데이터 다운로드에 실패했습니다.")
    
    all_prices_df = data_full['Close']
    
    # --- Tactical_Bond (IEF/TLT) 생성 ---
    rate_prices = all_prices_df['^TNX'].ffill()
    rate_ma = rate_prices.rolling(window=RATE_MA_WINDOW).mean()
    is_rising_rates = (rate_prices > rate_ma)
    
    bond_prices = pd.Series(
        np.where(
            is_rising_rates, 
            all_prices_df[BOND_RISING_RATE].ffill(),
            all_prices_df[BOND_FALLING_RATE].ffill()
        ), 
        index=all_prices_df.index
    )
    bond_prices.name = 'Tactical_Bond'
    
    # --- 최종 분석 데이터 준비 ---
    analysis_tickers = ['QQQ', 'GLD', 'Tactical_Bond']
    prices_for_signal = pd.concat([all_prices_df[['QQQ', 'GLD']].ffill(), bond_prices.ffill()], axis=1)
    
    # --- [3. 이격도(Hysteresis) 상태 계산] ---
    # 이 스크립트는 매일 실행되므로, '어제 상태'를 알기 위해
    # 최소 200일 전부터의 상태를 전부 재계산해야 합니다.
    
    # MA 및 밴드 미리 계산
    ma_lines = {}
    upper_bands = {}
    lower_bands = {}
    for ticker in analysis_tickers:
        for window in MA_WINDOWS:
            ma_key = f"{ticker}_{window}"
            ma_lines[ma_key] = prices_for_signal[ticker].rolling(window=window).mean()
            upper_bands[ma_key] = ma_lines[ma_key] * (1.0 + N_BAND)
            lower_bands[ma_key] = ma_lines[ma_key] * (1.0 - N_BAND)

    # '상태' 저장을 위한 변수 초기화 (0.0 = OFF, 1.0 = ON)
    yesterday_ma_states = {f"{ticker}_{window}": 0.0 for ticker in analysis_tickers for window in MA_WINDOWS}
    
    # 일별 반복문 (MA 계산이 완료된 시점부터)
    start_index = max(MA_WINDOWS) - 1 
    
    for i in range(start_index, len(prices_for_signal)):
        
        today_ma_states = {}
        
        for ticker in analysis_tickers:
            for window in MA_WINDOWS:
                ma_key = f"{ticker}_{window}"
                yesterday_state = yesterday_ma_states[ma_key]
                
                price = prices_for_signal[ticker].iloc[i]
                upper = upper_bands[ma_key].iloc[i]
                lower = lower_bands[ma_key].iloc[i]
                
                if pd.isna(upper): new_state = 0.0
                elif yesterday_state == 1.0: 
                    new_state = 1.0 if price >= lower else 0.0
                else: 
                    new_state = 1.0 if price > upper else 0.0
                
                today_ma_states[ma_key] = new_state
        
        # '어제 상태'를 '오늘 상태'로 업데이트
        yesterday_ma_states = today_ma_states
        
    # 반복문이 끝나면, 'yesterday_ma_states'에 가장 마지막 날(어제)의 최종 상태가 저장됨
    final_ma_states = yesterday_ma_states

    # --- [4. 최종 비중 계산] ---
    
    # 1. 어제 날짜
    yesterday = prices_for_signal.index[-1]
    
    # 2. 어제 기준 MA 점수 계산 (0~3점)
    ma_scores = pd.Series(0, index=analysis_tickers)
    for ticker in analysis_tickers:
        score = 0
        for window in MA_WINDOWS:
            score += final_ma_states[f"{ticker}_{window}"]
        ma_scores[ticker] = score

    # 3. 시나리오 A 스케일러(Scalar) 적용
    scalars = ma_scores.map(SCALAR_MAP) # 예: QQQ 0.75, GLD 0.50, Bond 1.0
    
    # 4. 최종 투자 비중
    invested_qqq = BASE_WEIGHTS['QQQ'] * scalars['QQQ']
    invested_gld = BASE_WEIGHTS['GLD'] * scalars['GLD']
    invested_bond = BASE_WEIGHTS['Tactical_Bond'] * scalars['Tactical_Bond']
    total_cash = 1.0 - (invested_qqq + invested_gld + invested_bond)
    
    # --- [5. 알림 메시지 생성] ---
    
    # 채권 종류 확인
    current_bond_ticker = BOND_RISING_RATE if is_rising_rates.iloc[-1] else BOND_FALLING_RATE
    
    # 전일 종가 및 증감율
    price_info = prices_for_signal.iloc[-1]
    price_change = prices_for_signal.pct_change().iloc[-1]
    
    report = []
    report.append(f"🔔 Adaptive-Hysteresis-TAA (Sharpe 1.80)")
    report.append(f"   ({yesterday.strftime('%Y-%m-%d')} 마감 기준)")
    report.append("="*30)
    
    # 1. 전일자 정보
    report.append("📈 [1] 전일 시장 현황")
    report.append(f"  - QQQ: ${price_info['QQQ']:.2f} ({price_change['QQQ']:.2%})")
    report.append(f"  - GLD: ${price_info['GLD']:.2f} ({price_change['GLD']:.2%})")
    report.append(f"  - 채권({current_bond_ticker}): ${price_info['Tactical_Bond']:.2f} ({price_change['Tactical_Bond']:.2%})")

    report.append("\n" + "="*30)
    
    # 2. MA 신호 상세
    report.append("📊 [2] MA 신호 (이격도 +/- 3% 적용)")
    for ticker in analysis_tickers:
        t_price = price_info[ticker]
        t_str = f"  - {ticker} (신호: {ma_scores[ticker]}/3개 ON)"
        report.append(t_str)
        
        for window in MA_WINDOWS:
            ma_key = f"{ticker}_{window}"
            ma_val = ma_lines[ma_key].iloc[-1]
            state = "ON" if final_ma_states[ma_key] == 1.0 else "OFF"
            disparity = (t_price / ma_val) - 1.0
            report.append(f"    - {window}일: {state} (이격도: {disparity:+.2%})")

    report.append("\n" + "="*30)
    
    # 3. 최종 비중
    report.append("💰 [3] 오늘 목표 비중 (리밸런싱)")
    report.append(f"  - QQQ: {invested_qqq:.2%}")
    report.append(f"  - GLD: {invested_gld:.2%}")
    
    if current_bond_ticker == 'IEF':
        report.append(f"  - IEF (채권): {invested_bond:.2%}")
        report.append(f"  - TLT (채권): 0.00%")
    else:
        report.append(f"  - IEF (채권): 0.00%")
        report.append(f"  - TLT (채권): {invested_bond:.2%}")
        
    report.append(f"  - 현금 (Cash): {total_cash:.2%}")
    report.append("-" * 30)
    report.append(f"  * 총합: {invested_qqq + invested_gld + invested_bond + total_cash:.2%}")
    
    return "\n".join(report)

# --- [6. 메인 실행] ---
if __name__ == "__main__":
    try:
        daily_report = get_daily_signals_and_report()
        # GitHub Actions가 이 print() 출력을 캡처하여 텔레그램으로 전송합니다.
        print(daily_report)
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}", file=sys.stderr)
        sys.exit(1)
