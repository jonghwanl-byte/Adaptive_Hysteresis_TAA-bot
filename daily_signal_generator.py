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
    
    all_prices_df = data_full['Close'].ffill() # 중간에 빈 데이터(휴일 등)를 채움
    
    # --- Tactical_Bond (IEF/TLT) 생성 ---
    rate_prices = all_prices_df['^TNX']
    rate_ma = rate_prices.rolling(window=RATE_MA_WINDOW).mean()
    is_rising_rates = (rate_prices > rate_ma)
    
    bond_prices = pd.Series(
        np.where(
            is_rising_rates, 
            all_prices_df[BOND_RISING_RATE],
            all_prices_df[BOND_FALLING_RATE]
        ), 
        index=all_prices_df.index
    )
    bond_prices.name = 'Tactical_Bond'
    
    # --- 최종 분석 데이터 준비 ---
    analysis_tickers = ['QQQ', 'GLD', 'Tactical_Bond']
    prices_for_signal = pd.concat([all_prices_df[['QQQ', 'GLD']], bond_prices], axis=1)
    
    # --- [3. 이격도(Hysteresis) 상태 계산] ---
    
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
    
    # '어제'와 '오늘'의 스케일러(비중)를 저장할 변수
    today_scalars = pd.Series(0.0, index=analysis_tickers)
    yesterday_scalars = pd.Series(0.0, index=analysis_tickers)
    
    # '어제'와 '오늘'의 MA 상태(ON/OFF)를 저장할 변수
    today_ma_states_dict = yesterday_ma_states.copy()
    yesterday_ma_states_dict = yesterday_ma_states.copy()

    # 일별 반복문 (MA 계산이 완료된 시점부터)
    start_index = max(MA_WINDOWS) - 1 
    
    for i in range(start_index, len(prices_for_signal)):
        
        today_scores = pd.Series(0, index=analysis_tickers)
        current_ma_states = {}
        
        for ticker in analysis_tickers:
            score = 0
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
                
                current_ma_states[ma_key] = new_state
                score += new_state
            
            today_scores[ticker] = score
        
        # '어제'가 마지막 날이면, '어제'의 스케일러와 상태를 저장
        if i == len(prices_for_signal) - 2:
            yesterday_scalars = today_scores.map(SCALAR_MAP)
            yesterday_ma_states_dict = current_ma_states
            
        # '오늘'이 마지막 날이면, '오늘'의 스케일러와 상태를 저장
        if i == len(prices_for_signal) - 1:
            today_scalars = today_scores.map(SCALAR_MAP)
            today_ma_states_dict = current_ma_states
        
        # '어제 상태'를 '오늘 상태'로 업데이트
        yesterday_ma_states = current_ma_states

    # --- [4. 최종 비중 계산] ---
    
    # 오늘 비중
    today_invested_qqq = BASE_WEIGHTS['QQQ'] * today_scalars['QQQ']
    today_invested_gld = BASE_WEIGHTS['GLD'] * today_scalars['GLD']
    today_invested_bond = BASE_WEIGHTS['Tactical_Bond'] * today_scalars['Tactical_Bond']
    today_total_cash = 1.0 - (today_invested_qqq + today_invested_gld + today_invested_bond)
    
    # 어제 비중
    yesterday_invested_qqq = BASE_WEIGHTS['QQQ'] * yesterday_scalars['QQQ']
    yesterday_invested_gld = BASE_WEIGHTS['GLD'] * yesterday_scalars['GLD']
    yesterday_invested_bond = BASE_WEIGHTS['Tactical_Bond'] * yesterday_scalars['Tactical_Bond']
    yesterday_total_cash = 1.0 - (yesterday_invested_qqq + yesterday_invested_gld + yesterday_invested_bond)
    
    # 비중 변경 여부 확인
    is_rebalancing_needed = not (today_scalars.equals(yesterday_scalars))
    
    # --- [5. 알림 메시지 생성] ---
    
    yesterday = prices_for_signal.index[-1]
    
    # 채권 종류 확인
    current_bond_ticker = BOND_RISING_RATE if is_rising_rates.iloc[-1] else BOND_FALLING_RATE
    
    # 전일 종가 및 증감율
    price_info = prices_for_signal.iloc[-1]
    price_change = prices_for_signal.pct_change().iloc[-1]
    
    report = []
    report.append(f"🔔 Adaptive Hysteresis TAA (Sharpe 1.80)")
    report.append(f"({yesterday.strftime('%Y-%m-%d %A')} 마감 기준)") # 요일 추가

    # [1] 리밸런싱 신호
    if is_rebalancing_needed:
        report.append("\n" + "🔼 ====================== 🔼")
        report.append("    리밸런싱 신호: \"매매 필요\"")
        report.append("🔼 ====================== 🔼")
        report.append("(MA 신호 변경으로 목표 비중이 어제와 다릅니다)")
    else:
        report.append("\n" + "🟢 ====================== 🟢")
        report.append("    리밸런싱 신호: \"매매 불필요\"")
        report.append("🟢 ====================== 🟢")
        report.append("(모든 MA 신호가 어제와 동일하게 유지되었습니다)")
    
    report.append("\n" + "---")

    # [2] 오늘 목표 비중
    report.append("💰 [1] 오늘 목표 비중 (신규)")
    
    def get_emoji(ticker):
        if today_scalars[ticker] != yesterday_scalars[ticker]:
            return "🎯"
        return "*"
    
    report.append(f" {get_emoji('QQQ')} QQQ: {today_invested_qqq:.1%}")
    report.append(f" {get_emoji('GLD')} GLD: {today_invested_gld:.1%}")
    
    if current_bond_ticker == 'IEF':
        report.append(f" {get_emoji('Tactical_Bond')} IEF (채권): {today_invested_bond:.1%}")
        report.append(f" * TLT (채권): 0.0%")
    else:
        report.append(f" * IEF (채권): 0.0%")
        report.append(f" {get_emoji('Tactical_Bond')} TLT (채권): {today_invested_bond:.1%}")
    
    # 현금 비중 변경 확인
    cash_emoji = "🎯" if today_total_cash != yesterday_total_cash else "*"
    report.append(f" {cash_emoji} 현금 (Cash): {today_total_cash:.1%}")
    
    report.append("\n" + "---")
    
    # [3] 비중 변경 상세
    report.append("📊 [2] 비중 변경 상세 (매매 신호)")
    report.append("\n" + "| 자산 | 변경 전 (어제) | 변경 후 (오늘) | 변경폭 |")
    report.append("| :--- | :---: | :---: | :---: |")
    
    def format_change(today, yesterday, ticker):
        delta = today - yesterday
        if abs(delta) < 0.0001: return "(유지)"
        emoji = "🔼" if delta > 0 else "🔽"
        return f"{emoji} {delta:+.1%}"
    
    report.append(f"| QQQ | {yesterday_invested_qqq:.1%} | {today_invested_qqq:.1%} | {format_change(today_invested_qqq, yesterday_invested_qqq, 'QQQ')} |")
    report.append(f"| GLD | {yesterday_invested_gld:.1%} | {today_invested_gld:.1%} | {format_change(today_invested_gld, yesterday_invested_gld, 'GLD')} |")
    
    if current_bond_ticker == 'IEF':
        report.append(f"| IEF | {yesterday_invested_bond:.1%} | {today_invested_bond:.1%} | {format_change(today_invested_bond, yesterday_invested_bond, 'Tactical_Bond')} |")
    else:
        report.append(f"| TLT | {yesterday_invested_bond:.1%} | {today_invested_bond:.1%} | {format_change(today_invested_bond, yesterday_invested_bond, 'Tactical_Bond')} |")
    
    report.append(f"| 현금 | {yesterday_total_cash:.1%} | {today_total_cash:.1%} | {format_change(today_total_cash, yesterday_total_cash, 'Cash')} |")
    
    report.append("\n" + "---")
    
    # [4] 전일 시장 현황
    report.append("📈 [3] 전일 시장 현황")
    
    def format_price_change(value):
        emoji = "🔵" if value >= 0 else "🔴"
        return f"{emoji} ({value:+.1%})"
        
    report.append(f"{format_price_change(price_change['QQQ'])} QQQ: ${price_info['QQQ']:.1f}")
    report.append(f"{format_price_change(price_change['GLD'])} GLD: ${price_info['GLD']:.1f}")
    report.append(f"{format_price_change(price_change['Tactical_Bond'])} 채권({current_bond_ticker}): ${price_info['Tactical_Bond']:.1f}")
    
    report.append("\n" + "---")
    
    # [5] MA 신호 상세
    report.append("🔍 [4] MA 신호 상세 (오늘 기준)")
    report.append(f"(이격도 +/- {N_BAND:.1%} 룰 적용)")
    
    for ticker in analysis_tickers:
        score = int(today_scalars[ticker] * 4 / (4/3)) # 1.0 -> 3, 0.75 -> 2, 0.5 -> 1, 0 -> 0
        status_emoji = "🟢ON" if score > 0 else "🔴OFF"
        report.append(f"\n**{ticker} (신호: {score}/3개 {status_emoji})**")
        
        for window in MA_WINDOWS:
            ma_key = f"{ticker}_{window}"
            
            today_state_val = today_ma_states_dict[ma_key]
            yesterday_state_val = yesterday_ma_states_dict[ma_key]
            
            state_emoji = "🟢ON" if today_state_val == 1.0 else "🔴OFF"
            
            # 신호 변경 상태
            if today_state_val > yesterday_state_val: state_change = "[신규 ON]"
            elif today_state_val < yesterday_state_val: state_change = "[신규 OFF]"
            else: state_change = "[유지]"
            
            t_price = price_info[ticker]
            ma_val = ma_lines[ma_key].iloc[-1]
            disparity = (t_price / ma_val) - 1.0
            
            report.append(f"* {window}일: {state_emoji} (이격도: {disparity:+.1%}) {state_change}")
    
    return "\n".join(report)

# --- [6. 메인 실행] ---
if __name__ == "__main__":
    try:
        # pandas 출력 옵션 (터미널에서 잘 보이도록)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)
        
        daily_report = get_daily_signals_and_report()
        # GitHub Actions가 이 print() 출력을 캡처하여 텔레그램으로 전송합니다.
        print(daily_report)
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}", file=sys.stderr)
        sys.exit(1)
