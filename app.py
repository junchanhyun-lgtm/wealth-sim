import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------
# 1. 헬퍼 함수
# -----------------------------------------------------------
def format_won(value_in_manwon):
    val = int(value_in_manwon)
    if val >= 10000:
        eok = val // 10000
        man = val % 10000
        if man > 0: return f"{eok}억 {man}만 원"
        else: return f"{eok}억 원"
    else:
        return f"{val}만 원"

# -----------------------------------------------------------
# 2. 시뮬레이션 엔진 (V21: 안전한 Die With Zero 로직 적용)
# -----------------------------------------------------------
class FinancialSimulator:
    def __init__(self, params):
        self.params = params
        
    def get_lifecycle_multiplier(self, age):
        """기본 생활비 생애주기 계수"""
        if age < 60: return 1.0
        elif 60 <= age < 70: return 0.9
        elif 70 <= age < 80: return 0.85
        else: return 0.8

    def get_diminishing_return(self, base_return, current_asset_won):
        """자산 10억 초과 시 수익률 체감"""
        threshold = 1_000_000_000 
        if current_asset_won <= threshold:
            return base_return
        else:
            decay = 1 - (0.12 * np.log10(current_asset_won / threshold))
            return max(base_return * decay, 0.015)

    def run_monte_carlo(self, n_simulations=5000):
        current_age = self.params['current_age']
        death_age = self.params['death_age']
        current_asset = self.params['current_asset'] * 10000
        
        base_monthly_income = (self.params['monthly_income'] * 10000) * 12
        apply_income_inflation = self.params['apply_income_inflation']
        base_monthly_expense = self.params['monthly_expense'] * 10000
        
        base_return = self.params['expected_return'] / 100
        volatility = self.params['volatility'] / 100
        inflation = self.params['inflation'] / 100
        personality = self.params['personality']
        target_asset_won = self.params['retire_by_asset'] * 10000
        
        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)
        
        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_spending_pv = np.zeros((n_simulations, simulation_years))
        
        for i in range(n_simulations):
            nominal_asset = current_asset
            random_returns = np.random.normal(base_return, volatility, simulation_years)
            
            path_assets_pv = []
            path_spending_pv = []
            is_retired = False
            
            for t, age in enumerate(years):
                discount_factor = (1 + inflation) ** t
                
                # A. 수입 파악
                current_nominal_income = 0
                if not is_retired:
                    if age >= self.params['retire_age']: is_retired = True
                    elif target_asset_won > 0 and nominal_asset >= target_asset_won: is_retired = True
                    else:
                        if apply_income_inflation:
                            current_nominal_income = base_monthly_income * discount_factor
                        else:
                            current_nominal_income = base_monthly_income

                # B. 추가 수입/지출 파악
                extra_income = 0
                extra_expense = 0
                if 'recurring_events' in self.params:
                    for event in self.params['recurring_events']:
                        end_age = event['start_age'] + event['duration']
                        if event['start_age'] <= age < end_age:
                            amt = (event['monthly_amount'] * 10000) * 12 * discount_factor
                            if amt > 0: extra_income += amt
                            else: extra_expense += abs(amt)

                # C. [핵심] 인출 가능 상한액 계산 (미래 부채 예약 적용)
                reserved_asset = 0
                if personality == '소비 집중형' and 'lump_events' in self.params:
                    for event in self.params['lump_events']:
                        if event['age'] > age and event['amount'] < 0: # 미래 지출
                            time_diff = event['age'] - age
                            future_cost = abs(event['amount'] * 10000) * ((1 + inflation) ** (t + time_diff))
                            # 안전하게 물가상승률+1%로만 할인하여 미리 떼어놓음
                            reserved_asset += future_cost / ((1 + inflation + 0.01) ** time_diff)

                effective_asset = max(0, nominal_asset - reserved_asset)
                withdrawal_limit = 0

                if effective_asset > 0:
                    if personality == '소비 집중형':
                        remaining_years = death_age - age
                        # 인출 계획 금리 상한 (최대 5%)
                        planning_rate = min(base_return * 0.7, 0.05)
                        if remaining_years > 0:
                            withdrawal_limit = (effective_asset * planning_rate) / (1 - (1 + planning_rate) ** -remaining_years)
                    elif personality == '중도형':
                        withdrawal_limit = effective_asset * 0.04

                # D. 총 지출 가능 금액 (공식: 수입 + 인출한도 - 추가지출)
                total_income_stream = current_nominal_income + extra_income
                
                lifecycle_factor = self.get_lifecycle_multiplier(age)
                nominal_basic_expense = (base_monthly_expense * 12) * discount_factor * lifecycle_factor

                if personality == '소비 집중형':
                    spending_capacity = total_income_stream + withdrawal_limit - extra_expense
                    nominal_actual_spending = max(spending_capacity, nominal_basic_expense * 0.5)
                elif personality == '중도형':
                    spending_capacity = total_income_stream + withdrawal_limit - extra_expense
                    nominal_actual_spending = max(spending_capacity, nominal_basic_expense)
                else: # 자산 형성형
                    spending_capacity = total_income_stream - extra_expense
                    nominal_actual_spending = max(spending_capacity, nominal_basic_expense)

                # E. 목돈 이벤트
                nominal_lump_sum = 0
                if 'lump_events' in self.params:
                    for event in self.params['lump_events']:
                        if event['age'] == age:
                            nominal_lump_sum += (event['amount'] * 10000) * discount_factor

                # F. 자산 변동
                net_withdrawal = nominal_actual_spending - total_income_stream
                
                adj_return = self.get_diminishing_return(random_returns[t], nominal_asset)
                gain = nominal_asset * adj_return
                
                nominal_asset = nominal_asset + gain - net_withdrawal + nominal_lump_sum
                if nominal_asset < 0: nominal_asset = 0
                
                # G. 결과 저장 (PV)
                path_assets_pv.append(nominal_asset / discount_factor)
                path_spending_pv.append(nominal_actual_spending / discount_factor / 12)
            
            sim_assets_pv[i, :] = path_assets_pv
            sim_spending_pv[i, :] = path_spending_pv
            
        return years, sim_assets_pv, sim_spending_pv

# -----------------------------------------------------------
# 3. UI 구성
# -----------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Financial Asset Sim V22")
    
    st.title("💰 고정밀 금융자산 시뮬레이터 V22")
    st.info("💡 **현재가치 기준:** 모든 결과 그래프는 물가상승분을 제외한 **'오늘의 구매력(현재가치)'**으로 환산되어 표시됩니다.")
    st.markdown("---")

    # [입력 1] 기본 정보
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. 기본 정보")
        personality = st.selectbox("재무 성향", ["자산 형성형", "소비 집중형", "중도형"], 
                                   help="자산 형성형: 부자되기 / 소비 집중형: 다 쓰고 죽기 / 중도형: 밸런스")
        current_age = st.number_input("현재 나이", 30, 60, 40)
        death_age = st.number_input("목표 수명", 80, 120, 90)
    
    with c2:
        st.subheader("2. 금융자산 현황 (단위: 만 원)")
        current_asset = st.number_input("현재 금융자산", 0, value=20000, step=100, help="부동산 제외, 운용 가능한 자산 (예: 2억 = 20,000)")
        
        col_inc1, col_inc2 = st.columns([2, 1])
        monthly_income = col_inc1.number_input("월 수입 (세후)", 0, value=500, step=10)
        apply_income_inflation = col_inc2.checkbox("물가상승 반영", value=True, help="체크: 실질소득 유지 / 해제: 실질소득 감소")
        
        monthly_expense = st.number_input("월 기본 생활비", 0, value=300, step=10, help="숨만 쉬어도 나가는 기본 생활비")

    with c3:
        st.subheader("3. 금융 가정")
        expected_return = st.number_input("연 세후 수익률(%)", 0.0, 30.0, 6.0, step=0.5)
        inflation = st.number_input("물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1)
        volatility = st.number_input("변동성(%)", 0.0, 50.0, 20.0, step=1.0)

    st.markdown("---")

    # [입력 2] 은퇴 & 이벤트
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("4. 은퇴 목표")
        retire_mode = st.radio("은퇴 기준", ["나이 기준", "자산 기준"], horizontal=True)
        if retire_mode == "나이 기준":
            retire_age = st.number_input("은퇴 나이", current_age, 90, 60)
            retire_by_asset = 0
        else:
            retire_by_asset = st.number_input("목표 금융자산 (현재가치/만 원)", 0, value=150000, step=1000)
            retire_age = 100

    with col_right:
        st.subheader("5. 추가 수입/지출 (단위: 만 원)")
        st.markdown("**👉 수입은 (+), 지출은 (-)로 입력하세요**")
        tab1, tab2 = st.tabs(["💸 목돈 (결혼/주택)", "📅 기간 수입/지출 (연금/의료비)"])
        
        if 'lump_events' not in st.session_state: st.session_state.lump_events = []
        if 'recurring_events' not in st.session_state: 
            st.session_state.recurring_events = [
                {'start_age': 70, 'duration': 30, 'name': '국민연금(예상)', 'monthly_amount': 100}, 
                {'start_age': 75, 'duration': 25, 'name': '노후 의료비', 'monthly_amount': -50}
            ]

        with tab1:
            with st.form("lump", clear_on_submit=True):
                lc1, lc2, lc3 = st.columns([1,2,2])
                l_age = lc1.number_input("나이", current_age, death_age, 50)
                l_name = lc2.text_input("내용", "자녀결혼")
                l_amt = lc3.number_input("금액 (현재가치)", value=-5000, step=100)
                if st.form_submit_button("추가"):
                    st.session_state.lump_events.append({'age': l_age, 'name': l_name, 'amount': l_amt})
            if st.session_state.lump_events:
                st.table(pd.DataFrame(st.session_state.lump_events)[['name','age','amount']])
                if st.button("목돈 초기화"): st.session_state.lump_events = []

        with tab2:
            with st.form("recur", clear_on_submit=True):
                rc1, rc2, rc3, rc4 = st.columns([1,1,2,2])
                r_start = rc1.number_input("시작 나이", current_age, death_age, 45)
                r_dur = rc2.number_input("지속(년)", 1, 30, 4)
                r_name = rc3.text_input("내용", "대학등록금")
                r_amt = rc4.number_input("월 금액 (현재가치)", value=-100, step=10)
                if st.form_submit_button("추가"):
                    st.session_state.recurring_events.append({'start_age': r_start, 'duration': r_dur, 'name': r_name, 'monthly_amount': r_amt})
            if st.session_state.recurring_events:
                r_df = pd.DataFrame(st.session_state.recurring_events)
                r_df['끝'] = r_df['start_age'] + r_df['duration']
                st.table(r_df[['name', 'start_age', '끝', 'monthly_amount']])
                if st.button("기간 목록 초기화"): st.session_state.recurring_events = []

    # [실행]
    st.markdown("---")
    if st.button("🚀 5,000회 시뮬레이션 시작", type="primary", use_container_width=True):
        st.divider()
        n_sims = 5000
        
        params = {
            'n_simulations': n_sims,
            'current_age': current_age, 'death_age': death_age,
            'current_asset': current_asset, 'monthly_income': monthly_income,
            'apply_income_inflation': apply_income_inflation,
            'monthly_expense': monthly_expense, 'expected_return': expected_return,
            'volatility': volatility, 'inflation': inflation,
            'personality': personality, 'retire_age': retire_age,
            'retire_by_asset': retire_by_asset, 
            'lump_events': st.session_state.lump_events,
            'recurring_events': st.session_state.recurring_events
        }
        
        with st.spinner(f"5,000번의 시뮬레이션을 정밀 계산 중입니다..."):
            simulator = FinancialSimulator(params)
            years, sim_assets, sim_spending = simulator.run_monte_carlo(n_simulations=n_sims)
        
        # 통계 계산
        median_assets = np.median(sim_assets, axis=0) / 100000000
        top_10_assets = np.percentile(sim_assets, 90, axis=0) / 100000000
        bottom_10_assets = np.percentile(sim_assets, 10, axis=0) / 100000000
        top_25_assets = np.percentile(sim_assets, 75, axis=0) / 100000000
        bottom_25_assets = np.percentile(sim_assets, 25, axis=0) / 100000000
        
        ruin_prob = (np.sum(sim_assets[:, -1] <= 0) / n_sims) * 100
        
        g_col, d_col = st.columns([2.5, 1])
        
        with g_col:
            st.header(f"📊 {personality} 결과 (현재가치 기준)")
            k1, k2, k3 = st.columns(3)
            k1.metric("파산 확률", f"{ruin_prob:.1f}%", help="5,000번 중 0원이 된 비율")
            k2.metric("중위값 자산 (50%)", f"{median_assets[-1]:.2f}억 원")
            k3.metric("최악의 경우 (하위 10%)", f"{bottom_10_assets[-1]:.2f}억 원", delta_color="inverse")
            
            # 자산 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years + years[::-1],
                y=np.concatenate([top_10_assets, bottom_10_assets[::-1]]),
                fill='toself', fillcolor='rgba(0,176,246,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='상위 10% ~ 하위 10%'
            ))
            fig.add_trace(go.Scatter(
                x=years + years[::-1],
                y=np.concatenate([top_25_assets, bottom_25_assets[::-1]]),
                fill='toself', fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='상위 25% ~ 하위 25%'
            ))
            fig.add_trace(go.Scatter(
                x=years, y=median_assets,
                line=dict(color='rgb(0,176,246)', width=3),
                name='중위값 (예상)'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(title="<b>연도별 자산 확률 분포 (단위: 억 원, 현재가치 기준)</b>", xaxis_title="나이", yaxis_title="억 원", height=450, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # 생활비 그래프
            median_spending_man = np.median(sim_spending, axis=0) / 10000
            fig2 = px.line(x=years, y=median_spending_man, markers=True, title="<b>월 총지출가능액 (중위금/현재가치 기준)</b>")
            fig2.update_traces(line_color='#E74C3C')
            fig2.update_layout(yaxis=dict(tickformat=","), yaxis_title="만 원", xaxis_title="나이", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

        with d_col:
            st.subheader("📝 적용된 로직")
            st.success("""
            **1. 수익률 체감 (Safety First)**
            * 금융자산 **10억 원** 초과 시 연 수익률이 점진적으로 감소합니다.
            
            **2. 생애주기 생활비 (Lifecycle)**
            * 활동량 감소 반영: 60대(80%) → 70대(70%) → 80대~(60%)
            
            **3. 노후 의료비 (Medical)**
            * **75세부터 월 -50만 원(지출)** 자동 적용
            * *(기간 수입/지출 탭에서 수정 가능)*
            
            **4. 국민연금 (Pension)**
            * **70세부터 월 +100만 원(수입)** 자동 적용
            * *(기간 수입/지출 탭에서 수정 가능)*
            """)
            
            st.subheader("🎲 몬테카를로 분석")
            st.info(f"""
            **5,000회 시뮬레이션 완료**
            
            **📉 파산 확률: {ruin_prob:.1f}%**
            * {n_sims}개의 가상 시나리오 중 {int(ruin_prob/100 * n_sims)}번은 자산이 소진되었습니다.
            
            **📊 자산 범위 ({death_age}세 기준)**
            * **상위 10%:** {top_10_assets[-1]:.2f}억 원
            * **중위 50%:** {median_assets[-1]:.2f}억 원
            * **하위 10%:** {bottom_10_assets[-1]:.2f}억 원
            """)

if __name__ == '__main__':
    main()
