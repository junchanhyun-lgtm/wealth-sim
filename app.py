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
# 2. 시뮬레이션 엔진
# -----------------------------------------------------------
class FinancialSimulator:
    def __init__(self, params):
        self.params = params
        
    def get_lifecycle_multiplier(self, age):
        if age < 60: return 1.0
        elif 60 <= age < 70: return 0.9
        elif 70 <= age < 80: return 0.85
        else: return 0.8

    def get_diminishing_return(self, base_return, current_asset_won):
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
        
        # 고급 옵션들
        use_fat_tail = self.params.get('use_fat_tail', False)
        use_sorr_test = self.params.get('use_sorr_test', False)
        sorr_1 = self.params.get('sorr_1', -0.2)
        sorr_2 = self.params.get('sorr_2', -0.2)
        use_flex_spending = self.params.get('use_flex_spending', False)
        use_ltc_shock = self.params.get('use_ltc_shock', False)
        
        # [V27 NEW] 글라이드 패스 & 상속
        use_glide_path = self.params.get('use_glide_path', False)
        target_legacy = self.params.get('target_legacy', 0) * 10000 # 만원 -> 원
        
        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)
        
        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_spending_pv = np.zeros((n_simulations, simulation_years))
        
        for i in range(n_simulations):
            nominal_asset = current_asset
            initial_asset_val = current_asset
            
            # 수익률 시퀀스 생성 (기본)
            if use_fat_tail:
                random_shock = np.random.standard_t(df=5, size=simulation_years) * (volatility / np.sqrt(5/3))
                base_random_returns = base_return + random_shock
            else:
                base_random_returns = np.random.normal(base_return, volatility, simulation_years)
            
            # [V27] 글라이드 패스 적용 (나이들수록 수익률/변동성 감소)
            final_returns = []
            for t, age in enumerate(years):
                current_ret = base_random_returns[t]
                
                if use_glide_path and age > 60:
                    # 60세 이후 매년 주식비중 감소 효과 (수익률 -0.1%p, 변동성 감소)
                    # 변동성 감소는 수익률 분포의 폭을 줄이는 것이나, 여기선 간단히 수익률을 보수적으로 깎음
                    aging_factor = (age - 60) * 0.0015 # 매년 0.15%p 씩 기대수익률 감소
                    current_ret = current_ret - aging_factor
                    # 최소한 물가상승률보단 높게 유지 (채권 등)
                    current_ret = max(current_ret, inflation + 0.005) 
                
                final_returns.append(current_ret)
            
            final_returns = np.array(final_returns)

            # SORR 테스트
            if use_sorr_test:
                final_returns[0] = sorr_1
                final_returns[1] = sorr_2

            path_assets_pv = []
            path_spending_pv = []
            is_retired = False
            
            for t, age in enumerate(years):
                discount_factor = (1 + inflation) ** t
                
                # A. 수입
                current_nominal_income = 0
                if not is_retired:
                    if age >= self.params['retire_age']: is_retired = True
                    elif target_asset_won > 0 and nominal_asset >= target_asset_won: is_retired = True
                    else:
                        if apply_income_inflation:
                            current_nominal_income = base_monthly_income * discount_factor
                        else:
                            current_nominal_income = base_monthly_income

                # B. 추가 수입/지출
                extra_income = 0
                extra_expense = 0
                if 'recurring_events' in self.params:
                    for event in self.params['recurring_events']:
                        end_age = event['start_age'] + event['duration']
                        if event['start_age'] <= age < end_age:
                            amt = (event['monthly_amount'] * 10000) * 12 * discount_factor
                            if amt > 0: extra_income += amt
                            else: extra_expense += abs(amt)

                # LTC (간병비)
                if use_ltc_shock and (death_age - 3 <= age <= death_age):
                    extra_expense += (300 * 10000) * 12 * discount_factor

                # C. 인출 상한액 (부채 예약 + 상속 예약)
                reserved_liabilities = 0
                
                # 1. 미래 목돈 지출 예약
                if personality == '소비 집중형' and 'lump_events' in self.params:
                    for event in self.params['lump_events']:
                        if event['age'] > age and event['amount'] < 0: 
                            time_diff = event['age'] - age
                            future_cost = abs(event['amount'] * 10000) * ((1 + inflation) ** (t + time_diff))
                            reserved_liabilities += future_cost / ((1 + inflation + 0.01) ** time_diff)
                
                # [V27] 2. 상속 목표액 예약 (Legacy Reservation)
                # 상속액은 사망 시점(death_age)의 목표이므로 현재가치로 할인하여 떼어둠
                if target_legacy > 0:
                    time_to_death = death_age - age
                    if time_to_death >= 0:
                        # 상속액도 안전하게 물가+알파로 할인해서 보존
                        reserved_liabilities += target_legacy * ((1 + inflation) ** t) / ((1 + inflation + 0.01) ** time_to_death)

                effective_asset = max(0, nominal_asset - reserved_liabilities)
                withdrawal_limit = 0

                if effective_asset > 0:
                    if personality == '소비 집중형':
                        remaining_years = death_age - age
                        planning_rate = min(base_return * 0.7, 0.05)
                        if remaining_years > 0:
                            withdrawal_limit = (effective_asset * planning_rate) / (1 - (1 + planning_rate) ** -remaining_years)
                    elif personality == '중도형':
                        withdrawal_limit = effective_asset * 0.04

                # D. 총 지출
                total_income_stream = current_nominal_income + extra_income
                lifecycle_factor = self.get_lifecycle_multiplier(age)
                nominal_basic_expense = (base_monthly_expense * 12) * discount_factor * lifecycle_factor

                if personality == '소비 집중형':
                    spending_capacity = total_income_stream + withdrawal_limit - extra_expense
                    nominal_actual_spending = max(spending_capacity, nominal_basic_expense * 0.5)
                elif personality == '중도형':
                    spending_capacity = total_income_stream + withdrawal_limit - extra_expense
                    nominal_actual_spending = max(spending_capacity, nominal_basic_expense)
                else: 
                    spending_capacity = total_income_stream - extra_expense
                    nominal_actual_spending = max(spending_capacity, nominal_basic_expense)

                # 긴축 재정
                if use_flex_spending and nominal_asset < initial_asset_val * 0.7: 
                    crisis_spending = nominal_basic_expense + extra_expense
                    nominal_actual_spending = min(nominal_actual_spending, crisis_spending)

                # E. 목돈
                nominal_lump_sum = 0
                if 'lump_events' in self.params:
                    for event in self.params['lump_events']:
                        if event['age'] == age:
                            nominal_lump_sum += (event['amount'] * 10000) * discount_factor

                # F. 자산 변동
                net_withdrawal = nominal_actual_spending - total_income_stream
                
                # [V27] 글라이드 패스가 적용된 수익률 사용
                adj_return = self.get_diminishing_return(final_returns[t], nominal_asset)
                gain = nominal_asset * adj_return
                
                nominal_asset = nominal_asset + gain - net_withdrawal + nominal_lump_sum
                if nominal_asset < 0: nominal_asset = 0
                if nominal_asset > initial_asset_val: initial_asset_val = nominal_asset

                # G. 저장
                path_assets_pv.append(nominal_asset / discount_factor)
                path_spending_pv.append(nominal_actual_spending / discount_factor / 12)
            
            sim_assets_pv[i, :] = path_assets_pv
            sim_spending_pv[i, :] = path_spending_pv
            
        return years, sim_assets_pv, sim_spending_pv

def main():
    st.set_page_config(layout="wide", page_title="Financial Asset Sim V27")
    
    st.title("💰 고정밀 금융자산 시뮬레이터 V27 (The Final)")
    st.info("💡 **현재가치 기준:** 모든 결과 그래프는 물가상승분을 제외한 **'오늘의 구매력(현재가치)'**으로 환산되어 표시됩니다.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. 기본 정보")
        personality = st.selectbox("재무 성향", ["자산 형성형", "소비 집중형", "중도형"], 
                                   help="자산 형성형: 부자되기 / 소비 집중형: 다 쓰고 죽기 / 중도형: 밸런스")
        current_age = st.number_input("현재 나이", 30, 60, 40)
        death_age = st.number_input("목표 수명", 80, 120, 90)
    
    with c2:
        st.subheader("2. 자산 및 수입")
        col_inc1, col_inc2 = st.columns([2, 1])
        monthly_income = col_inc1.number_input("월 수입 (세후)", 0, value=500, step=10)
        apply_income_inflation = col_inc2.checkbox("물가상승 반영", value=True)
        monthly_expense = st.number_input("월 기본 생활비", 0, value=300, step=10)

    with c3:
        st.subheader("3. 포트폴리오 & 리스크")
        use_advanced_portfolio = st.checkbox("✅ 고급 포트폴리오 (다중 계좌)", value=False)
        inflation = st.number_input("물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1)

    if use_advanced_portfolio:
        st.markdown("##### 📊 계좌별 전략 및 상관관계")
        diversification_effect = st.slider("🧩 자산 간 분산투자 효과 (Correlation)", 0.5, 1.0, 0.8, 0.05)
        
        p1, p2, p3 = st.columns(3)
        with p1:
            asset_a = st.number_input("자산 A (만 원)", 0, value=20000, step=100)
            ret_a = st.number_input("수익률 A (%)", 0.0, 100.0, 10.0, step=0.5)
            vol_a = st.number_input("변동성 A (%)", 0.0, 100.0, 15.0, step=1.0)
        with p2:
            asset_b = st.number_input("자산 B (만 원)", 0, value=0, step=100)
            ret_b = st.number_input("수익률 B (%)", 0.0, 100.0, 4.0, step=0.5)
            vol_b = st.number_input("변동성 B (%)", 0.0, 100.0, 5.0, step=1.0)
        with p3:
            asset_c = st.number_input("자산 C (만 원)", 0, value=0, step=100)
            ret_c = st.number_input("수익률 C (%)", 0.0, 100.0, 20.0, step=0.5)
            vol_c = st.number_input("변동성 C (%)", 0.0, 100.0, 30.0, step=1.0)

        total_asset_calc = asset_a + asset_b + asset_c
        if total_asset_calc > 0:
            w_a, w_b, w_c = asset_a/total_asset_calc, asset_b/total_asset_calc, asset_c/total_asset_calc
            avg_return = (ret_a * w_a) + (ret_b * w_b) + (ret_c * w_c)
            weighted_vol = (vol_a * w_a) + (vol_b * w_b) + (vol_c * w_c)
            final_volatility = weighted_vol * diversification_effect
            st.success(f"**💰 포트폴리오 요약:** 총 자산 {format_won(total_asset_calc)} / 수익률 {avg_return:.1f}% / 변동성 {final_volatility:.1f}%")
            current_asset = total_asset_calc
            expected_return = avg_return
            volatility = final_volatility
        else:
            current_asset = 0; expected_return = 0; volatility = 0
            st.warning("자산을 입력해주세요.")
    else:
        with c2:
            st.markdown("---")
            current_asset = st.number_input("현재 금융자산 (단일)", 0, value=20000, step=100)
        with c3:
            st.markdown("---")
            expected_return = st.number_input("연 세후 수익률(%)", 0.0, 30.0, 6.0, step=0.5)
            volatility = st.number_input("변동성(%)", 0.0, 50.0, 20.0, step=1.0)

    st.markdown("---")
    with st.expander("🔥 **블랙 스완 & 행동/생애 리스크 (고급 설정)**", expanded=False):
        c_risk1, c_risk2 = st.columns(2)
        use_fat_tail = c_risk1.checkbox("📉 팻 테일(Fat Tail) 적용", help="금융위기급 폭락이 더 자주 발생하는 분포를 사용합니다.")
        use_sorr_test = c_risk2.checkbox("😱 초반 폭락(SORR) 시나리오", help="시작 직후 2년 연속 폭락을 강제로 적용합니다.")
        
        c_risk3, c_risk4 = st.columns(2)
        use_flex_spending = c_risk3.checkbox("🧠 생존 본능(긴축) 발동", value=True, help="자산이 고점 대비 30% 하락하면, 지출을 '필수 생활비' 수준으로 강제 축소합니다.")
        use_ltc_shock = c_risk4.checkbox("🏥 말년 간병비 폭탄 (LTC)", value=True, help="사망 전 3년간 월 300만 원(현재가치)의 간병비 폭탄을 적용합니다.")
        
        c_risk5, c_risk6 = st.columns(2)
        # [V27 NEW] 글라이드 패스 & 상속
        use_glide_path = c_risk5.checkbox("📉 TDF 글라이드 패스 (자산 노화)", value=True, help="60세 이후 매년 기대수익률을 낮춰 안전 자산 비중 확대를 반영합니다.")
        target_legacy = c_risk6.number_input("🎁 목표 상속액 (현재가치/만 원)", 0, value=0, step=1000, help="이 금액은 소비하지 않고 끝까지 남겨둡니다.")

        if use_sorr_test:
            st.markdown("---")
            r_c1, r_c2 = st.columns(2)
            sorr_1 = r_c1.number_input("1년차 하락폭 (%)", -100.0, 0.0, -20.0, step=5.0) / 100
            sorr_2 = r_c2.number_input("2년차 하락폭 (%)", -100.0, 0.0, -20.0, step=5.0) / 100
        else:
            sorr_1 = -0.2; sorr_2 = -0.2

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("4. 은퇴 목표")
        retire_mode = st.radio("은퇴 기준", ["나이 기준", "자산 기준"], horizontal=True)
        retire_age = st.number_input("은퇴 나이", current_age, 90, 60) if retire_mode == "나이 기준" else 100
        retire_by_asset = st.number_input("목표 자산", 0, value=150000) if retire_mode == "자산 기준" else 0

    with col_right:
        st.subheader("5. 추가 수입/지출")
        tab1, tab2 = st.tabs(["💸 목돈", "📅 기간 지출"])
        if 'lump_events' not in st.session_state: st.session_state.lump_events = []
        if 'recurring_events' not in st.session_state: 
            st.session_state.recurring_events = [
                {'start_age': 70, 'duration': 30, 'name': '국민연금', 'monthly_amount': 100}, 
                {'start_age': 75, 'duration': 25, 'name': '의료비', 'monthly_amount': -50}
            ]
        with tab1:
            with st.form("lump", clear_on_submit=True):
                lc1, lc2, lc3 = st.columns([1,2,2])
                l_age = lc1.number_input("나이", current_age, death_age, 50)
                l_name = lc2.text_input("내용", "자녀결혼")
                l_amt = lc3.number_input("금액", value=-5000)
                if st.form_submit_button("추가"): st.session_state.lump_events.append({'age': l_age, 'name': l_name, 'amount': l_amt})
            if st.session_state.lump_events: 
                st.table(pd.DataFrame(st.session_state.lump_events)[['name','age','amount']])
                if st.button("목돈 초기화"): st.session_state.lump_events = []
        with tab2:
            with st.form("recur", clear_on_submit=True):
                rc1, rc2, rc3, rc4 = st.columns(4)
                r_start = rc1.number_input("시작", current_age, death_age, 60)
                r_dur = rc2.number_input("기간", 1, 30, 5)
                r_name = rc3.text_input("내용", "대학등록금")
                r_amt = rc4.number_input("월금액", value=-100)
                if st.form_submit_button("추가"): st.session_state.recurring_events.append({'start_age': r_start, 'duration': r_dur, 'name': r_name, 'monthly_amount': r_amt})
            if st.session_state.recurring_events: 
                st.table(pd.DataFrame(st.session_state.recurring_events)[['name','start_age','monthly_amount']])
                if st.button("기간 목록 초기화"): st.session_state.recurring_events = []

    if st.button("🚀 5,000회 시뮬레이션 시작", type="primary", use_container_width=True):
        st.divider()
        n_sims = 5000
        params = {
            'current_age': current_age, 'death_age': death_age, 'current_asset': current_asset,
            'monthly_income': monthly_income, 'apply_income_inflation': apply_income_inflation,
            'monthly_expense': monthly_expense, 'expected_return': expected_return,
            'volatility': volatility, 'inflation': inflation, 'personality': personality,
            'retire_age': retire_age, 'retire_by_asset': retire_by_asset, 
            'lump_events': st.session_state.lump_events, 'recurring_events': st.session_state.recurring_events,
            'use_fat_tail': use_fat_tail, 'use_sorr_test': use_sorr_test,
            'sorr_1': sorr_1, 'sorr_2': sorr_2,
            'use_flex_spending': use_flex_spending, 'use_ltc_shock': use_ltc_shock,
            'use_glide_path': use_glide_path, 'target_legacy': target_legacy
        }
        with st.spinner("초고속 연산 중..."):
            simulator = FinancialSimulator(params)
            years, sim_assets, sim_spending = simulator.run_monte_carlo(n_simulations=n_sims)
        
        median_assets = np.median(sim_assets, axis=0) / 100000000
        top_10_assets = np.percentile(sim_assets, 90, axis=0) / 100000000
        bottom_10_assets = np.percentile(sim_assets, 10, axis=0) / 100000000
        ruin_prob = (np.sum(sim_assets[:, -1] <= 0) / n_sims) * 100
        
        g_col, d_col = st.columns([2.5, 1])
        with g_col:
            st.header(f"📊 {personality} 결과 (현재가치)")
            k1, k2, k3 = st.columns(3)
            k1.metric("파산 확률", f"{ruin_prob:.1f}%")
            k2.metric("사망 시 자산 (중위값)", f"{median_assets[-1]:.2f}억 원")
            k3.metric("최악의 경우 (하위 10%)", f"{bottom_10_assets[-1]:.2f}억 원")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years+years[::-1], y=np.concatenate([top_10_assets, bottom_10_assets[::-1]]), fill='toself', fillcolor='rgba(0,176,246,0.1)', line=dict(color='rgba(255,255,255,0)'), name='범위(10~90%)'))
            fig.add_trace(go.Scatter(x=years, y=median_assets, line=dict(color='#2E86C1', width=3), name='중위값'))
            if target_legacy > 0:
                fig.add_hline(y=target_legacy/100000000, line_dash="dot", line_color="green", annotation_text="목표 상속액")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(title="<b>연도별 자산 추이</b>", xaxis_title="나이", yaxis_title="억 원", height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            median_spending = np.median(sim_spending, axis=0) / 10000
            fig2 = px.line(x=years, y=median_spending, markers=True, title="<b>월 총지출가능액 (중위금/현재가치 기준)</b>")
            fig2.update_traces(line_color='#E74C3C')
            fig2.update_layout(yaxis=dict(tickformat=","), yaxis_title="만 원", xaxis_title="나이")
            st.plotly_chart(fig2, use_container_width=True)
        with d_col:
            st.subheader("📝 적용된 4대 로직")
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
            # 리스크 경고
            if use_fat_tail: st.warning("⚠️ **팻 테일(Fat Tail) 적용됨**")
            if use_sorr_test: st.error(f"🔥 **SORR 테스트 적용됨** ({sorr_1*100:.0f}%, {sorr_2*100:.0f}%)")
            if use_glide_path: st.info("📉 **글라이드 패스(TDF) 작동 중** (나이들수록 안전 자산 전환)")
            if target_legacy > 0: st.success(f"🎁 **상속 예약:** {format_won(target_legacy/10000)} 보존 목표")

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
