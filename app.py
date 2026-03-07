import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
# 2. 시뮬레이션 엔진 (Monte Carlo Engine) 
# -----------------------------------------------------------
class FinancialSimulator:
    def __init__(self, params):
        self.params = params

    def get_diminishing_return(self, base_return, current_asset_won):
        threshold = 1_000_000_000 
        if current_asset_won <= threshold:
            return base_return
        else:
            decay = 1 - (0.12 * np.log10(current_asset_won / threshold))
            return max(base_return * decay, 0.015)

    def run_monte_carlo(self, n_simulations=5000, override_expense=None):
        current_age = self.params['current_age']
        death_age = self.params['death_age']
        current_asset = self.params['current_asset'] * 10000

        base_monthly_income = (self.params['monthly_income'] * 10000) * 12
        apply_income_inflation = self.params['apply_income_inflation']
        
        expense_val = override_expense if override_expense is not None else self.params['monthly_expense']
        base_monthly_expense = expense_val * 10000 * 12

        base_return = self.params['expected_return'] / 100
        volatility = self.params['volatility'] / 100
        inflation = self.params['inflation'] / 100
        target_asset_won = self.params['retire_by_asset'] * 10000

        use_fat_tail = self.params.get('use_fat_tail', False)
        use_sorr_test = self.params.get('use_sorr_test', False)
        sorr_1 = self.params.get('sorr_1', -0.2)
        sorr_2 = self.params.get('sorr_2', -0.2)
        use_flex_spending = self.params.get('use_flex_spending', False)
        use_glide_path = self.params.get('use_glide_path', False)

        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)

        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_assets_nom = np.zeros((n_simulations, simulation_years))
        
        friction_cost = 0.002 

        precalc_discount = np.array([(1 + inflation) ** t for t in range(simulation_years)])
        precalc_extra_income = np.zeros(simulation_years)
        precalc_extra_expense = np.zeros(simulation_years)
        precalc_lump_sum = np.zeros(simulation_years)

        recurring_df = self.params['recurring_events']
        lump_df = self.params['lump_events']

        for t, age in enumerate(years):
            df_factor = precalc_discount[t]
            
            if not recurring_df.empty:
                for _, row in recurring_df.iterrows():
                    start_age = row['시작나이']
                    end_age = start_age + row['기간(년)']
                    if start_age <= age < end_age:
                        amt = (row['월금액(만원)'] * 10000) * 12 * df_factor
                        if amt > 0: precalc_extra_income[t] += amt
                        else: precalc_extra_expense[t] += abs(amt)
                        
            if not lump_df.empty:
                for _, row in lump_df.iterrows():
                    if row['나이'] == age:
                        precalc_lump_sum[t] += (row['금액(만원)'] * 10000) * df_factor

        for i in range(n_simulations):
            nominal_asset = current_asset
            initial_asset_val = current_asset if current_asset > 0 else 1

            if use_fat_tail:
                random_shock = np.random.standard_t(df=5, size=simulation_years) * (volatility / np.sqrt(5/3))
                base_random_returns = base_return + random_shock
            else:
                base_random_returns = np.random.normal(base_return, volatility, simulation_years)

            final_returns = []
            for t, age in enumerate(years):
                current_ret = base_random_returns[t] - friction_cost
                if use_glide_path and age > 60:
                    current_ret = current_ret - ((age - 60) * 0.0015)
                    current_ret = max(current_ret, inflation + 0.005)
                final_returns.append(current_ret)

            final_returns = np.array(final_returns)
            if use_sorr_test:
                final_returns[0] = sorr_1
                final_returns[1] = sorr_2

            path_assets_pv = []
            path_assets_nom = []
            is_retired = False

            for t, age in enumerate(years):
                discount_factor = precalc_discount[t]
                extra_income = precalc_extra_income[t]
                extra_expense = precalc_extra_expense[t]
                nominal_lump_sum = precalc_lump_sum[t]

                current_nominal_income = 0
                if not is_retired:
                    if age >= self.params['retire_age']: is_retired = True
                    elif target_asset_won > 0 and nominal_asset >= target_asset_won: is_retired = True
                    else:
                        current_nominal_income = base_monthly_income * discount_factor if apply_income_inflation else base_monthly_income

                total_income_annual = current_nominal_income + extra_income
                base_need_annual = base_monthly_expense * discount_factor + extra_expense
                nominal_actual_spending = base_need_annual

                if use_flex_spending and nominal_asset < initial_asset_val * 0.7:
                    nominal_actual_spending = min(nominal_actual_spending, base_need_annual)

                net_cashflow = total_income_annual - nominal_actual_spending
                adj_return = self.get_diminishing_return(final_returns[t], nominal_asset)
                
                gain_on_base = nominal_asset * adj_return
                gain_on_cashflow = net_cashflow * (adj_return / 2)

                nominal_asset = nominal_asset + gain_on_base + net_cashflow + gain_on_cashflow + nominal_lump_sum
                
                if nominal_asset < 0: nominal_asset = 0
                if nominal_asset > initial_asset_val: initial_asset_val = nominal_asset

                path_assets_pv.append(nominal_asset / discount_factor)
                path_assets_nom.append(nominal_asset)

            sim_assets_pv[i, :] = path_assets_pv
            sim_assets_nom[i, :] = path_assets_nom

        return years, sim_assets_pv, sim_assets_nom

    def run_hybrid_analysis(self, main_sims=5000, search_sims=1000, target_ruin_prob=5.0):
        original_expense = self.params['monthly_expense']
        
        years, main_pv, main_nom = self.run_monte_carlo(n_simulations=main_sims)
        base_ruin = (np.sum(main_pv[:, -1] <= 0) / main_sims) * 100
        
        safe_extra = 0
        if base_ruin <= target_ruin_prob:
            low = original_expense
            high = original_expense + 3000 
            best_expense = original_expense
            
            for _ in range(7):
                mid = (low + high) / 2
                _, pv, _ = self.run_monte_carlo(n_simulations=search_sims, override_expense=mid)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100
                if ruin <= target_ruin_prob:
                    best_expense = mid
                    low = mid 
                else:
                    high = mid 
            safe_extra = max(0, int(best_expense - original_expense))

        if safe_extra > 0:
            raw_incs = [0, int(safe_extra * 0.5), safe_extra, safe_extra + 50, safe_extra + 150, safe_extra + 300]
        else:
            raw_incs = [0, 50, 100, 200, 300]
            
        incs = sorted(list(set(raw_incs)))
        
        results = []
        for inc in incs:
            if inc == 0:
                ruin = base_ruin
            else:
                _, pv, _ = self.run_monte_carlo(n_simulations=search_sims, override_expense=original_expense + inc)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100
            
            if inc == safe_extra and safe_extra > 0:
                label = f"+{inc}만 (안전한계선 🚩)"
            elif inc == 0:
                label = "현재 유지"
            else:
                label = f"+{inc}만 원"
                
            results.append({'라벨': label, '추가액': inc, '파산 확률(%)': ruin})
            
        return years, main_pv, main_nom, safe_extra, base_ruin, pd.DataFrame(results)

# -----------------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Quant Asset Sim V34.1 (Stable)")

    st.title("💰 퀀트 기반 금융자산 시뮬레이터 V34.1")
    st.info("💡 실물 경제의 복잡성을 모델링하여 파산 확률을 구하고, 소비 수준에 따른 **리스크 스트레스 매트릭스**를 제공합니다.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("1. 기본 정보")
        current_age = st.number_input("현재 나이", 20, 80, 40)
        death_age = st.number_input("목표 수명", 80, 120, 90)

    with c2:
        st.subheader("2. 월 수입 및 지출")
        col_inc1, col_inc2 = st.columns([2, 1])
        monthly_income = col_inc1.number_input("월 수입 (세후/만원)", 0, value=500, step=10)
        apply_income_inflation = col_inc2.checkbox("수입 물가연동", value=True)
        monthly_expense = st.number_input("월 기본 지출 (만원)", 0, value=300, step=10)

    with c3:
        st.subheader("3. 자산 수익률 및 변동성")
        use_advanced_portfolio = st.checkbox("✅ 다중 계좌 구성 (고급)", value=False)
        inflation = st.number_input("물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1)

    if use_advanced_portfolio:
        st.markdown("##### 📊 다중 계좌 포트폴리오")
        if 'port_df' not in st.session_state:
            st.session_state.port_df = pd.DataFrame([
                {"자산명": "글로벌 주식", "금액(만원)": 15000, "수익률(%)": 9.0, "변동성(%)": 18.0},
                {"자산명": "안전 예금", "금액(만원)": 5000, "수익률(%)": 3.0, "변동성(%)": 2.0}
            ])
            
        edited_port_df = st.data_editor(st.session_state.port_df, num_rows="dynamic", width="stretch")
        
        # [데이터 정제 1] 표의 빈칸 에러 방지 (결측치 제거)
        clean_port_df = edited_port_df.dropna(subset=['금액(만원)', '수익률(%)', '변동성(%)'])
        
        total_asset_calc = clean_port_df['금액(만원)'].sum()
        if total_asset_calc > 0:
            weights = clean_port_df['금액(만원)'] / total_asset_calc
            avg_return = (clean_port_df['수익률(%)'] * weights).sum()
            weighted_vol = (clean_port_df['변동성(%)'] * weights).sum()
            final_volatility = weighted_vol * 0.8 
            
            st.success(f"**포트폴리오 합계:** {format_won(total_asset_calc)} / 가중 수익률 {avg_return:.2f}% / 포트폴리오 변동성 {final_volatility:.2f}%")
            current_asset = total_asset_calc
            expected_return = avg_return
            volatility = final_volatility
        else:
            current_asset = 0; expected_return = 0; volatility = 0
            st.warning("자산을 1개 이상 올바르게 입력해주세요.")
    else:
        with c3:
            current_asset = st.number_input("현재 금융자산 (만원)", 0, value=20000, step=100)
            expected_return = st.number_input("연 세후 수익률(%)", 0.0, 30.0, 7.0, step=0.5)
            volatility = st.number_input("변동성(%)", 0.0, 50.0, 15.0, step=1.0)

    st.markdown("---")
    with st.expander("🔥 **블랙 스완 & 행동재무학 리스크 (고급 설정)**", expanded=False):
        c_risk1, c_risk2 = st.columns(2)
        use_fat_tail = c_risk1.checkbox("📉 팻 테일(Fat Tail) 확률 적용", help="블랙 스완(Black Swan) 붕괴가 더 잦게 발생하는 T-분포를 사용합니다.")
        use_sorr_test = c_risk2.checkbox("😱 초기 폭락 (SORR) 스트레스 테스트", help="시뮬레이션 시작 직후 2년 연속 시장 대폭락 시나리오를 적용해 봅니다.")

        c_risk3, c_risk4 = st.columns(2)
        use_flex_spending = c_risk3.checkbox("🧠 생존 본능 (긴축 규칙)", value=True, help="포트폴리오가 30% 증발하면 추가 지출을 동결하고 필수 생활비만 씁니다.")
        use_glide_path = c_risk4.checkbox("📉 자산 노화 (TDF Glide Path)", value=True, help="60세 이후 매년 기대수익률을 0.15%p씩 낮춰 안전자산 비중 증가를 반영합니다.")

        if use_sorr_test:
            st.markdown("---")
            r_c1, r_c2 = st.columns(2)
            sorr_1 = r_c1.number_input("1년차 하락폭 (%)", -100.0, 0.0, -20.0, step=5.0) / 100
            sorr_2 = r_c2.number_input("2년차 하락폭 (%)", -100.0, 0.0, -20.0, step=5.0) / 100
        else:
            sorr_1 = -0.2; sorr_2 = -0.2

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.subheader("4. 은퇴 목표")
        retire_mode = st.radio("은퇴 기준", ["나이 기준", "자산 기준"], horizontal=True)
        retire_age = st.number_input("은퇴 나이", current_age, 90, 60) if retire_mode == "나이 기준" else 100
        retire_by_asset = st.number_input("목표 자산 (만원)", 0, value=150000) if retire_mode == "자산 기준" else 0

    with col_right:
        st.subheader("5. 이벤트성 추가 수입/지출")
        tab1, tab2 = st.tabs(["💸 일회성 목돈", "📅 기간성 수입/지출"])
        
        with tab1:
            if 'lump_df' not in st.session_state:
                st.session_state.lump_df = pd.DataFrame([{"나이": 55, "내용": "자녀결혼지원", "금액(만원)": -5000}])
            edited_lump_df = st.data_editor(st.session_state.lump_df, num_rows="dynamic", key="lump_editor", width="stretch")
            # [데이터 정제 2] 목돈 입력 표 빈칸 제거
            clean_lump_df = edited_lump_df.dropna(subset=['나이', '금액(만원)'])
            
        with tab2:
            if 'recur_df' not in st.session_state:
                st.session_state.recur_df = pd.DataFrame([
                    {"시작나이": 65, "기간(년)": 25, "내용": "국민연금", "월금액(만원)": 120},
                    {"시작나이": 75, "기간(년)": 15, "내용": "만성질환 의료비", "월금액(만원)": -30}
                ])
            edited_recur_df = st.data_editor(st.session_state.recur_df, num_rows="dynamic", key="recur_editor", width="stretch")
            # [데이터 정제 3] 기간성 입력 표 빈칸 제거
            clean_recur_df = edited_recur_df.dropna(subset=['시작나이', '기간(년)', '월금액(만원)'])

    if st.button("🚀 5,000회 연산 및 정밀 스트레스 테스트", type="primary"):
        st.divider()
        n_sims = 5000
        params = {
            'current_age': current_age, 'death_age': death_age, 'current_asset': current_asset,
            'monthly_income': monthly_income, 'apply_income_inflation': apply_income_inflation,
            'monthly_expense': monthly_expense, 'expected_return': expected_return,
            'volatility': volatility, 'inflation': inflation, 
            'retire_age': retire_age, 'retire_by_asset': retire_by_asset,
            'lump_events': clean_lump_df, # 정제된 데이터프레임 삽입
            'recurring_events': clean_recur_df, # 정제된 데이터프레임 삽입
            'use_fat_tail': use_fat_tail, 'use_sorr_test': use_sorr_test,
            'sorr_1': sorr_1, 'sorr_2': sorr_2,
            'use_flex_spending': use_flex_spending, 'use_glide_path': use_glide_path
        }
        
        with st.spinner("복잡계 연산 및 하이브리드 한계선(Goal Seek) 추적 중..."):
            simulator = FinancialSimulator(params)
            years, main_pv, main_nom, safe_extra, base_ruin, stress_df = simulator.run_hybrid_analysis(main_sims=n_sims, search_sims=1000)
            
            st.session_state['sim_results'] = {
                'years': years, 'pv': main_pv, 'nom': main_nom, 'n_sims': n_sims,
                'safe_extra': safe_extra, 'base_ruin': base_ruin, 'stress_df': stress_df
            }

    if 'sim_results' in st.session_state:
        res = st.session_state['sim_results']
        years, sim_assets_pv, sim_assets_nom, n_sims = res['years'], res['pv'], res['nom'], res['n_sims']
        safe_extra, base_ruin, stress_df = res['safe_extra'], res['base_ruin'], res['stress_df']

        median_pv = np.median(sim_assets_pv, axis=0) / 100000000
        top_10_pv = np.percentile(sim_assets_pv, 90, axis=0) / 100000000
        bottom_10_pv = np.percentile(sim_assets_pv, 10, axis=0) / 100000000
        
        median_nom = np.median(sim_assets_nom, axis=0) / 100000000
        top_10_nom = np.percentile(sim_assets_nom, 90, axis=0) / 100000000
        bottom_10_nom = np.percentile(sim_assets_nom, 10, axis=0) / 100000000

        target_display_age = retire_age if retire_mode == "나이 기준" else 60
        if target_display_age not in years: target_display_age = death_age
        idx_target = years.index(target_display_age)

        if safe_extra > 0:
            st.success(f"🎉 **안전 한계선 도출 완료:** 파산 확률 5%를 방어하는 최대 여유 자금은 **매월 +{safe_extra:,}만 원**입니다. 이 한계선을 넘기면 리스크가 어떻게 치솟는지 아래 차트에서 확인하세요.")
        else:
            st.error(f"⚠️ **안전 마진 없음:** 현재 기본 지출만으로도 파산 확률이 {base_ruin:.1f}%에 달해 위험 구역에 진입했습니다. 여유 생활비를 확보할 수 없습니다.")

        st.markdown("---")
        g_col, d_col = st.columns([2.5, 1.2])
        
        with g_col:
            colors = ['#27AE60' if val <= 5.01 else '#F1C40F' if val < 15 else '#E74C3C' for val in stress_df['파산 확률(%)']]
            
            fig_stress = go.Figure(data=[go.Bar(
                x=stress_df['라벨'],
                y=stress_df['파산 확률(%)'],
                marker_color=colors,
                text=[f"{val:.1f}%" for val in stress_df['파산 확률(%)']],
                textposition='auto'
            )])
            fig_stress.update_layout(
                title="<b>[리스크 매트릭스] 월 추가 지출에 따른 파산 확률 (한계선 돌파 시 위험 폭발)</b>",
                yaxis_title="파산 확률 (%)", height=320, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
            )
            fig_stress.add_hline(y=5, line_dash="dot", line_color="green", annotation_text="안전 방어선 (5%)")
            st.plotly_chart(fig_stress, use_container_width=True)

            st.markdown("##### 📊 메인 자산 궤적 (현재 유지 기준)")
            k1, k2, k3 = st.columns(3)
            k1.metric("기본 파산 확률", f"{base_ruin:.1f}%")
            k2.metric(f"{target_display_age}세 예상액", f"{median_pv[idx_target]:.2f}억 원", f"명목 {median_nom[idx_target]:.2f}억", delta_color="off")
            k3.metric("최악의 경우 (하위 10%)", f"{bottom_10_pv[idx_target]:.2f}억 원", f"명목 {bottom_10_nom[idx_target]:.2f}억", delta_color="off")

            chart_view = st.radio("보기 기준 전환:", ["현재가치 (구매력 기준)", "명목가치 (단순 금액 기준)"], horizontal=True)
            
            fig = go.Figure()
            if "현재" in chart_view:
                y_median, y_top, y_bot = median_pv, top_10_pv, bottom_10_pv
            else:
                y_median, y_top, y_bot = median_nom, top_10_nom, bottom_10_nom

            fig.add_trace(go.Scatter(x=years+years[::-1], y=np.concatenate([y_top, y_bot[::-1]]), fill='toself', fillcolor='rgba(0,176,246,0.1)', line=dict(color='rgba(255,255,255,0)'), name='신뢰구간(10~90%)'))
            fig.add_trace(go.Scatter(x=years, y=y_median, line=dict(color='#2E86C1', width=3), name='중앙값'))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_title="나이", yaxis_title="자산 (억 원)", height=400, template="plotly_white", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with d_col:
            st.subheader("💡 퀀트 로직: 하이브리드 엔진")
            st.success("""
            **1. 적응형 스트레스 매트릭스**
            * 백그라운드에서 최적의 '안전 한계선(목표 파산율 5%)'을 정밀 타겟팅(Goal Seek)하여 🚩깃발을 꽂습니다. 
            * 사용자는 이 한계선을 기점으로 지출을 욕심낼 때 포트폴리오가 얼마나 급격히 붕괴하는지 직관적으로 체험할 수 있습니다.

            **2. 수익률 체감 제약 조건**
            * 10억, 100억 단위로 커질수록 기대수익률을 점진 삭감하여, 워렌 버핏도 피할 수 없는 '자본 크기의 중력'을 모델링합니다.

            **3. 생존 본능 (긴축 규칙)**
            * 자산이 고점 대비 30% 증발하면 여유 생활비를 즉각 동결하고 최소 지출로 전환하여 펀드를 보호합니다.

            **4. 행동재무학적 리스크 투여**
            * 팻 테일(T-분포) 확률과 SORR(은퇴 초반 대폭락) 시나리오를 통해 최악의 경우를 대비합니다.
            """)

if __name__ == '__main__':
    main()
