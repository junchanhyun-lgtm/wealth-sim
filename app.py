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
# 2. 시뮬레이션 엔진 (Monte Carlo Engine) - 기존 로직 100% 유지
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

    def run_monte_carlo(self, n_simulations=5000, override_extra_margin=0):
        current_age = self.params['current_age']
        death_age = self.params['death_age']
        current_asset = self.params['current_asset'] * 10000

        base_monthly_income = (self.params['monthly_income'] * 10000) * 12
        apply_income_inflation = self.params['apply_income_inflation']
        base_monthly_expense = self.params['monthly_expense'] * 10000 * 12

        base_return = self.params['expected_return'] / 100
        volatility = self.params['volatility'] / 100
        inflation = self.params['inflation'] / 100
        target_asset_won = self.params['retire_by_asset'] * 10000

        use_fat_tail = self.params.get('use_fat_tail', False)
        use_sorr_test = self.params.get('use_sorr_test', False)
        use_inflation_shock = self.params.get('use_inflation_shock', False)
        use_flex_spending = self.params.get('use_flex_spending', False)
        use_glide_path = self.params.get('use_glide_path', False)
        dwz_mode = self.params.get('dwz_mode', False)
        friction_cost = self.params.get('tax_fee_rate', 0.5) / 100

        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)
        retire_idx = max(0, self.params['retire_age'] - current_age)

        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_assets_nom = np.zeros((n_simulations, simulation_years))

        inflation_matrix = np.full((n_simulations, simulation_years), inflation)
        if use_inflation_shock:
            max_start = min(simulation_years - 3, retire_idx + 9)
            if max_start < retire_idx: max_start = retire_idx
            shock_starts = np.random.randint(retire_idx, max_start + 1, size=n_simulations)
            for i in range(n_simulations):
                start = shock_starts[i]
                end = min(simulation_years, start + 3)
                inflation_matrix[i, start:end] = 0.07

        discount_factors = np.ones((n_simulations, simulation_years))
        if simulation_years > 1:
            discount_factors[:, 1:] = np.cumprod(1 + inflation_matrix[:, :-1], axis=1)

        pv_extra_income = np.zeros(simulation_years)
        pv_extra_expense = np.zeros(simulation_years)
        pv_lump_sum = np.zeros(simulation_years)

        recurring_df = self.params['recurring_events']
        lump_df = self.params['lump_events']

        for t, age in enumerate(years):
            if not recurring_df.empty:
                for _, row in recurring_df.iterrows():
                    if row['시작나이'] <= age < row['시작나이'] + row['기간(년)']:
                        amt_val = abs(row['월금액(만원)']) * 10000 * 12
                        if row['유형'] == '수입': pv_extra_income[t] += amt_val
                        else: pv_extra_expense[t] += amt_val
            if not lump_df.empty:
                for _, row in lump_df.iterrows():
                    if row['나이'] == age:
                        amt_val = abs(row['금액(만원)']) * 10000
                        if row['유형'] == '수입': pv_lump_sum[t] += amt_val
                        else: pv_lump_sum[t] -= amt_val

        for i in range(n_simulations):
            nominal_asset = current_asset
            high_water_mark = current_asset if current_asset > 0 else 1

            if use_fat_tail:
                random_shock = np.random.standard_t(df=5, size=simulation_years) * (volatility / np.sqrt(5/3))
                base_random_returns = base_return + random_shock
            else:
                base_random_returns = np.random.normal(base_return, volatility, simulation_years)

            if use_sorr_test and simulation_years > retire_idx + 1:
                worst_indices = np.argsort(base_random_returns)[:2]
                target_indices = [retire_idx, retire_idx + 1]

                temp_returns = base_random_returns.copy()
                worst_vals = temp_returns[worst_indices]
                target_vals = temp_returns[target_indices]

                temp_returns[worst_indices] = target_vals
                temp_returns[target_indices] = worst_vals
                base_random_returns = temp_returns

            final_returns = []
            for t, age in enumerate(years):
                current_ret = base_random_returns[t] - friction_cost
                if use_glide_path and age > 60:
                    current_ret = current_ret - ((age - 60) * 0.0015)
                    current_ret = max(current_ret, inflation + 0.005)
                final_returns.append(current_ret)

            final_returns = np.array(final_returns)

            path_assets_pv = []
            path_assets_nom = []
            is_retired = False

            for t, age in enumerate(years):
                df_factor = discount_factors[i, t]
                extra_income = pv_extra_income[t] * df_factor
                extra_expense = pv_extra_expense[t] * df_factor
                nominal_lump_sum = pv_lump_sum[t] * df_factor

                current_nominal_income = 0
                if not is_retired:
                    if age >= self.params['retire_age']: is_retired = True
                    elif target_asset_won > 0 and nominal_asset >= target_asset_won: is_retired = True
                    else:
                        current_nominal_income = base_monthly_income * df_factor if apply_income_inflation else base_monthly_income

                if nominal_asset > high_water_mark:
                    high_water_mark = nominal_asset

                decay_factor = 1.0
                base_yolo_expense = override_extra_margin * 10000 * 12

                if dwz_mode:
                    if age >= 60:
                        decay_factor = max(0.70, (1 - 0.015) ** (age - 60))
                    if age >= 70:
                        base_yolo_expense = 0

                yolo_multiplier = 1.0
                if use_flex_spending:
                    drawdown = (high_water_mark - nominal_asset) / high_water_mark if high_water_mark > 0 else 0
                    if drawdown >= 0.30:
                        yolo_multiplier = 0.0
                    elif drawdown > 0.10:
                        yolo_multiplier = 1.0 - ((drawdown - 0.10) / 0.20)

                actual_yolo_expense = base_yolo_expense * yolo_multiplier
                base_expense_applied = base_monthly_expense * decay_factor
                target_lifestyle_annual_pv = base_expense_applied + actual_yolo_expense

                total_income_annual = current_nominal_income + extra_income
                nominal_actual_spending = (target_lifestyle_annual_pv * df_factor) + extra_expense

                net_cashflow = total_income_annual - nominal_actual_spending
                adj_return = self.get_diminishing_return(final_returns[t], nominal_asset)

                gain_on_base = nominal_asset * adj_return
                gain_on_cashflow = net_cashflow * (adj_return / 2)

                nominal_asset = nominal_asset + gain_on_base + net_cashflow + gain_on_cashflow + nominal_lump_sum

                if nominal_asset < 0: nominal_asset = 0

                path_assets_pv.append(nominal_asset / df_factor)
                path_assets_nom.append(nominal_asset)

            sim_assets_pv[i, :] = path_assets_pv
            sim_assets_nom[i, :] = path_assets_nom

        return years, sim_assets_pv, sim_assets_nom

    def run_hybrid_analysis(self, main_sims=5000, search_sims=1000):
        is_dwz = self.params.get('dwz_mode', False)
        target_ruin_prob = 15.0 if is_dwz else 5.0
        current_age = self.params['current_age']
        retire_age = self.params['retire_age']

        years, main_pv, main_nom = self.run_monte_carlo(n_simulations=main_sims, override_extra_margin=0)
        base_ruin = (np.sum(main_pv[:, -1] <= 0) / main_sims) * 100

        earliest_fire_age = retire_age
        if base_ruin <= target_ruin_prob and retire_age > current_age:
            for test_age in range(current_age, retire_age):
                temp_params = self.params.copy()
                temp_params['retire_age'] = test_age
                temp_sim = FinancialSimulator(temp_params)
                _, test_pv, _ = temp_sim.run_monte_carlo(n_simulations=500, override_extra_margin=0)
                test_ruin = (np.sum(test_pv[:, -1] <= 0) / 500) * 100
                if test_ruin <= target_ruin_prob:
                    earliest_fire_age = test_age
                    break

        safe_extra = 0
        if base_ruin <= target_ruin_prob:
            low, high = 0, 5000
            best_extra = 0
            for _ in range(8):
                mid = (low + high) / 2
                _, pv, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=mid)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100
                if ruin <= target_ruin_prob:
                    best_extra, low = mid, mid
                else:
                    high = mid
            safe_extra = int(best_extra)

        if safe_extra > 0:
            raw_incs = [0, int(safe_extra * 0.5), safe_extra, safe_extra + 50, safe_extra + 150, safe_extra + 300]
        else:
            raw_incs =

        incs = sorted(list(set(raw_incs)))

        results = []
        for inc in incs:
            if inc == 0: ruin = base_ruin
            else:
                _, pv, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=inc)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100

            if inc == safe_extra and safe_extra > 0: label = f"+{inc}만 (안전한계선 🚩)"
            elif inc == 0: label = "현재 유지 (0만 원)"
            else: label = f"+{inc}만 원"
            results.append({'라벨': label, '추가액': inc, '파산 확률(%)': ruin})

        return years, main_pv, main_nom, safe_extra, base_ruin, pd.DataFrame(results), target_ruin_prob, earliest_fire_age

    def run_sensitivity(self, base_ruin, sims=2000):
        scenarios = [
            ("수익률 -1%p 하락", 'expected_return', -1.0),
            ("수익률 +1%p 상승", 'expected_return', 1.0),
            ("물가상승률 +1%p 폭등", 'inflation', 1.0),
            ("기본생활비 +10% 증가", 'monthly_expense', self.params['monthly_expense'] * 0.1),
        ]

        sens_results = []
        for label, key, delta in scenarios:
            temp_params = self.params.copy()
            temp_params[key] += delta
            temp_sim = FinancialSimulator(temp_params)
            _, pv, _ = temp_sim.run_monte_carlo(n_simulations=sims, override_extra_margin=0)
            ruin = (np.sum(pv[:, -1] <= 0) / sims) * 100
            impact = ruin - base_ruin
            sens_results.append({"시나리오": label, "파산확률": ruin, "충격(%)": impact})

        return pd.DataFrame(sens_results)

# -----------------------------------------------------------
# 3. Streamlit UI (V45.0 Hyper-Personalized Edition)
# -----------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="나만의 은퇴 관제탑 (V45.0)")

    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); background-color: #ffffff;
        }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }
        div.stAlert > div { border-radius: 10px; }
        [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; }
        /* YOLO 대시보드 박스 스타일 */
        .yolo-box {
            background-color: #f0fdf4; border: 2px solid #22c55e; 
            border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;
        }
        .yolo-title { color: #166534; font-size: 1.4rem; font-weight: 700; margin: 0; }
        .yolo-value { color: #15803d; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎯 나만의 퀀트 은퇴 관제탑 (V45.0)")
    st.info("💡 고객님의 9.7억 포트폴리오(통합 수익률 14%, 변동성 17.5%)에 완벽하게 맞춰진 초개인화 엔진입니다.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    # 기본값(Default) 하드코딩 적용: 42세, 9.7억, 600만, 14.0%, 17.5%
    with c1:
        with st.container(border=True):
            st.subheader("👤 1. 기본 정보")
            current_age = st.number_input("현재 나이", 20, 80, 42, key='in_age') # 42세로 변경
            death_age = st.number_input("목표 수명", 80, 120, 90, key='in_death')

    with c2:
        with st.container(border=True):
            st.subheader("💵 2. 기본 수입 및 지출")
            col_inc1, col_inc2 = st.columns()
            monthly_income = col_inc1.number_input("월 수입 (세후/만원)", 0, value=500, step=10, key='in_income')
            apply_income_inflation = col_inc2.checkbox("수입 물가연동", value=False, key='in_inc_inf')
            monthly_expense = st.number_input("월 필수 기본 지출 (만원)", 0, value=600, step=10, key='in_expense') # 600만 고정

    with c3:
        with st.container(border=True):
            # 복잡한 다중계좌 표 완전 삭제, 단일 통합계좌 모드로 직관적 세팅
            st.subheader("📈 3. 자산 및 거시 지표")
            current_asset = st.number_input("현재 금융자산 (만원)", 0, value=97000, step=100, key='in_asset') # 97000만(9.7억) 고정
            col_ret1, col_ret2 = st.columns(2)
            expected_return = col_ret1.number_input("통합 세후 수익률(%)", 0.0, 30.0, 14.0, step=0.5, key='in_ret') # 14.0% 고정
            volatility = col_ret2.number_input("통합 변동성(%)", 0.0, 50.0, 17.5, step=0.5, key='in_vol') # 17.5% 고정

            st.markdown("---")
            col_ret3, col_ret4 = st.columns(2)
            inflation = col_ret3.number_input("평시 물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1, key='in_inf')
            tax_fee_rate = col_ret4.number_input("마찰 비용/수수료(연%)", 0.0, 5.0, 0.5, step=0.1, key='in_tax')

    st.markdown("---")

    with st.expander("🔥 **블랙 스완 & 다이 위드 제로 (고급 설정)**", expanded=False): # 기본적으로 닫아두어 깔끔하게 유지
        with st.container(border=True):
            st.markdown("##### 🏖️ 라이프스타일 퀀트 최적화")
            dwz_mode = st.checkbox("🔥 Die with Zero 최적화 (투트랙 욜로 + 15% 리스크 허용)", value=True, key='in_dwz')

        with st.container(border=True):
            st.markdown("##### 🚨 블랙 스완 방어선 (치명적 위기 테스트)")
            c_risk1, c_risk2 = st.columns(2)
            use_fat_tail = c_risk1.checkbox("📉 팻 테일(Fat Tail) 확률 적용", value=True, key='in_fat')
            use_sorr_test = c_risk2.checkbox("😱 은퇴 직후 2년 폭락 (SORR 셔플링)", value=True, key='in_sorr')

            c_risk3, c_risk4 = st.columns(2)
            use_flex_spending = c_risk3.checkbox("🧠 점진적 생존 본능 (긴축 규칙)", value=True, key='in_flex')
            use_glide_path = c_risk4.checkbox("📉 자산 노화 (TDF Glide Path)", value=True, key='in_glide')

            use_inflation_shock = st.checkbox("🔥 스태그플레이션 발작 충격 (구매력 파괴)", value=True, key='in_shock')

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        with st.container(border=True):
            st.subheader("🎯 4. 은퇴 목표")
            retire_mode = st.radio("은퇴 기준", ["나이 기준", "자산 기준"], horizontal=True, key='in_ret_mode')
            retire_age = st.number_input("은퇴 나이", current_age, 90, 60, key='in_ret_age') if retire_mode == "나이 기준" else 100
            retire_by_asset = st.number_input("목표 자산 (만원)", 0, value=150000, key='in_ret_asset') if retire_mode == "자산 기준" else 0

    with col_right:
        with st.container(border=True):
            st.subheader("📱 5. 이벤트성 추가 수입/지출")
            tab1, tab2 = st.tabs(["💸 일회성 목돈", "📅 기간성 수입/지출 (연금 포함)"])

            with tab1:
                # session_state 덮어쓰기 로직을 삭제하여 튕김/초기화 현상 완벽 해결
                if 'lump_df' not in st.session_state:
                    st.session_state.lump_df = pd.DataFrame([
                        {"나이": 45, "유형": "지출", "내용": "이벤트(예:차량)", "금액(만원)": 5000}
                    ])
                edited_lump_df = st.data_editor(st.session_state.lump_df, num_rows="dynamic", key="lump_editor", use_container_width=True,
                    column_config={"유형": st.column_config.SelectboxColumn("유형", options=["수입", "지출"], required=True),
                                   "금액(만원)": st.column_config.NumberColumn("금액(만원)", min_value=0)})
                clean_lump_df = edited_lump_df.dropna(subset=['나이', '유형', '금액(만원)'])

            with tab2:
                if 'recur_df' not in st.session_state:
                    st.session_state.recur_df = pd.DataFrame([
                        {"시작나이": 70, "기간(년)": 30, "유형": "수입", "내용": "국민연금", "월금액(만원)": 100, "확정연금": True},
                        {"시작나이": 70, "기간(년)": 30, "유형": "수입", "내용": "주택연금", "월금액(만원)": 200, "확정연금": True}
                    ])
                edited_recur_df = st.data_editor(st.session_state.recur_df, num_rows="dynamic", key="recur_editor", use_container_width=True,
                    column_config={
                        "유형": st.column_config.SelectboxColumn("유형", options=["수입", "지출"], required=True),
                        "월금액(만원)": st.column_config.NumberColumn("월금액(만원)", min_value=0),
                        "확정연금": st.column_config.CheckboxColumn("확정연금(방어율용)")
                    })
                clean_recur_df = edited_recur_df.dropna(subset=['시작나이', '기간(년)', '유형', '월금액(만원)'])

    if st.button("🚀 10,000회 연산 및 정밀 스트레스 테스트 시작", type="primary", use_container_width=True):
        st.divider()
        n_sims = 10000
        params = {
            'current_age': current_age, 'death_age': death_age, 'current_asset': current_asset,
            'monthly_income': monthly_income, 'apply_income_inflation': apply_income_inflation,
            'monthly_expense': monthly_expense, 'expected_return': expected_return,
            'volatility': volatility, 'inflation': inflation, 'tax_fee_rate': tax_fee_rate,
            'retire_age': retire_age, 'retire_by_asset': retire_by_asset,
            'lump_events': clean_lump_df, 'recurring_events': clean_recur_df,
            'use_fat_tail': use_fat_tail, 'use_sorr_test': use_sorr_test,
            'use_inflation_shock': use_inflation_shock,
            'use_flex_spending': use_flex_spending, 'use_glide_path': use_glide_path,
            'dwz_mode': dwz_mode
        }

        with st.spinner("복잡계 퀀트 엔진 및 V45.0 방어력 계산 중..."):
            simulator = FinancialSimulator(params)
            years, main_pv, main_nom, safe_extra, base_ruin, stress_df, t_ruin, earliest_fire = simulator.run_hybrid_analysis(main_sims=n_sims, search_sims=1000)
            sens_df = simulator.run_sensitivity(base_ruin, sims=2000)

            total_pension = 0
            if '확정연금' in clean_recur_df.columns:
                pension_df = clean_recur_df[(clean_recur_df['유형'] == '수입') & (clean_recur_df['확정연금'] == True)]
                total_pension = pension_df['월금액(만원)'].sum()
            defense_rate = (total_pension / monthly_expense * 100) if monthly_expense > 0 else 0

            bottom_10_pv = np.percentile(main_pv, 10, axis=0)
            depletion_age = "고갈 안 됨"
            if np.any(bottom_10_pv <= 0):
                depletion_idx = np.argmax(bottom_10_pv <= 0)
                depletion_age = f"{years[depletion_idx]}세"

            peak = bottom_10_pv
            current_peak_idx = 0
            max_dd = 0
            mdd_start_idx = 0
            mdd_end_idx = 0

            for t in range(len(bottom_10_pv)):
                if bottom_10_pv[t] > peak:
                    peak = bottom_10_pv[t]
                    current_peak_idx = t
                else:
                    if peak > 0:
                        dd = (peak - bottom_10_pv[t]) / peak
                        if dd > max_dd:
                            max_dd = dd
                            mdd_start_idx = current_peak_idx
                            mdd_end_idx = t

            recovery_years = "회복 불가"
            if max_dd > 0:
                for t in range(mdd_end_idx, len(bottom_10_pv)):
                    if bottom_10_pv[t] >= bottom_10_pv[mdd_start_idx]:
                        recovery_years = f"{t - mdd_start_idx}년"
                        break

            # -----------------------------------------------------------
            # [목표 달성] YOLO 예산 형광펜 하이라이트 대시보드 출력
            # -----------------------------------------------------------
            if safe_extra > 0:
                st.markdown(f"""
                <div class='yolo-box'>
                    <p class='yolo-title'>💰 파산 확률 {t_ruin:.0f}% 방어선</p>
                    <p class='yolo-value'>이번 달 추가로 써도 되는 욜로(YOLO) 예산 = 월 {safe_extra:,}만 원</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ **안전 마진 없음:** 현재 필수 지출 수준으로도 자산이 고갈될 위험이 높습니다.")

            # -----------------------------------------------------------
            # [더 현실적으로] 시뮬레이터 내부 '숨은 삭감 로직' 시각화
            # -----------------------------------------------------------
            decayed_return_10b = 14.0 * (1 - (0.12 * np.log10(200000 / 100000))) # 자산 20억 가정 (threshold 10억)
            decayed_return_75age = max(14.0 - ((75 - 60) * 0.15), 0)

            with st.expander("🔍 현재 시스템에 적용된 가혹한 현실 조건 (숨은 페널티 확인)", expanded=True):
                st.info(f"""
                **고객님의 입력값({expected_return}%)은 미래에 그대로 적용되지 않고, 엔진 내부에서 다음과 같이 강제로 깎여서 계산되고 있습니다.**
                
                * ⚖️ **규모의 저주 (슬리피지 페널티):** 자산이 10억을 초과하는 순간부터 수익률이 하락합니다. (예: 20억 도달 시 기대수익률은 **약 {decayed_return_10b:.1f}%**로 강제 하향 적용 중)
                * ⏳ **자산 노화 (TDF 글라이드 패스):** 60세 이후부터 매년 0.15%p씩 기본 수익률이 깎입니다. (예: 75세 시점에는 기대수익률이 **최대 {decayed_return_75age:.1f}%**로 강제 하향 적용 중)
                
                👉 **즉, 지금 도출된 욜로(YOLO) 예산은 먼 미래의 퀀트 알파 붕괴와 예금 이자에 가까운 낮은 수익률까지 모두 얻어맞고도 살아남은 '진짜 안전한 돈'입니다.**
                """)

            st.session_state['sim_results'] = {
                'years': years, 'pv': main_pv, 'nom': main_nom, 'n_sims': n_sims,
                'safe_extra': safe_extra, 'base_ruin': base_ruin, 'stress_df': stress_df,
                'sens_df': sens_df,
                'dwz_mode': dwz_mode, 't_ruin': t_ruin, 'defense_rate': defense_rate,
                'depletion_age': depletion_age, 'lump_df': clean_lump_df,
                'earliest_fire': earliest_fire, 'max_dd': max_dd * 100, 'recovery_years': recovery_years
            }

    if 'sim_results' in st.session_state:
        res = st.session_state['sim_results']
        years, sim_assets_pv, sim_assets_nom = res['years'], res['pv'], res['nom']
        safe_extra, base_ruin, stress_df = res['safe_extra'], res['base_ruin'], res['stress_df']
        is_dwz, target_ruin = res['dwz_mode'], res['t_ruin']
        defense_rate, depletion_age = res['defense_rate'], res['depletion_age']
        res_lump_df, earliest_fire = res['lump_df'], res['earliest_fire']
        max_dd, recovery_years = res['max_dd'], res['recovery_years']

        median_pv = np.median(sim_assets_pv, axis=0) / 100000000
        top_10_pv = np.percentile(sim_assets_pv, 90, axis=0) / 100000000
        bottom_10_pv = np.percentile(sim_assets_pv, 10, axis=0) / 100000000

        median_nom = np.median(sim_assets_nom, axis=0) / 100000000
        top_10_nom = np.percentile(sim_assets_nom, 90, axis=0) / 100000000
        bottom_10_nom = np.percentile(sim_assets_nom, 10, axis=0) / 100000000

        target_display_age = retire_age if retire_mode == "나이 기준" else 60
        if target_display_age not in years: target_display_age = death_age
        idx_target = years.index(target_display_age)

        if earliest_fire < retire_age:
            st.success(f"🏃‍♂️ **조기 은퇴 당기기 가능:** 현재 지출만 유지하신다면 목표 은퇴 나이({retire_age}세)보다 훨씬 이른 **{earliest_fire}세**에 당장 사표를 던지셔도 파산하지 않습니다!")

        if defense_rate > 0:
            st.info(f"🛡️ **노후 확정 생계 방어율: {defense_rate:.1f}%** (체크된 연금 수입 합계액이 매월 필수 지출액의 {defense_rate:.1f}%를 방어합니다.)")

        st.markdown("<br>", unsafe_allow_html=True)
        g_col, d_col = st.columns([2.5, 1.2])

        with g_col:
            colors = ['#27AE60' if val <= target_ruin + 0.01 else '#F1C40F' if val < target_ruin + 10 else '#E74C3C' for val in stress_df['파산 확률(%)']]
            fig_stress = go.Figure(data=[go.Bar(x=stress_df['라벨'], y=stress_df['파산 확률(%)'], marker_color=colors, text=[f"{val:.1f}%" for val in stress_df['파산 확률(%)']], textposition='auto')])
            title_suffix = f"(70세 컷오프 & {target_ruin:.0f}% 허용)" if is_dwz else f"(평생 고정 & {target_ruin:.0f}% 방어)"
            fig_stress.update_layout(
                title=f"<b>월 여유 생활비별 파산 확률 {title_suffix}</b>",
                yaxis_title="파산 확률 (%)", height=300,
                plot_bgcolor='rgba(252, 252, 252, 1)', paper_bgcolor='rgba(255, 255, 255, 1)',
                margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(fixedrange=True, showgrid=False), yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#f0f0f0')
            )
            fig_stress.add_hline(y=target_ruin, line_dash="dot", line_color="green", annotation_text=f"안전 방어선 ({target_ruin:.0f}%)")
            with st.container(border=True):
                st.plotly_chart(fig_stress, use_container_width=True)

            st.markdown("##### 📈 메인 자산 궤적 (현재 지출 유지 시)")
            k1, k2, k3 = st.columns(3)
            k1.metric("기본 파산 확률", f"{base_ruin:.1f}%", f"최악 고갈: {depletion_age}", delta_color="off")
            k2.metric(f"{target_display_age}세 예상액", f"{median_pv[idx_target]:.2f}억 원", f"명목 {median_nom[idx_target]:.2f}억", delta_color="off")
            k3.metric("최악의 경우 (하위 10%)", f"{bottom_10_pv[idx_target]:.2f}억 원", f"MDD: -{max_dd:.1f}% (회복 {recovery_years})", delta_color="off")

            chart_view = st.radio("보기 기준 전환:", ["현재가치 (구매력 기준)", "명목가치 (단순 금액 기준)"], horizontal=True)

            fig = go.Figure()
            if "현재" in chart_view:
                y_median, y_top, y_bot = median_pv, top_10_pv, bottom_10_pv
            else:
                y_median, y_top, y_bot = median_nom, top_10_nom, bottom_10_nom

            fig.add_trace(go.Scatter(x=years+years[::-1], y=np.concatenate([y_top, y_bot[::-1]]), fill='toself', fillcolor='rgba(46, 134, 193, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='신뢰구간(10~90%)', hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=years, y=y_median, line=dict(color='#2E86C1', width=3), name='중앙값', hovertemplate='%{y:.2f}억 원<extra></extra>'))
            fig.add_trace(go.Scatter(x=years, y=y_bot, line=dict(color='#E74C3C', width=2, dash='dot'), name='하위 10%', hovertemplate='%{y:.2f}억 원<extra></extra>'))

            fig.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)

            if retire_age in years:
                fig.add_vline(x=retire_age, line_dash="dash", line_color="#95a5a6", annotation_text="은퇴")
            if is_dwz and 70 in years:
                fig.add_vline(x=70, line_dash="dot", line_color="#9b59b6", annotation_text="YOLO 종료")

            for _, row in res_lump_df.iterrows():
                if row['금액(만원)'] >= 10000 and row['나이'] in years:
                    fig.add_vline(x=row['나이'], line_dash="dot", line_color="#f39c12", annotation_text=row['내용'])

            fig.update_layout(
                xaxis_title="나이", yaxis_title="자산 (억 원)", height=450,
                plot_bgcolor='rgba(252, 252, 252, 1)', paper_bgcolor='rgba(255, 255, 255, 1)',
                hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
                margin=dict(t=20, l=10, r=10),
                xaxis=dict(fixedrange=True, showgrid=False),
                yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#eaeaea')
            )

            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### 🌪️ 변수 민감도 분석 (무엇이 내 은퇴를 망치는가?)")
            sens_df = res['sens_df'].sort_values(by="충격(%)", key=abs, ascending=True)
            t_colors = ['#E74C3C' if val > 0 else '#27AE60' for val in sens_df['충격(%)']]
            fig_torn = go.Figure(go.Bar(
                x=sens_df['충격(%)'], y=sens_df['시나리오'], orientation='h', marker_color=t_colors,
                text=[f"+{v:.1f}%p" if v > 0 else f"{v:.1f}%p" for v in sens_df['충격(%)']], textposition='auto'
            ))
            fig_torn.update_layout(
                title="<b>해당 사건 발생 시 '내 파산 확률'의 증감 폭</b>",
                xaxis_title="파산 확률 변동 폭 (%p)", height=250,
                plot_bgcolor='rgba(252, 252, 252, 1)', paper_bgcolor='rgba(255, 255, 255, 1)',
                margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(fixedrange=True, showgrid=False), yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#eaeaea')
            )
            fig_torn.add_vline(x=0, line_width=2, line_color="#333333")
            with st.container(border=True):
                st.plotly_chart(fig_torn, use_container_width=True)

        with d_col:
            with st.container(border=True):
                st.subheader("💡 퀀트 로직 요약")
                st.success("""
                **1. 조기 은퇴 계산기 🏃‍♂️**
                * 목표 은퇴 나이보다 일찍 퇴사해도 파산 확률을 방어할 수 있는지 역산하여 '가장 빠른 은퇴 나이'를 증명합니다.

                **2. MDD 방어력 📉**
                * 하위 10%의 최악의 폭락장 궤적을 추적하여, 내 계좌가 고점 대비 몇 %나 녹아내리는지(MDD), 원금 회복까지 몇 년을 버텨야 하는지 냉정하게 계산합니다.

                **3. 마일스톤 시각화 📊**
                * 그래프 위에 집 구입, 은퇴, 사치 종료 등 인생의 굵직한 이벤트가 자동 표기되어 자산 궤적의 굴곡을 완벽하게 설명합니다.
                """)

if __name__ == '__main__':
    main()
