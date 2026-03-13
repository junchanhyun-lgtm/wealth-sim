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

def calc_rolling_stats(returns_matrix, window_years):
    n_sims, n_cols = returns_matrix.shape
    if window_years > n_cols: return 0.0, 0.0
    cum_rets = []
    for start in range(n_cols - window_years + 1):
        window_slice = returns_matrix[:, start:start+window_years]
        geom_ret = np.prod(1 + window_slice, axis=1) - 1
        cum_rets.append(geom_ret)
    cum_rets = np.array(cum_rets).flatten()
    win_rate = (np.sum(cum_rets > 0) / len(cum_rets)) * 100
    median_cagr = (np.median(cum_rets + 1) ** (1/window_years) - 1) * 100
    return win_rate, median_cagr

# -----------------------------------------------------------
# 2. 퀀트 시뮬레이션 코어 엔진
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
        friction_cost = self.params['tax_fee_rate'] / 100
        target_asset_won = self.params['retire_by_asset'] * 10000

        use_fat_tail = self.params.get('use_fat_tail', False)
        use_sorr_test = self.params.get('use_sorr_test', False)
        use_inflation_shock = self.params.get('use_inflation_shock', False)
        use_flex_spending = self.params.get('use_flex_spending', False)
        dwz_mode = self.params.get('dwz_mode', False)

        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)
        retire_idx = max(0, self.params['retire_age'] - current_age)

        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_assets_nom = np.zeros((n_simulations, simulation_years))
        sim_returns = np.zeros((n_simulations, simulation_years)) 

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

        mu_array = np.full(simulation_years, base_return)
        vol_array = np.full(simulation_years, volatility)

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

        if use_fat_tail:
            base_returns_matrix = np.random.standard_t(df=5, size=(n_simulations, simulation_years))
            base_returns_matrix = base_returns_matrix * (vol_array / np.sqrt(5/3)) + mu_array
        else:
            base_returns_matrix = np.random.normal(loc=0.0, scale=1.0, size=(n_simulations, simulation_years))
            base_returns_matrix = base_returns_matrix * vol_array + mu_array

        for i in range(n_simulations):
            nominal_asset = current_asset
            high_water_mark = current_asset if current_asset > 0 else 1
            temp_returns = base_returns_matrix[i].copy()

            if use_sorr_test and simulation_years > retire_idx + 1:
                worst_indices = np.argsort(temp_returns)[:2]
                target_indices = [retire_idx, retire_idx + 1]
                worst_vals = temp_returns[worst_indices]
                target_vals = temp_returns[target_indices]
                temp_returns[worst_indices] = target_vals
                temp_returns[target_indices] = worst_vals

            final_returns = []
            for t, age in enumerate(years):
                current_ret = temp_returns[t] - friction_cost
                final_returns.append(current_ret)

            final_returns = np.array(final_returns)
            sim_returns[i, :] = final_returns 

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
                medical_spike_expense = 0
                yolo_ratio = 1.0

                if dwz_mode:
                    if age <= 65:
                        yolo_ratio = 1.0
                    elif age <= 80:
                        yolo_ratio = 0.3
                        decay_factor = max(0.70, (1 - 0.015) ** (age - 60))
                    else:
                        yolo_ratio = 0.0
                        medical_spike_expense = 1500000 * 12 * df_factor
                        decay_factor = 0.60
                    base_yolo_expense = base_yolo_expense * yolo_ratio

                yolo_multiplier = 1.0
                if use_flex_spending:
                    drawdown = (high_water_mark - nominal_asset) / high_water_mark if high_water_mark > 0 else 0
                    if drawdown < 0.05: yolo_multiplier = 1.0
                    elif drawdown < 0.10: yolo_multiplier = 0.8
                    elif drawdown < 0.15: yolo_multiplier = 0.6
                    elif drawdown < 0.20: yolo_multiplier = 0.4
                    elif drawdown < 0.25: yolo_multiplier = 0.2
                    else: yolo_multiplier = 0.0

                actual_yolo_expense = base_yolo_expense * yolo_multiplier
                base_expense_applied = base_monthly_expense * decay_factor
                target_lifestyle_annual_pv = base_expense_applied + actual_yolo_expense + medical_spike_expense

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

        return years, sim_assets_pv, sim_assets_nom, sim_returns

    def run_hybrid_analysis(self, main_sims=5000, search_sims=1000):
        is_dwz = self.params.get('dwz_mode', False)
        target_ruin_prob = 15.0 if is_dwz else 5.0
        current_age = self.params['current_age']
        retire_age = self.params['retire_age']

        years, main_pv, main_nom, main_returns = self.run_monte_carlo(n_simulations=main_sims, override_extra_margin=0)
        base_ruin = (np.sum(main_pv[:, -1] <= 0) / main_sims) * 100

        earliest_fire_age = retire_age
        if base_ruin <= target_ruin_prob and retire_age > current_age:
            for test_age in range(current_age, retire_age):
                temp_params = self.params.copy()
                temp_params['retire_age'] = test_age
                temp_sim = FinancialSimulator(temp_params)
                _, test_pv, _, _ = temp_sim.run_monte_carlo(n_simulations=500, override_extra_margin=0)
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
                _, pv, _, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=mid)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100
                if ruin <= target_ruin_prob:
                    best_extra = mid
                    low = mid
                else:
                    high = mid
            safe_extra = int(best_extra)

        incs_set = set([0])
        if safe_extra > 0:
            incs_set.update([int(safe_extra * 0.5), safe_extra, safe_extra + 50, safe_extra + 150, safe_extra + 300])
        else:
            incs_set.update([50, 100, 200, 300])

        incs = sorted(list(incs_set))
        results = []
        for inc in incs:
            if inc == 0: ruin = base_ruin
            else:
                _, pv, _, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=inc)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100
            label = f"+{inc}만 (안전한계선 🚩)" if inc == safe_extra and safe_extra > 0 else "현재 유지 (0만 원)" if inc == 0 else f"+{inc}만 원"
            results.append({'라벨': label, '추가액': inc, '파산 확률(%)': ruin})

        return years, main_pv, main_nom, main_returns, safe_extra, base_ruin, pd.DataFrame(results), target_ruin_prob, earliest_fire_age

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
            _, pv, _, _ = temp_sim.run_monte_carlo(n_simulations=sims, override_extra_margin=0)
            ruin = (np.sum(pv[:, -1] <= 0) / sims) * 100
            impact = ruin - base_ruin
            sens_results.append({"시나리오": label, "파산확률": ruin, "충격(%)": impact})
        return pd.DataFrame(sens_results)

# -----------------------------------------------------------
# 3. Streamlit UI (V46.6 Final - 툴팁 복구 완료)
# -----------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="My Quant Asset Sim (V46.6 Final)")

    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); background-color: #ffffff;
        }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }
        div.stAlert > div { border-radius: 10px; }
        [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; }
        .yolo-box {
            background-color: #f0fdf4; border: 2px solid #22c55e;
            border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;
        }
        .yolo-title { color: #166534; font-size: 1.4rem; font-weight: 700; margin: 0; }
        .yolo-value { color: #15803d; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0; }
        </style>
    """, unsafe_allow_html=True)

    st.title("💰 전담 퀀트 금융자산 종합 관리 시스템 (V46.6 Final)")
    st.info("💡 100% 시스템 트레이딩. 중간 개입 불가. 지정된 날짜에만 기계적 매매를 수행하십시오.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.subheader("👤 1. 기본 정보")
            current_age = st.number_input("현재 나이", 20, 80, 40, key='in_age')
            death_age = st.number_input("목표 수명", 80, 120, 90, key='in_death', help="이 나이까지 자산이 고갈되지 않아야 시뮬레이션 성공(파산 확률 0%)으로 판정합니다.")

    with c2:
        with st.container(border=True):
            st.subheader("💵 2. 기본 수입 및 지출")
            col_inc1, col_inc2 = st.columns(2)
            monthly_income = col_inc1.number_input("월 수입 (세후/만원)", 0, value=500, step=10, key='in_income', help="한의원 순수익입니다.")
            apply_income_inflation = col_inc2.checkbox("수입 물가연동", value=False, key='in_inc_inf', help="체크 시, 은퇴 전까지 소득도 물가상승률만큼 동반 상승한다고 가정합니다. (현재 보수적으로 고정 가정 권장)")
            monthly_expense = st.number_input("월 필수 기본 지출 (만원)", 0, value=600, step=10, key='in_expense', help="가족 생계유지를 위한 필수 생활비입니다. 이 금액을 1순위로 방어한 뒤 남는 돈을 '욜로 예산'으로 배정합니다.")

    with c3:
        with st.container(border=True):
            st.subheader("📈 3. 자산 및 거시 지표")
            current_asset = st.number_input("현재 금융자산 (만원)", 0, value=97000, step=100, key='in_asset', help="대출 3억 원이 포함된 운용 가능 총액입니다.")
            col_ret1, col_ret2 = st.columns(2)
            expected_return = col_ret1.number_input("연 세후 수익률(%)", 0.0, 30.0, 14.0, step=0.5, key='in_ret', help="포트폴리오 백테스트 수익률. 자산이 커지면 엔진 내부에서 호가창 마찰(Slippage) 페널티를 부여합니다.")
            volatility = col_ret2.number_input("변동성(%)", 0.0, 50.0, 17.5, step=1.0, key='in_vol', help="퀀트 시스템 트레이더의 야수성을 반영하여 90세 사망 시점까지 영구적으로 유지되는 고정 변동성입니다.")

            st.markdown("---")
            col_ret3, col_ret4 = st.columns(2)
            inflation = col_ret3.number_input("평시 물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1, key='in_inf', help="물가 상승으로 인한 구매력 하락을 자산 궤적에 반영합니다.")
            tax_fee_rate = col_ret4.number_input("세금/수수료(연%)", 0.0, 5.0, 0.5, step=0.1, key='in_tax', help="매매 수수료, 슬리피지, 퀀트 교체 시 발생하는 마찰 비용을 매년 선제적으로 삭감합니다.")

    st.markdown("---")

    with st.expander("🔥 **블랙 스완 & 다이 위드 제로 (고급 설정)**", expanded=True):
        with st.container(border=True):
            st.markdown("##### 🏖️ 라이프스타일 퀀트 최적화")
            dwz_mode = st.checkbox("🔥 Die with Zero 최적화 (Go-Go, Slow-Go, No-Go 3단계 커브 적용)", value=True, key='in_dwz', help="[현실화 커브] 65세까지는 욜로 예산 100% 사용, 80세까지는 30% 사용, 81세 이후 사치 0%. 대신 81세부터 매월 150만 원의 간병/의료비 폭탄이 강제 부과됩니다.")

        with st.container(border=True):
            st.markdown("##### 🚨 블랙 스완 방어선 (치명적 위기 테스트)")
            c_risk1, c_risk2 = st.columns(2)
            use_fat_tail = c_risk1.checkbox("📉 팻 테일(Fat Tail) 확률 적용", value=True, key='in_fat', help="정규분포를 무시하고 코로나나 금융위기급 '대폭락장'이 올 수학적 확률을 인위적으로 높여 시스템의 파산 여부를 가혹하게 검증합니다.")
            use_sorr_test = c_risk2.checkbox("😱 은퇴 직후 2년 폭락 (SORR 셔플링)", value=True, key='in_sorr', help="[수익률 순서 리스크] 전체 투자 기간 중 가장 끔찍한 수익률이 하필 '은퇴 직후 인출기'에 연달아 터지는 가장 억울한 시나리오를 강제 주입합니다.")

            c_risk3, c_risk4 = st.columns(2)
            use_flex_spending = c_risk3.checkbox("🧠 다단계 생존 본능 (5% 단위 긴축 룰)", value=True, key='in_flex', help="전고점 대비 자산이 5% 하락할 때마다 추가 욜로(YOLO) 지출을 20%씩 기계적으로 삭감하여 복리 훼손을 방어합니다.")
            use_inflation_shock = c_risk4.checkbox("🔥 스태그플레이션 발작 충격 (구매력 파괴)", value=True, key='in_shock', help="무작위 시점에 3년 연속 '물가상승률 7%'라는 초인플레이션 폭탄을 투하하여 구매력이 박살나도 계좌가 버티는지 검증합니다.")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        with st.container(border=True):
            st.subheader("🎯 4. 은퇴 목표")
            retire_mode = st.radio("은퇴 기준", ["나이 기준", "자산 기준"], horizontal=True, key='in_ret_mode', help="정해진 나이에 은퇴할지, 아니면 목표 금액이 모였을 때 즉시 사표를 던질지 선택합니다.")
            retire_age = st.number_input("은퇴 나이", current_age, 90, 60, key='in_ret_age', help="이 나이부터 근로 소득이 0원이 되고 계좌에서 돈을 빼서 쓰기 시작합니다.") if retire_mode == "나이 기준" else 100
            retire_by_asset = st.number_input("목표 자산 (만원)", 0, value=150000, key='in_ret_asset', help="순자산 30억 목표 시 조기 은퇴합니다.") if retire_mode == "자산 기준" else 0

    with col_right:
        with st.container(border=True):
            st.subheader("📱 5. 이벤트성 추가 수입/지출")
            tab1, tab2 = st.tabs(["💸 일회성 목돈", "📅 기간성 수입/지출 (연금 포함)"])

            with tab1:
                if 'lump_df' not in st.session_state:
                    st.session_state.lump_df = pd.DataFrame([
                        {"나이": 41, "유형": "지출", "내용": "대출상환", "금액(만원)": 10000},
                        {"나이": 45, "유형": "지출", "내용": "주택구입", "금액(만원)": 32000}
                    ])
                edited_lump_df = st.data_editor(st.session_state.lump_df, num_rows="dynamic", key="lump_editor", use_container_width=True,
                    column_config={"유형": st.column_config.SelectboxColumn("유형", options=["수입", "지출"], required=True),
                                   "금액(만원)": st.column_config.NumberColumn("금액(만원)", min_value=0)})
                clean_lump_df = edited_lump_df.dropna(subset=['나이', '유형', '금액(만원)'])
                st.session_state.lump_df = clean_lump_df

            with tab2:
                if 'recur_df' not in st.session_state:
                    st.session_state.recur_df = pd.DataFrame([
                        {"시작나이": 47, "기간(년)": 5, "유형": "지출", "내용": "자동차할부", "월금액(만원)": 250, "확정연금": False},
                        {"시작나이": 40, "기간(년)": 20, "유형": "지출", "내용": "부모님용돈", "월금액(만원)": 100, "확정연금": False},
                        {"시작나이": 40, "기간(년)": 5, "유형": "수입", "내용": "주6일 초과근무", "월금액(만원)": 200, "확정연금": False},
                        {"시작나이": 60, "기간(년)": 30, "유형": "지출", "내용": "지역 건보료 폭탄", "월금액(만원)": 50, "확정연금": False},
                        {"시작나이": 70, "기간(년)": 30, "유형": "수입", "내용": "국민연금", "월금액(만원)": 100, "확정연금": True},
                        {"시작나이": 70, "기간(년)": 30, "유형": "수입", "내용": "주택연금", "월금액(만원)": 200, "확정연금": True}
                    ])
                edited_recur_df = st.data_editor(st.session_state.recur_df, num_rows="dynamic", key="recur_editor", use_container_width=True,
                    column_config={
                        "유형": st.column_config.SelectboxColumn("유형", options=["수입", "지출"], required=True),
                        "월금액(만원)": st.column_config.NumberColumn("월금액(만원)", min_value=0),
                        "확정연금": st.column_config.CheckboxColumn("확정연금(방어율용)", help="국민연금 등 평생 죽을 때까지 삭감 없이 나오는 확실한 수입만 체크하세요.")
                    })
                clean_recur_df = edited_recur_df.dropna(subset=['시작나이', '기간(년)', '유형', '월금액(만원)'])
                st.session_state.recur_df = clean_recur_df

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
            'use_flex_spending': use_flex_spending,
            'dwz_mode': dwz_mode
        }

        with st.spinner("복잡계 퀀트 엔진 연산 수행 중..."):
            simulator = FinancialSimulator(params)
            
            years, main_pv, main_nom, main_returns, safe_extra, base_ruin, stress_df, t_ruin, earliest_fire = simulator.run_hybrid_analysis(main_sims=n_sims, search_sims=1000)
            sens_df = simulator.run_sensitivity(base_ruin, sims=2000)

            total_pension = 0
            if '확정연금' in clean_recur_df.columns:
                pension_df = clean_recur_df[(clean_recur_df['유형'] == '수입') & (clean_recur_df['확정연금'] == True)]
                total_pension = pension_df['월금액(만원)'].sum()
            defense_rate = (total_pension / monthly_expense * 100) if monthly_expense > 0 else 0

            # 결괏값 출력 전 확실한 체감가치(PV) 안내
            st.info("💡 **[가치 평가 기준]** 본 시뮬레이터의 모든 결괏값(자산, YOLO 예산 등)은 입력된 평시 물가상승률을 역산하여, 미래 화폐 가치를 **'현재 체감하는 구매력(Present Value)'** 기준으로 완벽히 변환하여 표시합니다.")

            if safe_extra > 0:
                st.markdown(f"""
                <div class='yolo-box'>
                    <p class='yolo-title'>💰 파산 확률 {t_ruin:.0f}% 방어선 통과</p>
                    <p class='yolo-value'>이번 달 추가로 써도 되는 욜로(YOLO) 예산 = 월 {safe_extra:,}만 원</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ **안전 마진 없음:** 현재 지출로 파산 위험이 높습니다. 지출 통제가 시급합니다.")

            st.session_state['sim_results'] = {
                'years': years, 'pv': main_pv, 'nom': main_nom, 'returns': main_returns,
                'n_sims': n_sims, 'safe_extra': safe_extra, 'base_ruin': base_ruin, 
                'stress_df': stress_df, 'sens_df': sens_df,
                'dwz_mode': dwz_mode, 't_ruin': t_ruin, 'defense_rate': defense_rate,
                'lump_df': clean_lump_df, 'earliest_fire': earliest_fire
            }

    if 'sim_results' in st.session_state:
        res = st.session_state['sim_results']
        
        years, sim_assets_pv, sim_assets_nom, sim_returns = res['years'], res['pv'], res['nom'], res['returns']
        safe_extra, base_ruin, stress_df = res['safe_extra'], res['base_ruin'], res['stress_df']
        is_dwz, target_ruin = res['dwz_mode'], res['t_ruin']
        defense_rate = res['defense_rate']
        res_lump_df, earliest_fire = res['lump_df'], res['earliest_fire']

        final_assets = sim_assets_pv[:, -1]
        top10_idx = np.abs(final_assets - np.percentile(final_assets, 90)).argmin()
        median_idx = np.abs(final_assets - np.percentile(final_assets, 50)).argmin()
        bot10_idx = np.abs(final_assets - np.percentile(final_assets, 10)).argmin()

        with st.expander(f"⏳ [멘탈 방어] 구간별 승률(Rolling Window)", expanded=False):
            st.markdown("###### 📊 10,000번의 우주를 분석한 '보유 기간별' 시스템 승률")
            c_r1, c_r2, c_r3, c_r4 = st.columns(4)
            r_windows = [1, 3, 5, 10]
            r_cols = [c_r1, c_r2, c_r3, c_r4]
            for r_w, r_c in zip(r_windows, r_cols):
                w_rate, m_cagr = calc_rolling_stats(sim_returns, r_w)
                r_c.metric(f"{r_w}년 유지 시 승률", f"{w_rate:.1f}%", f"해당 구간 연평균: {m_cagr:.2f}%", delta_color="off")
            st.caption("※ 기간이 길어질수록 승률이 수렴하는 것은 장기 퀀트 투자의 절대적 수학 당위성입니다.")

        paths = {
            "상위 10% (운수 좋은 날)": {"ret": sim_returns[top10_idx, :], "pv": sim_assets_pv[top10_idx, :]},
            "중간값 (가장 현실적)": {"ret": sim_returns[median_idx, :], "pv": sim_assets_pv[median_idx, :]},
            "하위 10% (스트레스)": {"ret": sim_returns[bot10_idx, :], "pv": sim_assets_pv[bot10_idx, :]}
        }

        with st.expander(f"📊 [심층 분석] 상/중/하위 시나리오별 수익률 및 자산 궤적 3종 비교", expanded=False):
            st.markdown(f"**총 {res['n_sims']:,}번의 평행우주 중, 자산 성과 기준 상위 10%, 50%, 90% 위치에 있는 3개의 대표 궤적입니다.**")

            c_m1, c_m2, c_m3 = st.columns(3)
            cols = [c_m1, c_m2, c_m3]
            comp_data = {"나이": years}
            
            target_age_idx = years.index(60) if 60 in years else -1

            for i, (label, data) in enumerate(paths.items()):
                ret_array = data["ret"]
                pv_array = data["pv"]
                cagr = (np.prod(1 + ret_array) ** (1 / len(years)) - 1) * 100
                age60_pv_eok = pv_array[target_age_idx] / 100000000 if target_age_idx != -1 else 0
                
                cols[i].metric(label, f"60세 자산 {age60_pv_eok:.1f}억 원", f"연평균(CAGR): {cagr:.2f}%", delta_color="off")
                
                short_label = label.split(" ")[0] + " " + label.split(" ")[1]
                comp_data[f"[{short_label}] 수익률(%)"] = np.round(ret_array * 100, 2)
                comp_data[f"[{short_label}] 자산(억)"] = np.round(pv_array / 100000000, 2)

            st.markdown("---")
            comp_df = pd.DataFrame(comp_data).set_index("나이")
            
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("###### 📈 연도별 적용 수익률 추이 비교")
                st.line_chart(comp_df[[c for c in comp_df.columns if "수익률" in c]], height=300)
            with c_chart2:
                st.markdown("###### 💰 연도별 자산 잔고 추이 비교 (현재가치)")
                st.line_chart(comp_df[[c for c in comp_df.columns if "자산" in c]], height=300)

            st.markdown("###### 📋 상세 교차 검증 테이블")
            st.dataframe(comp_df, use_container_width=True, height=250)

        st.markdown("<br>", unsafe_allow_html=True)
        g_col, d_col = st.columns([2.5, 1.2])

        with g_col:
            colors = ['#27AE60' if val <= target_ruin + 0.01 else '#F1C40F' if val < target_ruin + 10 else '#E74C3C' for val in stress_df['파산 확률(%)']]
            fig_stress = go.Figure(data=[go.Bar(x=stress_df['라벨'], y=stress_df['파산 확률(%)'], marker_color=colors, text=[f"{val:.1f}%" for val in stress_df['파산 확률(%)']], textposition='auto')])
            title_suffix = f"(81세 사치 컷오프 & {target_ruin:.0f}% 방어)" if is_dwz else f"(평생 고정 & {target_ruin:.0f}% 방어)"
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
            median_pv = np.median(sim_assets_pv, axis=0) / 100000000
            top_10_pv = np.percentile(sim_assets_pv, 90, axis=0) / 100000000
            bottom_10_pv = np.percentile(sim_assets_pv, 10, axis=0) / 100000000
            
            target_display_age = retire_age if retire_mode == "나이 기준" else 60
            if target_display_age not in years: target_display_age = death_age
            idx_target = years.index(target_display_age)

            k1, k2, k3 = st.columns(3)
            k1.metric("기본 파산 확률", f"{base_ruin:.1f}%")
            k2.metric(f"{target_display_age}세 예상액", f"{median_pv[idx_target]:.2f}억 원")
            k3.metric("최악의 경우 (하위 10%)", f"{bottom_10_pv[idx_target]:.2f}억 원")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years+years[::-1], y=np.concatenate([top_10_pv, bottom_10_pv[::-1]]), fill='toself', fillcolor='rgba(46, 134, 193, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='신뢰구간(10~90%)', hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=years, y=median_pv, line=dict(color='#2E86C1', width=3), name='중앙값', hovertemplate='%{y:.2f}억 원<extra></extra>'))
            fig.add_trace(go.Scatter(x=years, y=bottom_10_pv, line=dict(color='#E74C3C', width=2, dash='dot'), name='하위 10%', hovertemplate='%{y:.2f}억 원<extra></extra>'))
            fig.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)

            if retire_age in years:
                fig.add_vline(x=retire_age, line_dash="dash", line_color="#95a5a6", annotation_text="은퇴 시작")
            if is_dwz and 81 in years:
                fig.add_vline(x=81, line_dash="dot", line_color="#9b59b6", annotation_text="사치 종료 & 간병비")

            for _, row in res_lump_df.iterrows():
                if row['금액(만원)'] >= 10000 and row['나이'] in years:
                    fig.add_vline(x=row['나이'], line_dash="dot", line_color="#f39c12", annotation_text=row['내용'])

            fig.update_layout(
                xaxis_title="나이", yaxis_title="현재 체감 자산 (억 원)", height=450,
                plot_bgcolor='rgba(252, 252, 252, 1)', paper_bgcolor='rgba(255, 255, 255, 1)',
                hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
                margin=dict(t=20, l=10, r=10),
                xaxis=dict(fixedrange=True, showgrid=False),
                yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#eaeaea')
            )
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### 🌪️ 변수 민감도 분석 (파산 트리거)")
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
                st.subheader("💡 퀀트 로직 및 원칙 요약")
                st.success("""
                **1. 체감가치(PV) 변환 원칙 🧮**
                모든 궤적의 자산 금액은 인플레이션(물가상승률)을 역산하여 깎아낸 '현재 시점의 구매력'만을 엄격하게 표시합니다.

                **2. 롤링 윈도우 승률 ⏳**
                임의의 시점에 투자하여 N년을 보유했을 때의 승률입니다. 시장 예측을 배제하고 기계적으로 장기 보유해야 하는 통계적 근거입니다.

                **3. 시스템 야수성 유지 (변동성) 🦁**
                나이가 든다고 자산을 임의로 채권으로 돌리지 않습니다. 시스템 트레이더의 원칙에 따라 죽을 때까지 설정된 고정 변동성(17.5%)을 유지합니다.

                **4. 다단계 생존 본능 (긴축 룰) 📉**
                폭락장이 왔을 때 가만히 앉아 당하지 않습니다. 전고점 대비 자산이 5% 하락할 때마다 사치(YOLO) 지출을 20%씩 기계적으로 삭감하여 복리 훼손을 막습니다.

                **5. 슬리피지 및 마찰 비용 💸**
                자산 규모가 10억, 20억 단위로 커질수록 호가창 마찰과 세금으로 인해 실제 포트폴리오 수익률이 점진적으로 깎이는 디버프 로직이 내부적으로 가동됩니다.

                **6. 수익률 순서 리스크 (SORR) 😱**
                전체 생애 중 가장 운이 없는 최악의 수익률 2개가 하필 '은퇴 직후 인출기'에 연달아 발생하도록 셔플링하여 계좌의 내구성을 가혹하게 검증합니다.

                **7. Die with Zero & 의료비 폭탄 🏥**
                65세부터 서서히 사치 예산을 줄이고, 81세부터는 모든 YOLO 지출을 0원으로 락다운합니다. 동시에 매월 150만 원의 간병비 스파이크를 강제로 발생시킵니다.

                **8. 팻 테일(Fat Tail) 리스크 🌪️**
                코로나19, 서브프라임 모기지 사태처럼 정규분포 통계를 무시하고 벌어지는 '극단적 꼬리 위험(대폭락)' 발생 확률을 엔진에 인위적으로 대폭 주입합니다.
                """)

if __name__ == '__main__':
    main()
