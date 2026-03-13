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
# 2. 퀀트 시뮬레이션 코어 엔진 (수익률 추적 기능 추가)
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
        use_glide_path = self.params.get('use_glide_path', False)
        dwz_mode = self.params.get('dwz_mode', False)

        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)
        retire_idx = max(0, self.params['retire_age'] - current_age)

        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_assets_nom = np.zeros((n_simulations, simulation_years))
        sim_returns = np.zeros((n_simulations, simulation_years)) # ✅ 연도별 수익률 저장 매트릭스 추가

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

        if use_fat_tail:
            base_returns_matrix = np.random.standard_t(df=5, size=(n_simulations, simulation_years)) * (volatility / np.sqrt(5/3)) + base_return
        else:
            base_returns_matrix = np.random.normal(base_return, volatility, (n_simulations, simulation_years))

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
                if use_glide_path and age > 60:
                    current_ret = current_ret - ((age - 60) * 0.0015)
                    current_ret = max(current_ret, inflation + 0.005)
                final_returns.append(current_ret)

            final_returns = np.array(final_returns)
            sim_returns[i, :] = final_returns # ✅ 계산된 해당 궤적의 연도별 수익률 저장

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
                    if drawdown >= 0.30:
                        yolo_multiplier = 0.0
                    elif drawdown > 0.10:
                        yolo_multiplier = 1.0 - ((drawdown - 0.10) / 0.20)

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

        return years, sim_assets_pv, sim_assets_nom, sim_returns # ✅ sim_returns 반환 추가

    def run_hybrid_analysis(self, main_sims=5000, search_sims=1000):
        is_dwz = self.params.get('dwz_mode', False)
        target_ruin_prob = 15.0 if is_dwz else 5.0
        current_age = self.params['current_age']
        retire_age = self.params['retire_age']

        # ✅ 반환값 언패킹 수정 (main_returns 추가)
        years, main_pv, main_nom, main_returns = self.run_monte_carlo(n_simulations=main_sims, override_extra_margin=0)
        base_ruin = (np.sum(main_pv[:, -1] <= 0) / main_sims) * 100

        earliest_fire_age = retire_age
        if base_ruin <= target_ruin_prob and retire_age > current_age:
            for test_age in range(current_age, retire_age):
                temp_params = self.params.copy()
                temp_params['retire_age'] = test_age
                temp_sim = FinancialSimulator(temp_params)
                # ✅ 반환값 4개로 언패킹 처리
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
                # ✅ 반환값 4개로 언패킹 처리
                _, pv, _, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=mid)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100
                if ruin <= target_ruin_prob:
                    best_extra = mid
                    low = mid
                else:
                    high = mid
            safe_extra = int(best_extra)

        incs_set = set()
        incs_set.add(0)
        if safe_extra > 0:
            incs_set.add(int(safe_extra * 0.5))
            incs_set.add(safe_extra)
            incs_set.add(safe_extra + 50)
            incs_set.add(safe_extra + 150)
            incs_set.add(safe_extra + 300)
        else:
            incs_set.add(50)
            incs_set.add(100)
            incs_set.add(200)
            incs_set.add(300)

        incs = sorted(list(incs_set))
        results = []
        for inc in incs:
            if inc == 0: ruin = base_ruin
            else:
                # ✅ 반환값 4개로 언패킹 처리
                _, pv, _, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=inc)
                ruin = (np.sum(pv[:, -1] <= 0) / search_sims) * 100

            if inc == safe_extra and safe_extra > 0: label = f"+{inc}만 (안전한계선 🚩)"
            elif inc == 0: label = "현재 유지 (0만 원)"
            else: label = f"+{inc}만 원"
            results.append({'라벨': label, '추가액': inc, '파산 확률(%)': ruin})

        # ✅ main_returns 반환 추가
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
            # ✅ 반환값 4개로 언패킹 처리
            _, pv, _, _ = temp_sim.run_monte_carlo(n_simulations=sims, override_extra_margin=0)
            ruin = (np.sum(pv[:, -1] <= 0) / sims) * 100
            impact = ruin - base_ruin
            sens_results.append({"시나리오": label, "파산확률": ruin, "충격(%)": impact})
        return pd.DataFrame(sens_results)

# -----------------------------------------------------------
# 3. Streamlit UI (V45.1 + 1등 수익률 추적 추가)
# -----------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="My Quant Asset Sim (V45.1)")

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
        .briefing-box {
            background-color: #f8fafc; border-left: 5px solid #3b82f6;
            padding: 20px; border-radius: 5px; margin-bottom: 25px; font-size: 1.1rem; line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("💰 내 전용 퀀트 금융자산 시뮬레이터 (V45.1)")
    st.info("💡 나만의 라이프스타일에 맞춰 불필요한 기능은 없애고 직관성을 극대화했습니다.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.subheader("👤 1. 기본 정보")
            current_age = st.number_input("현재 나이", 20, 80, 40, key='in_age')
            death_age = st.number_input("목표 수명", 80, 120, 90, key='in_death', help="이 나이까지 자산이 고갈되지 않아야 시뮬레이션 성공으로 판정합니다.")

    with c2:
        with st.container(border=True):
            st.subheader("💵 2. 기본 수입 및 지출")
            col_inc1, col_inc2 = st.columns(2)
            monthly_income = col_inc1.number_input("월 수입 (세후/만원)", 0, value=500, step=10, key='in_income')
            apply_income_inflation = col_inc2.checkbox("수입 물가연동", value=False, key='in_inc_inf', help="체크 시, 은퇴 전까지 매년 내 월급도 물가상승률만큼 똑같이 인상된다고 가정합니다.")
            monthly_expense = st.number_input("월 필수 기본 지출 (만원)", 0, value=600, step=10, key='in_expense', help="숨만 쉬어도 나가는 필수 생계비입니다. 이 금액을 기준으로 '추가로 쓸 수 있는 욜로 예산'을 시뮬레이터가 계산해 줍니다.")

    with c3:
        with st.container(border=True):
            st.subheader("📈 3. 자산 및 거시 지표")
            current_asset = st.number_input("현재 금융자산 (만원)", 0, value=97000, step=100, key='in_asset', help="현재 주식, 예금 등에 굴러가고 있는 총자산 규모입니다.")
            col_ret1, col_ret2 = st.columns(2)
            expected_return = col_ret1.number_input("연 세후 수익률(%)", 0.0, 30.0, 14.0, step=0.5, key='in_ret', help="포트폴리오의 기대 수익률입니다. (자산이 10억을 넘어가면 엔진 내부에서 이 수치를 강제로 깎아내려 보수적으로 계산합니다.)")
            volatility = col_ret2.number_input("변동성(%)", 0.0, 50.0, 17.5, step=1.0, key='in_vol', help="수익률이 위아래로 흔들리는 정도입니다. 보통 주식 100%면 15~20% 수준입니다.")

            st.markdown("---")
            col_ret3, col_ret4 = st.columns(2)
            inflation = col_ret3.number_input("평시 물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1, key='in_inf', help="매년 돈의 가치가 하락하는 비율입니다. 결과창의 그래프는 이 물가상승분을 모두 빼고 '현재 체감하는 가치'로 보여줍니다.")
            tax_fee_rate = col_ret4.number_input("세금/수수료(연%)", 0.0, 5.0, 0.5, step=0.1, key='in_tax', help="증권사 수수료, 매매 슬리피지, 배당소득세 등 매년 계좌에서 조용히 녹아내리는 마찰 비용입니다.")

    st.markdown("---")

    with st.expander("🔥 **블랙 스완 & 다이 위드 제로 (고급 설정)**", expanded=True):
        with st.container(border=True):
            st.markdown("##### 🏖️ 라이프스타일 퀀트 최적화")
            dwz_mode = st.checkbox("🔥 Die with Zero 최적화 (Go-Go, Slow-Go, No-Go 3단계 커브 적용)", value=True, key='in_dwz',
                                   help="[현실화 커브] 65세까지는 욜로 예산 100% 사용, 80세까지는 30% 사용, 81세 이후 사치 0%. 대신 81세부터 매월 150만 원의 간병/의료비 폭탄이 강제 부과됩니다.")

        with st.container(border=True):
            st.markdown("##### 🚨 블랙 스완 방어선 (치명적 위기 테스트)")
            c_risk1, c_risk2 = st.columns(2)
            use_fat_tail = c_risk1.checkbox("📉 팻 테일(Fat Tail) 확률 적용", value=True, key='in_fat',
                                            help="정규분포를 무시하고 2008년 금융위기나 코로나처럼 '말도 안 되는 대폭락장'이 올 수학적 확률을 인위적으로 크게 높여 가혹하게 테스트합니다.")
            use_sorr_test = c_risk2.checkbox("😱 은퇴 직후 2년 폭락 (SORR 셔플링)", value=True, key='in_sorr',
                                             help="[수익률 순서 리스크] 전체 투자 기간 중 가장 끔찍한 수익률이 하필 '은퇴 직후 1~2년 차'에 연달아 터지는 가장 억울한 시나리오를 강제 주입합니다.")

            c_risk3, c_risk4 = st.columns(2)
            use_flex_spending = c_risk3.checkbox("🧠 점진적 생존 본능 (긴축 규칙)", value=True, key='in_flex',
                                                 help="[가이튼-클링거 룰] 전고점 대비 자산이 10% 깎이면 사치(YOLO) 지출을 줄이고, 30% 폭락하면 욜로 지출을 0원으로 락다운하여 계좌 파산을 방어하는 스마트 알고리즘입니다.")
            use_glide_path = c_risk4.checkbox("📉 자산 노화 (TDF Glide Path)", value=True, key='in_glide',
                                              help="[수익률 삭감 로직] 60세가 넘어가면 뇌의 판단력 저하 및 안전자산 선호 현상을 반영하여, 매년 기대 수익률을 0.15%p씩 강제로 깎아내립니다.")

            use_inflation_shock = st.checkbox("🔥 스태그플레이션 발작 충격 (구매력 파괴)", value=True, key='in_shock',
                                              help="은퇴 후 어느 시점에 무작위로 3년 연속 '물가상승률 7%'라는 초인플레이션 폭탄을 투하하여 구매력이 박살나도 계좌가 버티는지 검증합니다.")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        with st.container(border=True):
            st.subheader("🎯 4. 은퇴 목표")
            retire_mode = st.radio("은퇴 기준", ["나이 기준", "자산 기준"], horizontal=True, key='in_ret_mode', help="정해진 나이에 은퇴할지, 아니면 목표 금액이 모였을 때 즉시 사표를 던질지 선택합니다.")
            retire_age = st.number_input("은퇴 나이", current_age, 90, 60, key='in_ret_age', help="이 나이부터 근로 소득이 0원이 되고 계좌에서 돈을 빼서 쓰기 시작합니다.") if retire_mode == "나이 기준" else 100
            retire_by_asset = st.number_input("목표 자산 (만원)", 0, value=150000, key='in_ret_asset') if retire_mode == "자산 기준" else 0

    with col_right:
        with st.container(border=True):
            st.subheader("📱 5. 이벤트성 추가 수입/지출")
            st.caption("🚨 입력 금액은 먼 미래 지출이더라도 **'현재 체감하는 물가'** 기준으로 적어주세요.")
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
                st.info("💡 **[노후 방어율]** 죽을 때까지 안정적으로 나오는 '수입' 항목에는 **'확정연금'** 칸에 체크(☑️)해 주세요.")
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
                        "확정연금": st.column_config.CheckboxColumn("확정연금(방어율용)", help="국민연금 등 평생 죽을 때까지 삭감 없이 나오는 확실한 수입만 체크하세요. 생계비 방어율 계산에 사용됩니다.")
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
            'use_flex_spending': use_flex_spending, 'use_glide_path': use_glide_path,
            'dwz_mode': dwz_mode
        }

        with st.spinner("복잡계 퀀트 엔진 및 V45.1 벡터화 연산 중..."):
            simulator = FinancialSimulator(params)
            
            # ✅ main_returns 추가로 받기
            years, main_pv, main_nom, main_returns, safe_extra, base_ruin, stress_df, t_ruin, earliest_fire = simulator.run_hybrid_analysis(main_sims=n_sims, search_sims=1000)
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

            running_max = np.maximum.accumulate(bottom_10_pv)
            drawdowns = np.zeros_like(bottom_10_pv)
            valid_mask = running_max > 0
            drawdowns[valid_mask] = (running_max[valid_mask] - bottom_10_pv[valid_mask]) / running_max[valid_mask]

            max_dd = np.max(drawdowns)
            recovery_years = "회복 불가"

            if max_dd > 0:
                mdd_end_idx = np.argmax(drawdowns)
                mdd_start_idx = np.argmax(bottom_10_pv[:mdd_end_idx + 1])
                recovery_mask = bottom_10_pv[mdd_end_idx:] >= bottom_10_pv[mdd_start_idx]
                if np.any(recovery_mask):
                    recovery_years = f"{np.argmax(recovery_mask)}년"

            if safe_extra > 0:
                st.markdown(f"""
                <div class='briefing-box'>
                    <b>👨‍💼 수석 아키텍트의 결과 브리핑:</b><br>
                    "현재 {current_age}세이신 고객님, 연산이 완료되었습니다. 60세 은퇴 후 <b>월 50만 원의 건보료 폭탄</b>과 81세 이후 <b>월 150만 원의 간병비 스파이크</b>라는 악조건을 모두 엔진에 주입했습니다.<br>
                    그 결과, 65세까지는 매월 {600 + safe_extra:,}만 원을 소비하시고 그 이후엔 사치를 점진적으로 줄여가시는 조건(Go-Go, Slow-Go, No-Go)으로 파산 확률을 완벽히 방어해 냈습니다."
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='yolo-box'>
                    <p class='yolo-title'>💰 파산 확률 {t_ruin:.0f}% 방어선</p>
                    <p class='yolo-value'>이번 달 추가로 써도 되는 욜로(YOLO) 예산 = 월 {safe_extra:,}만 원</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ **안전 마진 없음:** 현재 지출로 파산 위험이 높습니다.")

            with st.expander("🔍 현재 시스템에 적용된 가혹한 현실 조건 (숨은 삭감 확인)", expanded=True):
                st.info(f"""
                **고객님의 입력값({expected_return}%)은 다음과 같이 엔진 내부에서 강제로 깎여서 계산되고 있습니다.**

                * ⚖️ **자산 규모 페널티:** 자산이 10억을 초과할수록 슬리피지로 인해 수익률이 하락합니다.
                * ⏳ **자산 노화 (TDF):** 60세 이후부터 매년 0.15%p씩 기본 수익률이 강제 하향 적용됩니다.
                * 🏥 **생애 말기 의료비 (DWZ):** 81세 이후부터 매월 150만 원의 간병/의료비 지출이 추가 발생합니다.

                👉 **결론: 지금 도출된 욜로(YOLO) 예산은 먼 미래의 악조건을 모두 견뎌내고 산출된 '진짜 쓸 수 있는 돈'입니다.**
                """)

            # ✅ session_state에 returns 데이터 추가 저장
            st.session_state['sim_results'] = {
                'years': years, 'pv': main_pv, 'nom': main_nom, 'returns': main_returns,
                'n_sims': n_sims, 'safe_extra': safe_extra, 'base_ruin': base_ruin, 
                'stress_df': stress_df, 'sens_df': sens_df,
                'dwz_mode': dwz_mode, 't_ruin': t_ruin, 'defense_rate': defense_rate,
                'depletion_age': depletion_age, 'lump_df': clean_lump_df,
                'earliest_fire': earliest_fire, 'max_dd': max_dd * 100, 'recovery_years': recovery_years
            }

    if 'sim_results' in st.session_state:
        res = st.session_state['sim_results']
        # ✅ returns 꺼내오기
        years, sim_assets_pv, sim_assets_nom, sim_returns = res['years'], res['pv'], res['nom'], res['returns'] 
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

       # -----------------------------------------------------------
        # ✅ 신규 교체 영역: 상위 10%, 중간값(50%), 하위 10% 다중 시나리오 동시 비교
        # -----------------------------------------------------------
        # 1. 10,000개 시나리오 중 '최종 연도 자산'을 기준으로 백분위 타겟값 계산
        final_assets = sim_assets_pv[:, -1]
        
        top10_target_val = np.percentile(final_assets, 90)
        median_target_val = np.percentile(final_assets, 50)
        bot10_target_val = np.percentile(final_assets, 10)
        
        # 2. 타겟값과 가장 일치하는 단 1개의 '실제 평행우주' 인덱스 추출
        top10_idx = np.abs(final_assets - top10_target_val).argmin()
        median_idx = np.abs(final_assets - median_target_val).argmin()
        bot10_idx = np.abs(final_assets - bot10_target_val).argmin()
        
        # 3. 각 궤적의 연도별 데이터 매핑
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
            
            for i, (label, data) in enumerate(paths.items()):
                ret_array = data["ret"]
                pv_array = data["pv"]
                
                # 기하평균(Geometric Mean)을 통한 해당 궤적의 실제 연평균 수익률(CAGR) 산출
                cagr = (np.prod(1 + ret_array) ** (1 / len(years)) - 1) * 100
                final_pv_eok = pv_array[-1] / 100000000
                
                cols[i].metric(label, f"최종 {final_pv_eok:.1f}억 원", f"연평균(CAGR): {cagr:.2f}%", delta_color="off")
                
                # 차트 및 테이블 출력을 위한 데이터 프레임 조립
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
                fig.add_vline(x=retire_age, line_dash="dash", line_color="#95a5a6", annotation_text="은퇴 (건보료 시작)")
            if is_dwz and 81 in years:
                fig.add_vline(x=81, line_dash="dot", line_color="#9b59b6", annotation_text="사치 종료 & 간병비 시작")

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
