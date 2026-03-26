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
# 2. 퀀트 시뮬레이션 코어 엔진 (V57)
# -----------------------------------------------------------
class FinancialSimulator:
    def __init__(self, params):
        self.params = params

    def run_monte_carlo(self, n_simulations=5000, override_extra_margin=0):
        current_age = self.params['current_age']
        death_age = self.params['death_age']
        current_asset = self.params['current_asset'] * 10000.0

        base_monthly_income = (self.params['monthly_income'] * 10000) * 12
        apply_income_inflation = self.params['apply_income_inflation']
        base_monthly_expense = self.params['monthly_expense'] * 10000 * 12

        inflation = self.params['inflation'] / 100.0
        friction_cost = self.params['tax_fee_rate'] / 100.0
        
        use_fat_tail = self.params.get('use_fat_tail', False)
        use_inflation_shock = self.params.get('use_inflation_shock', False)
        use_flex_spending = self.params.get('use_flex_spending', False)
        dwz_mode = self.params.get('dwz_mode', False)
        
        use_glide_path = self.params.get('use_glide_path', True)

        years = list(range(current_age, death_age + 1))
        simulation_years = len(years)
        retire_idx = max(0, self.params['retire_age'] - current_age)

        inflation_matrix = np.full((n_simulations, simulation_years), inflation)
        shock_mask = np.zeros((n_simulations, simulation_years), dtype=bool)
        
        if use_inflation_shock:
            max_start = max(retire_idx, min(simulation_years - 3, retire_idx + 9))
            shock_starts = np.random.randint(retire_idx, max_start + 1, size=n_simulations)
            for i in range(3): 
                cols = np.clip(shock_starts + i, 0, simulation_years - 1)
                shock_mask[np.arange(n_simulations), cols] = True
                inflation_matrix[np.arange(n_simulations), cols] = 0.07

        discount_factors = np.ones((n_simulations, simulation_years))
        if simulation_years > 1:
            discount_factors[:, 1:] = np.cumprod(1 + inflation_matrix[:, :-1], axis=1)

        recurring_df = self.params['recurring_events']
        lump_df = self.params['lump_events']

        pv_extra_income = np.zeros(simulation_years)
        pv_extra_expense = np.zeros(simulation_years)
        nom_extra_income = np.zeros(simulation_years)
        nom_extra_expense = np.zeros(simulation_years)
        pv_lump_sum = np.zeros(simulation_years)

        for t, age in enumerate(years):
            if not recurring_df.empty:
                for _, row in recurring_df.iterrows():
                    if row['시작나이'] <= age < row['시작나이'] + row['기간(년)']:
                        amt_val = abs(row['월금액(만원)']) * 10000 * 12
                        if row.get('물가연동', False):
                            if row['유형'] == '수입': pv_extra_income[t] += amt_val
                            else: pv_extra_expense[t] += amt_val
                        else:
                            if row['유형'] == '수입': nom_extra_income[t] += amt_val
                            else: nom_extra_expense[t] += amt_val
            if not lump_df.empty:
                for _, row in lump_df.iterrows():
                    if row['나이'] == age:
                        amt_val = abs(row['금액(만원)']) * 10000
                        if row['유형'] == '수입': pv_lump_sum[t] += amt_val
                        else: pv_lump_sum[t] -= amt_val

        mu_base = np.zeros(simulation_years)
        vol_base = np.zeros(simulation_years)
        
        ret_pre = self.params['expected_return_pre'] / 100.0
        ret_post = self.params['expected_return_post'] / 100.0
        v_pre = self.params['vol_pre'] / 100.0
        v_post = self.params['vol_post'] / 100.0
        
        transition_years = 5
        
        for t, age in enumerate(years):
            if not use_glide_path:
                mu_base[t] = ret_post if age >= self.params['retire_age'] else ret_pre
                vol_base[t] = v_post if age >= self.params['retire_age'] else v_pre
            else:
                if age <= self.params['retire_age'] - transition_years:
                    mu_base[t] = ret_pre
                    vol_base[t] = v_pre
                elif age >= self.params['retire_age']:
                    mu_base[t] = ret_post
                    vol_base[t] = v_post
                else:
                    ratio = (age - (self.params['retire_age'] - transition_years)) / transition_years
                    mu_base[t] = ret_pre * (1 - ratio) + ret_post * ratio
                    vol_base[t] = v_pre * (1 - ratio) + v_post * ratio

        mu_matrix = np.tile(mu_base, (n_simulations, 1))
        vol_matrix = np.tile(vol_base, (n_simulations, 1))

        if use_inflation_shock:
            shock_penalty = 0.05
            vol_multiplier = 1.5
            mu_matrix[shock_mask] -= shock_penalty
            vol_matrix[shock_mask] *= vol_multiplier

        if use_fat_tail:
            z_matrix = np.random.standard_t(df=5, size=(n_simulations, simulation_years)) / np.sqrt(5/3)
        else:
            z_matrix = np.random.normal(loc=0.0, scale=1.0, size=(n_simulations, simulation_years))

        temp_returns = z_matrix * vol_matrix + mu_matrix - friction_cost
        
        sim_returns = np.zeros_like(temp_returns)
        sim_returns[:, 0] = temp_returns[:, 0]
        reversion_strength = 0.10 

        for t in range(1, simulation_years):
            excess_prev = sim_returns[:, t-1] - mu_matrix[:, t-1]
            sim_returns[:, t] = temp_returns[:, t] - (reversion_strength * excess_prev)

        market_index_matrix = np.cumprod(1 + sim_returns, axis=1) * 100.0
        high_water_mark_matrix = np.maximum.accumulate(market_index_matrix, axis=1)
        market_drawdown_matrix = (high_water_mark_matrix - market_index_matrix) / high_water_mark_matrix

        sim_assets_pv = np.zeros((n_simulations, simulation_years))
        sim_assets_nom = np.zeros((n_simulations, simulation_years))
        
        current_assets = np.full(n_simulations, current_asset) 

        for t, age in enumerate(years):
            df_factor = discount_factors[:, t] 
            
            extra_inc = (pv_extra_income[t] * df_factor) + nom_extra_income[t]
            extra_exp = (pv_extra_expense[t] * df_factor) + nom_extra_expense[t]
            nom_lump = pv_lump_sum[t] * df_factor
            
            is_retired = True if age >= self.params['retire_age'] else False
            if not is_retired:
                current_nom_inc = base_monthly_income * df_factor if apply_income_inflation else np.full(n_simulations, base_monthly_income)
            else:
                current_nom_inc = np.zeros(n_simulations)
                
            decay_factor = 1.0
            yolo_ratio = 1.0
            
            if dwz_mode:
                if age <= 65:
                    yolo_ratio = 1.0
                elif age <= 80:
                    yolo_ratio = 0.3
                    decay_factor = max(0.70, (1 - 0.015) ** (age - 60))
                else:
                    yolo_ratio = 0.0
                    decay_factor = 0.60
                    
            base_yolo_expense = override_extra_margin * 10000 * 12 * yolo_ratio
            
            yolo_mult = np.ones(n_simulations)
            if use_flex_spending:
                drawdowns_t = market_drawdown_matrix[:, t]
                yolo_mult = np.where(drawdowns_t < 0.05, 1.0,
                            np.where(drawdowns_t < 0.10, 0.8,
                            np.where(drawdowns_t < 0.15, 0.6,
                            np.where(drawdowns_t < 0.20, 0.4,
                            np.where(drawdowns_t < 0.25, 0.2, 0.0)))))
                            
            actual_yolo_expense = base_yolo_expense * yolo_mult
            base_expense_applied = base_monthly_expense * decay_factor
            
            target_lifestyle_annual_pv = base_expense_applied + actual_yolo_expense
            
            total_income_annual = current_nom_inc + extra_inc
            nominal_actual_spending = (target_lifestyle_annual_pv * df_factor) + extra_exp
            
            net_cashflow = total_income_annual - nominal_actual_spending
            
            threshold = 1_000_000_000.0
            safe_assets = np.maximum(current_assets, 1.0) 
            penalty = np.where(current_assets > threshold, 0.015 * np.log10(safe_assets / threshold), 0.0)
            
            adj_return = sim_returns[:, t] - penalty
            
            gain_on_base = current_assets * adj_return
            gain_on_cashflow = net_cashflow * (adj_return / 2.0)
            
            current_assets = current_assets + gain_on_base + net_cashflow + gain_on_cashflow + nom_lump
            current_assets = np.maximum(current_assets, 0.0) 
            
            sim_assets_nom[:, t] = current_assets
            sim_assets_pv[:, t] = current_assets / df_factor
            
        return years, sim_assets_pv, sim_assets_nom, sim_returns

    def run_hybrid_analysis(self, main_sims=5000, search_sims=1000):
        is_dwz = self.params.get('dwz_mode', False)
        target_ruin_prob = 20.0 if is_dwz else 15.0
        
        current_age = self.params['current_age']
        retire_age = self.params['retire_age']

        years, main_pv, main_nom, main_returns = self.run_monte_carlo(n_simulations=main_sims, override_extra_margin=0)
        
        base_ruin = (np.sum(np.any(main_pv <= 0, axis=1)) / main_sims) * 100

        safe_extra = 0
        if base_ruin <= target_ruin_prob:
            low, high = 0, 5000
            best_extra = 0
            for _ in range(8):
                mid = (low + high) / 2
                _, pv, _, _ = self.run_monte_carlo(n_simulations=search_sims, override_extra_margin=mid)
                ruin = (np.sum(np.any(pv <= 0, axis=1)) / search_sims) * 100
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
                ruin = (np.sum(np.any(pv <= 0, axis=1)) / search_sims) * 100
            label = f"+{inc}만 (안전방어선 🚩)" if inc == safe_extra and safe_extra > 0 else "현재 유지 (0만 원)" if inc == 0 else f"+{inc}만 원"
            results.append({'라벨': label, '추가액': inc, '파산 확률(%)': ruin})

        return years, main_pv, main_nom, main_returns, safe_extra, base_ruin, pd.DataFrame(results), target_ruin_prob

    def run_sensitivity(self, base_ruin, sims=2000):
        scenarios = [
            ("수익률 -1%p 하락 (전구간)", 'expected_return', -1.0),
            ("물가상승률 +1%p 폭등", 'inflation', 1.0),
            ("기본생활비 +10% 증가", 'monthly_expense', self.params['monthly_expense'] * 0.1),
        ]
        sens_results = []
        for label, key, delta in scenarios:
            temp_params = self.params.copy()
            if key == 'expected_return':
                temp_params['expected_return_pre'] += delta
                temp_params['expected_return_post'] += delta
            else:
                temp_params[key] += delta
                
            temp_sim = FinancialSimulator(temp_params)
            _, pv, _, _ = temp_sim.run_monte_carlo(n_simulations=sims, override_extra_margin=0)
            ruin = (np.sum(np.any(pv <= 0, axis=1)) / sims) * 100
            impact = ruin - base_ruin
            sens_results.append({"시나리오": label, "파산확률": ruin, "충격(%)": impact})
        return pd.DataFrame(sens_results)

# -----------------------------------------------------------
# 3. Streamlit UI (V57 Final)
# -----------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="My Quant Asset Sim (V57)")

    st.markdown("""
        <style>
        [data-testid="stMetric"] { border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); background-color: #ffffff; }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }
        div.stAlert > div { border-radius: 10px; }
        .yolo-box { background-color: #f0fdf4; border: 2px solid #22c55e; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px; }
        .yolo-title { color: #166534; font-size: 1.4rem; font-weight: 700; margin: 0; }
        .yolo-value { color: #15803d; font-size: 2.2rem; font-weight: 800; margin: 10px 0 0 0; }
        </style>
    """, unsafe_allow_html=True)

    st.title("💰 전담 퀀트 금융자산 종합 관리 시스템 (V57)")
    st.info("💡 V57 업데이트: 파산타겟(15/20%) 유지, 툴팁 복구, 주택구입 50세 연기, 상속 이벤트 4회분 반영")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.subheader("👤 1. 기본 정보 & 은퇴 설정")
            current_age = st.number_input("현재 나이", 20, 80, 40, key='in_age', help="현재 나이를 입력하십시오.")
            retire_age = st.number_input("은퇴 나이 (소득 중단 시점)", current_age, 90, 60, key='in_ret_age', help="본 시스템의 1차 목표 지점입니다. 60세 시점의 자산 방어율을 중점적으로 추적합니다.") 
            death_age = st.number_input("목표 수명", 80, 120, 90, key='in_death', help="이 나이까지 자산이 고갈되지 않아야 파산 확률 0%로 판정합니다.")

    with c2:
        with st.container(border=True):
            st.subheader("💵 2. 기본 수입 및 지출")
            col_inc1, col_inc2 = st.columns(2)
            monthly_income = col_inc1.number_input("월 수입 (세후/만원)", 0, value=500, step=10, key='in_income', help="한의원 순수익입니다.")
            apply_income_inflation = col_inc2.checkbox("수입 물가연동", value=False, key='in_inc_inf', help="체크 시, 은퇴 전까지 소득도 물가상승률만큼 동반 상승한다고 가정합니다.")
            monthly_expense = st.number_input("월 필수 지출 (대출 이자 포함/만원)", 0, value=600, step=10, key='in_expense', help="가족 생활비 및 대출 이자가 모두 포함된 총 지출액입니다.")
            
            st.markdown("---")
            col_ret3, col_ret4 = st.columns(2)
            inflation = col_ret3.number_input("평시 물가 상승률(%)", 0.0, 10.0, 2.5, step=0.1, key='in_inf', help="물가 상승으로 인한 구매력 하락을 자산 궤적에 역산하여 반영합니다.")
            tax_fee_rate = col_ret4.number_input("거래세/수수료(연%)", 0.0, 5.0, 0.5, step=0.1, key='in_tax', help="매매 수수료 및 거래세 등을 매년 선제적으로 삭감합니다.")

    with c3:
        with st.container(border=True):
            st.subheader("📈 3. 자산 및 변동성 설정 (Phase 전환)")
            current_asset = st.number_input("현재 금융자산 (만원)", 0, value=97000, step=100, key='in_asset', help="대출이 포함된 총 운용 자산입니다. 기대수익률은 대출이자 및 제세금이 블렌딩된 수치로 간주합니다.")
            
            st.markdown("###### ⚔️ 은퇴 전 (공격형 퀀트)")
            col_pre1, col_pre2 = st.columns(2)
            expected_return_pre = col_pre1.number_input("기대수익률(%)", 0.0, 30.0, 16.0, step=0.5, key='in_ret_pre', help="근로 소득이 있을 때의 공격적 퀀트 수익률 타겟입니다. (사전 세금/마찰비용 블렌딩 반영)")
            vol_pre = col_pre2.number_input("변동성(%)", 0.0, 50.0, 17.5, step=1.0, key='in_vol_pre', help="퀀트 시스템 트레이더의 야수성을 반영한 변동성입니다.")
            
            st.markdown("###### 🛡️ 은퇴 후 (방어형 7:3 스위칭)")
            col_post1, col_post2 = st.columns(2)
            expected_return_post = col_post1.number_input("기대수익률(%)", 0.0, 30.0, 12.0, step=0.5, key='in_ret_post', help="A전략 7:3(주식/채권) + B/C전략 유지 시 계좌 전체의 합산 기대수익률입니다.")
            vol_post = col_post2.number_input("변동성(%)", 0.0, 50.0, 11.0, step=1.0, key='in_vol_post', help="채권 편입으로 인한 비상관성(0) 효과로 전체 계좌의 하방 변동성이 11% 수준으로 통제됩니다.")

    st.markdown("---")

    with st.expander("🔥 **블랙 스완 & 고도화 세팅 (Advanced)**", expanded=True):
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1.container(border=True):
            st.markdown("##### 🏖️ 라이프스타일 최적화")
            dwz_mode = st.checkbox("🔥 Die with Zero 최적화 (파산 확률 20% 타겟)", value=True, key='in_dwz', help="체크 시 기본 15%에서 20%로 타겟 확률을 상향 조정하며 효용을 극대화합니다.")
            use_flex_spending = st.checkbox("🧠 다단계 생존 본능 (시장 5% 하락 시 긴축)", value=True, key='in_flex', help="시장이 하락할 때 지출을 통제하는 동적 인출 로직을 작동시킵니다.")
            use_glide_path = st.checkbox("🛬 동적 글라이드 패스 (은퇴 5년 전부터 연착륙)", value=True, key='in_glide', help="은퇴 시점에 임박하여 수익률과 변동성을 5년에 걸쳐 계단식으로 부드럽게 낮춥니다.")

        with c_adv2.container(border=True):
            st.markdown("##### 🚨 극한 스트레스 테스팅")
            use_fat_tail = st.checkbox("📉 팻 테일(Fat Tail) 대폭락장 적용", value=True, key='in_fat', help="정규 분포를 벗어나는 극단적인 금융위기급 폭락 확률을 시뮬레이션에 포함합니다.")
            use_inflation_shock = st.checkbox("🔥 스태그플레이션 (수익률 영구 타격)", value=True, key='in_shock', help="3년간 고물가 및 수익률 하락이 동시에 오는 블랙스완 시나리오를 적용합니다.")

    st.markdown("---")
    st.subheader("📱 4. 이벤트성 추가 수입/지출")
    tab1, tab2 = st.tabs(["💸 일회성 목돈 (상속/구입 등)", "📅 기간성 수입/지출 (연금/초과근무 등)"])
    
    with tab1:
        if 'lump_df' not in st.session_state:
            st.session_state.lump_df = pd.DataFrame([
                {"나이": 41, "유형": "지출", "내용": "대출상환", "금액(만원)": 10000},
                {"나이": 41, "유형": "수입", "내용": "상속", "금액(만원)": 2000},
                {"나이": 50, "유형": "지출", "내용": "주택구입", "금액(만원)": 32000},
                {"나이": 51, "유형": "수입", "내용": "상속", "금액(만원)": 2000},
                {"나이": 61, "유형": "수입", "내용": "상속", "금액(만원)": 5000},
                {"나이": 71, "유형": "수입", "내용": "상속", "금액(만원)": 5000}
            ])
        edited_lump_df = st.data_editor(st.session_state.lump_df, num_rows="dynamic", use_container_width=True,
                                        column_config={"유형": st.column_config.SelectboxColumn("유형", options=["수입", "지출"])})
        clean_lump_df = edited_lump_df.dropna(subset=['나이', '유형', '금액(만원)'])
        
    with tab2:
        if 'recur_df' not in st.session_state:
            st.session_state.recur_df = pd.DataFrame([
                {"시작나이": 40, "기간(년)": 20, "유형": "지출", "내용": "부모님용돈", "월금액(만원)": 100, "확정연금": False, "물가연동": False},
                {"시작나이": 40, "기간(년)": 10, "유형": "수입", "내용": "주6일 초과근무", "월금액(만원)": 200, "확정연금": False, "물가연동": False}, 
                {"시작나이": 47, "기간(년)": 5, "유형": "지출", "내용": "자동차 할부금", "월금액(만원)": 250, "확정연금": False, "물가연동": False}, 
                {"시작나이": 60, "기간(년)": 30, "유형": "지출", "내용": "지역 건보료 폭탄", "월금액(만원)": 50, "확정연금": False, "물가연동": True},
                {"시작나이": 70, "기간(년)": 20, "유형": "수입", "내용": "국민연금", "월금액(만원)": 100, "확정연금": True, "물가연동": True},
                {"시작나이": 70, "기간(년)": 20, "유형": "수입", "내용": "주택연금", "월금액(만원)": 100, "확정연금": True, "물가연동": True} 
            ])
        edited_recur_df = st.data_editor(st.session_state.recur_df, num_rows="dynamic", use_container_width=True,
                                         column_config={"유형": st.column_config.SelectboxColumn("유형", options=["수입", "지출"]),
                                                        "확정연금": st.column_config.CheckboxColumn("확정연금", help="주가에 상관없이 평생 지급되는 안정적 수입"),
                                                        "물가연동": st.column_config.CheckboxColumn("물가연동", help="체크 시 매년 물가상승률만큼 수입/지출액 증가. (명목 고정액은 체크 해제)")})
        clean_recur_df = edited_recur_df.dropna(subset=['시작나이', '기간(년)', '유형', '월금액(만원)'])

    if st.button("🚀 5,000회 연산 및 정밀 스트레스 테스트 시작", type="primary", use_container_width=True):
        st.divider()
        n_sims = 5000  
        params = {
            'current_age': current_age, 'death_age': death_age, 'current_asset': current_asset,
            'monthly_income': monthly_income, 'apply_income_inflation': apply_income_inflation,
            'monthly_expense': monthly_expense, 
            'expected_return_pre': expected_return_pre, 'vol_pre': vol_pre,
            'expected_return_post': expected_return_post, 'vol_post': vol_post,
            'inflation': inflation, 'tax_fee_rate': tax_fee_rate, 'retire_age': retire_age,
            'lump_events': clean_lump_df, 'recurring_events': clean_recur_df,
            'use_fat_tail': use_fat_tail, 
            'use_inflation_shock': use_inflation_shock, 'use_flex_spending': use_flex_spending,
            'dwz_mode': dwz_mode,
            'use_glide_path': use_glide_path
        }

        with st.spinner("복합 조세 모듈 및 글라이드 패스 연산 수행 중..."):
            simulator = FinancialSimulator(params)
            years, main_pv, main_nom, main_returns, safe_extra, base_ruin, stress_df, t_ruin = simulator.run_hybrid_analysis(main_sims=n_sims, search_sims=500)
            sens_df = simulator.run_sensitivity(base_ruin, sims=2000)

            total_pension = 0
            if '확정연금' in clean_recur_df.columns:
                pension_df = clean_recur_df[(clean_recur_df['유형'] == '수입') & (clean_recur_df['확정연금'] == True)]
                total_pension = pension_df['월금액(만원)'].sum()
            defense_rate = (total_pension / monthly_expense * 100) if monthly_expense > 0 else 0

            st.info("💡 **[가치 평가 기준]** 본 시뮬레이터의 모든 결괏값은 인플레이션을 역산한 **'현재 체감 구매력(Present Value)'** 기준으로 완벽히 변환되어 표시됩니다.")

            if safe_extra > 0:
                st.markdown(f"""
                <div class='yolo-box'>
                    <p class='yolo-title'>💰 파산 확률 {t_ruin:.0f}% 방어선 통과</p>
                    <p class='yolo-value'>이번 달 추가로 써도 되는 욜로(YOLO) 예산 = 월 {safe_extra:,}만 원</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"⚠️ **안전 마진 없음:** 기본 파산 확률이 {base_ruin:.1f}%로 타겟({t_ruin:.0f}%)을 초과합니다. 지출 통제가 시급합니다.")

            st.session_state['sim_results'] = {
                'years': years, 'pv': main_pv, 'nom': main_nom, 'returns': main_returns,
                'n_sims': n_sims, 'safe_extra': safe_extra, 'base_ruin': base_ruin,
                'stress_df': stress_df, 'sens_df': sens_df,
                'dwz_mode': dwz_mode, 't_ruin': t_ruin, 'defense_rate': defense_rate, 'lump_df': clean_lump_df,
                'retire_age': retire_age
            }

    if 'sim_results' in st.session_state:
        res = st.session_state['sim_results']
        years, sim_assets_pv, sim_returns = res['years'], res['pv'], res['returns']
        base_ruin, stress_df, sens_df = res['base_ruin'], res['stress_df'], res['sens_df']
        is_dwz, target_ruin = res['dwz_mode'], res['t_ruin']
        res_lump_df = res['lump_df']
        tgt_retire = res['retire_age']

        final_assets = sim_assets_pv[:, -1]
        top10_idx = np.abs(final_assets - np.percentile(final_assets, 90)).argmin()
        median_idx = np.abs(final_assets - np.percentile(final_assets, 50)).argmin()
        bot10_idx = np.abs(final_assets - np.percentile(final_assets, 10)).argmin()

        with st.expander(f"⏳ [멘탈 방어] 구간별 승률(Rolling Window)", expanded=False):
            st.markdown("###### 📊 우주를 분석한 '보유 기간별' 시스템 승률 (평균회귀 10% 작용)")
            c_r1, c_r2, c_r3, c_r4 = st.columns(4)
            r_windows = [1, 3, 5, 10]
            r_cols = [c_r1, c_r2, c_r3, c_r4]
            for r_w, r_c in zip(r_windows, r_cols):
                w_rate, m_cagr = calc_rolling_stats(sim_returns, r_w)
                r_c.metric(f"{r_w}년 유지 시 승률", f"{w_rate:.1f}%", f"해당 구간 연평균: {m_cagr:.2f}%", delta_color="off")
            st.caption("※ 평균 회귀(Mean Reversion) 로직이 탑재되어, 기간이 길어질수록 수익률이 기댓값에 수렴합니다. (강도: 10%)")

        paths = {
            "상위 10% (운수 좋은 날)": {"ret": sim_returns[top10_idx, :], "pv": sim_assets_pv[top10_idx, :]},
            "중간값 (가장 현실적)": {"ret": sim_returns[median_idx, :], "pv": sim_assets_pv[median_idx, :]},
            "하위 10% (스트레스)": {"ret": sim_returns[bot10_idx, :], "pv": sim_assets_pv[bot10_idx, :]}
        }
        with st.expander(f"📊 [심층 분석] {tgt_retire}세(은퇴) 도달 시점 시나리오별 자산 궤적 3종 비교", expanded=False):
            st.markdown(f"**총 {res['n_sims']:,}번의 평행우주 중, 자산 성과 기준 대표 궤적입니다.**")
            c_m1, c_m2, c_m3 = st.columns(3)
            cols = [c_m1, c_m2, c_m3]
            comp_data = {"나이": years}
            target_age_idx = years.index(tgt_retire) if tgt_retire in years else -1

            for i, (label, data) in enumerate(paths.items()):
                ret_array = data["ret"]
                pv_array = data["pv"]
                cagr = (np.prod(1 + ret_array) ** (1 / len(years)) - 1) * 100
                tgt_pv_eok = pv_array[target_age_idx] / 100000000 if target_age_idx != -1 else 0
                cols[i].metric(label, f"{tgt_retire}세 자산 {tgt_pv_eok:.1f}억 원", f"연평균(CAGR): {cagr:.2f}%", delta_color="off")

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

        st.markdown("<br>", unsafe_allow_html=True)
        g_col, d_col = st.columns([2.5, 1.2])

        with g_col:
            colors = ['#27AE60' if val <= target_ruin + 0.01 else '#F1C40F' if val < target_ruin + 10 else '#E74C3C' for val in stress_df['파산 확률(%)']]
            fig_stress = go.Figure(data=[go.Bar(x=stress_df['라벨'], y=stress_df['파산 확률(%)'], marker_color=colors, text=[f"{val:.1f}%" for val in stress_df['파산 확률(%)']], textposition='auto')])
            title_suffix = f"(81세 컷오프 & {target_ruin:.0f}% 방어)" if is_dwz else f"({target_ruin:.0f}% 방어)"
            fig_stress.update_layout(title=f"<b>월 여유 생활비별 파산 확률 {title_suffix}</b>", yaxis_title="파산 확률 (%)", height=300, plot_bgcolor='rgba(252, 252, 252, 1)', margin=dict(l=20, r=20, t=40, b=20))
            fig_stress.add_hline(y=target_ruin, line_dash="dot", line_color="green", annotation_text=f"안전 방어선 ({target_ruin:.0f}%)")
            with st.container(border=True):
                st.plotly_chart(fig_stress, use_container_width=True)

            st.markdown(f"##### 📈 메인 자산 궤적 ({tgt_retire}세 1차 방어선 집중)")
            median_pv = np.median(sim_assets_pv, axis=0) / 100000000
            top_10_pv = np.percentile(sim_assets_pv, 90, axis=0) / 100000000
            bottom_10_pv = np.percentile(sim_assets_pv, 10, axis=0) / 100000000

            idx_target = years.index(tgt_retire) if tgt_retire in years else -1

            k1, k2, k3 = st.columns(3)
            k1.metric("90세 최종 파산 확률", f"{base_ruin:.1f}%")
            k2.metric(f"{tgt_retire}세 예상 자산 (중앙값)", f"{median_pv[idx_target]:.2f}억 원")
            k3.metric(f"최악의 경우 (하위 10%)", f"{bottom_10_pv[idx_target]:.2f}억 원")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years+years[::-1], y=np.concatenate([top_10_pv, bottom_10_pv[::-1]]), fill='toself', fillcolor='rgba(46, 134, 193, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='신뢰구간(10~90%)', hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=years, y=median_pv, line=dict(color='#2E86C1', width=3), name='중앙값', hovertemplate='%{y:.2f}억 원<extra></extra>'))
            fig.add_trace(go.Scatter(x=years, y=bottom_10_pv, line=dict(color='#E74C3C', width=2, dash='dot'), name='하위 10%', hovertemplate='%{y:.2f}억 원<extra></extra>'))
            fig.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)

            if tgt_retire in years:
                fig.add_vline(x=tgt_retire, line_dash="dash", line_color="#95a5a6", annotation_text="은퇴 & 수비형 전환")
            if is_dwz and 81 in years:
                fig.add_vline(x=81, line_dash="dot", line_color="#9b59b6", annotation_text="사치 종료")

            for _, row in res_lump_df.iterrows():
                if row['금액(만원)'] >= 10000 and row['나이'] in years:
                    fig.add_vline(x=row['나이'], line_dash="dot", line_color="#f39c12", annotation_text=row['내용'])

            fig.update_layout(xaxis_title="나이", yaxis_title="현재 체감 자산 (억 원)", height=450, plot_bgcolor='rgba(252, 252, 252, 1)', hovermode="x unified", margin=dict(t=20, l=10, r=10))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### 🌪️ 변수 민감도 분석 (파산 트리거)")
            sens_df_sorted = sens_df.sort_values(by="충격(%)", key=abs, ascending=True)
            t_colors = ['#E74C3C' if val > 0 else '#27AE60' for val in sens_df_sorted['충격(%)']]
            fig_torn = go.Figure(go.Bar(
                x=sens_df_sorted['충격(%)'], y=sens_df_sorted['시나리오'], orientation='h', marker_color=t_colors,
                text=[f"+{v:.1f}%p" if v > 0 else f"{v:.1f}%p" for v in sens_df_sorted['충격(%)']], textposition='auto'
            ))
            fig_torn.update_layout(title="<b>해당 사건 발생 시 '파산 확률' 증감 폭</b>", xaxis_title="파산 확률 변동 폭 (%p)", height=250, plot_bgcolor='rgba(252, 252, 252, 1)', margin=dict(l=20, r=20, t=40, b=20))
            fig_torn.add_vline(x=0, line_width=2, line_color="#333333")
            with st.container(border=True):
                st.plotly_chart(fig_torn, use_container_width=True)

        with d_col:
            with st.container(border=True):
                st.subheader("💡 퀀트 코어 엔진: V57 튜닝 로직")
                st.info(f"""
                **1. 자산 평가 및 연금 방어율 (PV Discounting)**
                모든 시뮬레이션 결과값은 인플레이션을 역산한 **'현재 체감 구매력'**입니다. 현재 월 필수 지출 대비 확정 연금(국민/주택)의 방어율은 **{res['defense_rate']:.1f}%**입니다.

                **2. 기계적 매매 마찰 비용 (Slippage Decay)**
                수익률 모델링과 별개로, 자산 규모가 10억 원을 초과할 때마다 연 4회 리밸런싱에서 발생하는 호가 스프레드 비용을 수식(`0.015 * log10(자산/10억)`)에 따라 매년 자산에서 확정 삭감합니다.

                **3. 상하방 평균 회귀 (Mean Reversion - 10%)**
                자본 시장의 중력을 모사한 자기회귀(AR-1) 모델이 적용되었습니다. 전년도 시장이 폭등하거나 폭락하면, 다음 해의 기대수익률은 기계적으로 역방향(10%)으로 끌어당겨져 비현실적인 추세를 차단합니다. 

                **4. 스태그플레이션 영구 충격 (Zero-Sum Removed)**
                초인플레이션 발작 시, 3년간 수익률이 하락하고 변동성이 폭증합니다. 비현실적인 V자 반등(보상) 로직을 제거하여 계좌가 입은 영구적 타격을 현실적으로 반영합니다.

                **5. 다단계 생존 본능 (Dynamic Withdrawal)**
                계좌 잔고가 아닌 순수 시장 주가지수가 전고점 대비 5% 하락할 때마다 사치(YOLO) 지출을 20%씩 강제 삭감합니다. 
                """)

if __name__ == '__main__':
    main()
