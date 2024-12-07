import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# 변수 설명을 위한 함수 정의
def get_variable_description(var):
    descriptions = {
        'BMI': '체질량 지수 (kg/m²)',
        'Avg_Sleep': '평균 수면 시간 (시간)',
        'Avg_Smartphone_Time': '하루 평균 스마트폰 사용 시간 (시간)',
        'PA_TOT': '일주일간 신체활동 일수',
        'F_BR': '일주일간 아침식사 일수',
        'F_FRUIT': '일주일간 과일 섭취 일수',
        'PR_BI': '체형 인식 (1: 매우 마름 ~ 5: 매우 비만)',
        'M_SAD': '슬픔 빈도 (1: 없음 ~ 4: 매일)',
        'M_LON': '외로움 정도 (1: 전혀 없음 ~ 4: 항상)',
        'V_TRT': '학교폭력 경험 횟수',
        'E_S_RCRD': '학업 성적 (1: 상 ~ 5: 하)',
        'E_SES': '경제적 상태 (1: 상 ~ 5: 하)'
    }
    return descriptions.get(var, var)

# 캐시 함수 정의
@st.cache_data
def load_data():
    file_path = 'kyrbs2023.sav'
    df, meta = pyreadstat.read_sav(file_path)
    return df, meta

@st.cache_data
def preprocess_data(df):
    # 기존의 전처리 로직을 함수로 분리
    df = df.copy()
    df['AGE'] = 2024 - df['AGE'] + 1
    df['BMI'] = df['WT'] / ((df['HT'] / 100) ** 2)
    df['Avg_Sleep'] = ((df['M_WK_HR'] + df['M_WK_MM'] / 60) - (df['M_SLP_HR'] + df['M_SLP_MM'] / 60) +
                       (df['M_WK_HR_K'] + df['M_WK_MM_K'] / 60) - (df['M_SLP_HR_K'] + df['M_SLP_MM_K'] / 60)) / 2
    df['Avg_Smartphone_Time'] = (df['INT_SPWD_TM'] + df['INT_SPWK_TM']) / 2
    df['M_STR'] = 6 - df['M_STR']
    return df

@st.cache_data
def get_clean_data(_df, clean_data=True, selected_features=None, dependent_var='M_STR'):
    df = _df.copy()
    
    if clean_data:
        # 결측치 처리
        if selected_features:
            for var in [dependent_var] + selected_features:
                if var in df.columns:
                    df = df[~df[var].isin([9999, 8888])]
        
        # 연속형 변수 이상치 처리
        continuous_vars = ['BMI', 'Avg_Sleep', 'Avg_Smartphone_Time', 'PA_TOT']
        for var in continuous_vars:
            if var in df.columns and (not selected_features or var in selected_features):
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]
    
    return df

@st.cache_data
def calculate_correlations(df, independent_vars, dependent_var='M_STR'):
    correlations = []
    for var in independent_vars:
        if var in df.columns:
            corr = df[dependent_var].corr(df[var])
            correlations.append(corr)
    
    corr_df = pd.DataFrame({
        '변수명': independent_vars,
        '변수': [variable_labels[var] for var in independent_vars],
        '상관계수': correlations
    })
    return corr_df

# 1. 데이터 로드
df, meta = load_data()

# 2. 변수 생성
df = preprocess_data(df)

# Streamlit App
st.title("청소년 스트레스에 미치는 영향 분석")

# 탭 설정
tabs = st.tabs(["응답자 정보", "종속 변인 시각화", "독립 변인 시각화", "상관관계 분석", "머신러닝 예측"])

# 첫 번째 탭: 응답자 정보 시각화
with tabs[0]:
    st.header("응답자 정보 시각화")
    
    # 나이 분포
    st.subheader("나이 분포")
    df['나이'] = 2024 - df['AGE'] + 1
    age_hist = px.histogram(df, x='나이', nbins=10, title="나이 분포")
    st.plotly_chart(age_hist)
    age_counts = df['나이'].value_counts().sort_index()
    st.dataframe(pd.DataFrame({'나이': age_counts.index, '인원': age_counts.values}), use_container_width=True)

    # 성별 분포
    st.subheader("성별 분포")
    gender_map = {1: '남자', 2: '여자'}
    df['성별'] = df['SEX'].map(gender_map)
    gender_pie = px.pie(df, names='성별', title="성별 분포")
    st.plotly_chart(gender_pie)
    gender_counts = df['성별'].value_counts()
    st.dataframe(pd.DataFrame({'성별': gender_counts.index, '인원': gender_counts.values}), use_container_width=True)

    # 도시 분포
    st.subheader("도시 분포")
    city_counts = df['CITY'].value_counts().reset_index()
    city_counts.columns = ['도시', '응답자 수']
    city_bar = px.bar(city_counts, x='도시', y='응답자 수', title="도시 분포")
    st.plotly_chart(city_bar)
    st.dataframe(city_counts, use_container_width=True)


# 두 번째 탭: 종속 변인 시각화
with tabs[1]:
    st.header("종속 변인 시각화")
    graph_type = st.radio("그래프 유형 선택", ['원그래프', '막대그래프', '상자그림'], key='dependent_var_radio')

    # 스트레스 레이블 매핑 (순서 변경)
    stress_map = {
        5: '매우 많이 받음',
        4: '많이 받음',
        3: '약간 받음',
        2: '별로 받지 않음',
        1: '전혀 받지 않음'
    }
    df['스트레스'] = df['M_STR'].map(stress_map)
    
    # 순서 지정 (순서 변경)
    stress_order = ['매우 많이 받음', '많이 받음', '약간 받음', '별로 받지 않음', '전혀 받지 않음']
    
    if graph_type == '원그래프':
        stress_counts = df['스트레스'].value_counts().reindex(stress_order)
        stress_pie = px.pie(values=stress_counts.values, names=stress_counts.index, title="스트레스 정도 분포")
        st.plotly_chart(stress_pie)
    elif graph_type == '막대그래프':
        stress_counts = df['스트레스'].value_counts().reindex(stress_order).reset_index()
        stress_counts.columns = ['스트레스 정도', '응답자 수']
        stress_bar = px.bar(stress_counts, x='스트레스 정도', y='응답자 수', 
                           title="스트레스 정도 분포")
        st.plotly_chart(stress_bar)
    elif graph_type == '상자그림':
        stress_box = px.box(df, y='M_STR', title="스트레스 정도 상자그림")
        st.plotly_chart(stress_box)
    
    # 테이블 표시
    stress_counts = df['스트레스'].value_counts().reindex(stress_order)
    st.dataframe(pd.DataFrame({'스트레스 정도': stress_counts.index, '응답자 수': stress_counts.values}), 
                 use_container_width=True)

# 세 번째 탭: 독립 변인 시각화
with tabs[2]:
    st.header("독립 변인 시각화")
    
    # 변수 레이블 매핑 및 전처리
    variable_labels = {
        'BMI': 'BMI',
        'Avg_Sleep': '평균 수면 시간',
        'Avg_Smartphone_Time': '평균 스마트폰 사용 시간',
        'PA_TOT': '신체활동',
        'F_BR': '아침식사',
        'F_FRUIT': '과일섭취',
        'PR_BI': '체형인식',
        'M_SAD': '슬픔',
        'M_LON': '외로움',
        'V_TRT': '폭력경험',
        'E_S_RCRD': '학업성적',
        'E_SES': '경제상태'
    }
    
    # 각 변수별 값 매핑
    value_maps = {
        'F_BR': {1: '최근 7일 동안 먹지 않음',
                 2: '1-2일',
                 3: '3-4일',
                 4: '5-6일',
                 5: '매일'},
        'F_FRUIT': {1: '최근 7일 동안 먹지 않음',
                    2: '1-2일',
                    3: '3-4일',
                    4: '5-6일',
                    5: '매일'},
        'PR_BI': {1: '매우 마른 편',
                  2: '약간 마른 편',
                  3: '보통',
                  4: '약간 살이 찐 편',
                  5: '매우 살이 찐 편'},
        'M_SAD': {1: '없음',
                  2: '가끔',
                  3: '자주',
                  4: '매일'},
        'M_LON': {1: '전혀 느끼지 않음',
                  2: '가끔 느낌',
                  3: '자주 느낌',
                  4: '항상 느낌'},
        'E_S_RCRD': {1: '상',
                     2: '중상',
                     3: '중',
                     4: '중하',
                     5: '하'},
        'E_SES': {1: '상',
                  2: '중상',
                  3: '중',
                  4: '중하',
                  5: '하'}
    }

    variable = st.selectbox("변인 선택", list(variable_labels.keys()), 
                           format_func=lambda x: variable_labels[x])
    
    # 연속형 변수와 범주형 변수 구분
    continuous_vars = ['BMI', 'Avg_Sleep', 'Avg_Smartphone_Time', 'PA_TOT']
    
    if variable in continuous_vars:
        # 연속형 변수는 구간으로 나누어 시각화
        data = df[variable].dropna()
        bins = 10
        graph_type = st.radio("그래프 유형 선택", ['히스토그램', '상자그림'], 
                            key='independent_var_radio')
        
        if graph_type == '히스토그램':
            fig = px.histogram(df, x=variable, nbins=bins,
                             title=f"{variable_labels[variable]} 분포")
            st.plotly_chart(fig)
        else:
            fig = px.box(df, y=variable,
                        title=f"{variable_labels[variable]} 상자그림")
            st.plotly_chart(fig)
            
        # 술통계량 표시
        stats = pd.DataFrame({
            '통계량': ['평균', '표준편차', '최소값', '최대값'],
            '값': [data.mean(), data.std(), data.min(), data.max()]
        })
        st.dataframe(stats, use_container_width=True)
        
    else:
        # 범주형 변수는 원래대로 시각화
        if variable in value_maps:
            df[f'{variable}_label'] = df[variable].map(value_maps[variable])
            data = df[f'{variable}_label']
        else:
            data = df[variable]
            
        graph_type = st.radio("그래프 유형 선택", ['원그래프', '막대그래프'], 
                            key='independent_var_radio')
        
        counts = data.value_counts().sort_index()
        
        if graph_type == '원그래프':
            fig = px.pie(values=counts.values, names=counts.index,
                        title=f"{variable_labels[variable]} 분포")
        else:
            counts_df = counts.reset_index()
            counts_df.columns = [variable_labels[variable], '응답자 수']
            fig = px.bar(counts_df, x=variable_labels[variable], y='응답자 수',
                        title=f"{variable_labels[variable]} 분포")
            
        st.plotly_chart(fig)
        
        # 빈도표 표시
        freq_df = pd.DataFrame({
            variable_labels[variable]: counts.index,
            '응답자 수': counts.values,
            '비율(%)': (counts.values / len(df) * 100).round(1)
        })
        st.dataframe(freq_df, use_container_width=True)

# 네 번째 탭: 상관관계 분석
with tabs[3]:
    st.header("상관관계 분석")
    
    # 데이터 전처리
    st.subheader("데이터 전처리")
    clean_data = st.checkbox("결측치 및 이상치 제거", value=True,
                           help="결측치(9999, 8888) 및 이상치를 제거합니다.")
    
    # 분석용 데이터프레임 생성
    analysis_df = df.copy()
    
    # 분석할 변수들 선택
    dependent_var = 'M_STR'  # 종속변수 (스트레스)
    independent_vars = ['BMI', 'Avg_Sleep', 'Avg_Smartphone_Time', 'PA_TOT', 
                       'F_BR', 'F_FRUIT', 'PR_BI', 'M_SAD', 'M_LON', 
                       'V_TRT', 'E_S_RCRD', 'E_SES']
    
    # 결측치 및 이상치 처리
    if clean_data:
        # 결측치(9999, 8888) 처리
        for var in [dependent_var] + independent_vars:
            if var in df.columns:
                analysis_df = analysis_df[~analysis_df[var].isin([9999, 8888])]
        
        # 연속형 변수에 대해서만 이상치 제거
        continuous_vars = ['BMI', 'Avg_Sleep', 'Avg_Smartphone_Time', 'PA_TOT']
        for var in continuous_vars:
            if var in analysis_df.columns:
                Q1 = analysis_df[var].quantile(0.25)
                Q3 = analysis_df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                analysis_df = analysis_df[
                    (analysis_df[var] >= lower_bound) & 
                    (analysis_df[var] <= upper_bound)
                ]
    
    # 전처리 전후 데이터 수 비교
    st.write(f"전체 데이터 수: {len(df):,}개")
    st.write(f"전처리 후 데이터 수: {len(analysis_df):,}개")
    st.write(f"제거된 데이터 수: {len(df) - len(analysis_df):,}개")
    
    # 산점도 분석
    st.subheader("산점도 분석")
    selected_var = st.selectbox(
        "독립변인 선택", 
        independent_vars,
        format_func=lambda x: variable_labels[x]
    )
    
    # 회귀분석 수행
    from scipy import stats
    import numpy as np
    
    # 결측치가 있는 행 제거
    valid_data = analysis_df[[selected_var, dependent_var]].dropna()
    
    # numpy array로 변환하여 회귀분석 수행
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid_data[selected_var].values,
        valid_data[dependent_var].values
    )
    
    # 데이터 포인트 빈도 계산
    point_counts = valid_data.groupby([selected_var, dependent_var]).size().reset_index(name='count')
    max_count = point_counts['count'].max()
    
    # 산점도 생성 부분 수정
    scatter_fig = px.scatter(
        point_counts,  # 빈도가 계산된 데이터프레임 사용
        x=selected_var,
        y=dependent_var,
        size='count',  # 빈도에 따라 크기 조절
        size_max=20,   # 최대 크기 설정
        labels={
            selected_var: f"{variable_labels[selected_var]}\n({get_variable_description(selected_var)})",
            dependent_var: "스트레스 정도 (5: 매우 많이 받음 ~ 1: 전혀 받지 않음)",
            'count': '빈도'  # 범례 레이블 한글화
        },
        title=f'스트레스와 {variable_labels[selected_var]}의 관계',
        opacity=0.7,
        color='count',  # 빈도에 따 색상도 변경
        color_continuous_scale='Blues',  # 푸른색 계열로 변경
        hover_data={'count': True}  # 호버 시 빈도 표시
    )
    
    # 추세선 추가
    x_range = np.linspace(
        valid_data[selected_var].min(),
        valid_data[selected_var].max(),
        100
    )
    y_range = slope * x_range + intercept
    
    scatter_fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            name=f'추세선 (r={r_value:.3f})',
            mode='lines',
            line=dict(color='red', width=2)
        )
    )
    
    # 그래프 스타일링
    scatter_fig.update_layout(
        showlegend=True,
        xaxis_title=f"{variable_labels[selected_var]}\n({get_variable_description(selected_var)})",
        yaxis_title="스트레스 정도 (5: 매우 많이 받음 ~ 1: 전혀 받지 않음)",
        plot_bgcolor='rgba(255,255,255,1)',  # 완전한 흰색으로 설정
        paper_bgcolor='rgba(255,255,255,1)'  # 완전한 흰색으로 설정
    )
    
    # 격자 추가
    scatter_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='solid')
    scatter_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='solid')
    
    st.plotly_chart(scatter_fig)
    
    # 회귀식과 통계량 표시
    st.write(f"""
    **회귀분석 결과:**
    - 회귀식: y = {slope:.3f}x + {intercept:.3f}
    - 상관계수(r): {r_value:.3f}
    - 결정계수(R²): {r_value**2:.3f}
    - p-value: {p_value:.3f}
    """)
    
    # 선택된 변의 상관계수 표시
    current_corr = analysis_df[dependent_var].corr(analysis_df[selected_var])
    st.write(f"**현재 선택된 변수와의 상관계수: {current_corr:.3f}**")
    
    # 상관관계 분석 결과
    st.subheader("전체 상관관계 분석 결과")
    
    # 변수 선택 방식
    filter_method = st.radio(
        "변수 선택 방식",
        ["상관계수 절댓값 기준", "직접 선택"],
        key="filter_method"
    )
    
    # 상관계수 계산
    correlations = []
    for var in independent_vars:
        if var in analysis_df.columns:
            corr = analysis_df[dependent_var].corr(analysis_df[var])
            correlations.append(corr)
    
    # 상관계수 터프레임 생성
    corr_df = pd.DataFrame({
        '변수명': independent_vars,
        '변수': [variable_labels[var] for var in independent_vars],
        '상관계수': correlations
    })
    
    if filter_method == "상관계수 절댓값 기준":
        # 상관계수 절댓값 기준 설정
        corr_threshold = st.slider(
            "상관계수 절댓값 기준 선택",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="선택한 값 이상의 절댓값을 가진 상관계수만 표시됩니다."
        )
        
        # 기준에 따라 필터링
        filtered_corr_df = corr_df[abs(corr_df['상관계수']) >= corr_threshold]
        
    else:  # 직접 선택
        # 다중 선택 위젯
        selected_vars = st.multiselect(
            "분석할 변수 선택",
            options=independent_vars,
            default=independent_vars,
            format_func=lambda x: f"{variable_labels[x]} (상관계수: {corr_df[corr_df['변수명']==x]['상관계수'].values[0]:.3f})"
        )
        
        # 선택된 변수들만 필터
        filtered_corr_df = corr_df[corr_df['변수명'].isin(selected_vars)]
    
    # 상관계수 절댓 기으로 정렬
    filtered_corr_df = filtered_corr_df.sort_values('상관계수', key=abs, ascending=False)
    
    # 선택된 변수들의 수 표시
    st.write(f"선택된 변수 수: {len(filtered_corr_df)}개")
    
    # 히트맵 생성
    if not filtered_corr_df.empty:
        # 히트맵을 위한 데이터 준비
        heatmap_data = pd.DataFrame([filtered_corr_df['상관계수'].values], 
                                  columns=filtered_corr_df['변수'].values)
        
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdBu_r',  # 양수는 빨강, 음수는 파랑
            aspect='auto',
            title='스트레스와 독립변인 간의 상관관계'
        )
        
        # 그래프 스타일링
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="스트레스",
            xaxis_title="독립변인",
            title={
                'text': '스트레스와 독립변인 간의 상관관계<br><sup>스트레스: 5(매우 많이 받음) ~ 1(전혀 받지 않음)</sup>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        # 히트맵에 상관계수 값 표시
        for i in range(len(filtered_corr_df)):
            fig.add_annotation(
                x=i,
                y=0,
                text=f"{filtered_corr_df['상관계수'].iloc[i]:.3f}",
                showarrow=False,
                font=dict(color='black')
            )
        
        st.plotly_chart(fig)
        
        # 상관계수 상세 정보
        st.subheader("상관계수 상세 정보")
        display_df = filtered_corr_df[['변수', '상관계수']].copy()
        display_df['상관관계 강도'] = display_df['상관계수'].apply(
            lambda x: '강한 상관관계' if abs(x) >= 0.7 else 
                     '중간 정도의 상관관계' if abs(x) >= 0.3 else 
                     '약한 상관관계'
        )
        display_df['상관관계 방향'] = display_df['상관계수'].apply(
            lambda x: '정적 상관' if x > 0 else '부적 상관' if x < 0 else '무상관'
        )
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("택한 기준을 만족하는 변수가 없습니다.")
    
    # 상관관계 해석
    st.subheader("상관관계 해석")
    st.write("""
    * 상관계수는 -1에서 1 사이의 값을 가지며, 절댓값이 클수록 강한 상관관계를 나타냅니다.
    * 양수는 정적 상관(한 변수가 증가하면 스트레스도 증가)을 나냅니다.
    * 음수는 부적 상관(한 변수가 증가하면 스트레스는 감소)을 나타냅니다.
    * |r| ≥ 0.7: 강한 상관관계
    * 0.3 ≤ |r| < 0.7: 중간 정도의 상관관계
    * |r| < 0.3: 약한 상관관계
    """)

# 다섯 번째 탭: 머신러닝 예측
with tabs[4]:
    st.header("머신러닝을 통한 스트레스 예측")
    
    # 데이터 전처리 옵션
    st.subheader("데이터 전처리")
    clean_data = st.checkbox("결측치 및 이상치 제거", value=True,
                           help="결측치(9999, 8888) 및 이상치를 제거합니다.",
                           key="ml_clean_data")  # 고유한 키 추가
    
    # 분석용 데이터프레임 생성
    analysis_df = df.copy()
    
    # 결측치 및 이상치 처리
    if clean_data:
        # 결측치(9999, 8888) 처리
        for var in [dependent_var] + independent_vars:
            if var in df.columns:
                analysis_df = analysis_df[~analysis_df[var].isin([9999, 8888])]
        
        # 연속형 변수에 대해서만 이상치 제거
        continuous_vars = ['BMI', 'Avg_Sleep', 'Avg_Smartphone_Time', 'PA_TOT']
        for var in continuous_vars:
            if var in analysis_df.columns:
                Q1 = analysis_df[var].quantile(0.25)
                Q3 = analysis_df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                analysis_df = analysis_df[
                    (analysis_df[var] >= lower_bound) & 
                    (analysis_df[var] <= upper_bound)
                ]
    
    # 전처리 전후 데이터 수 비교
    st.write(f"전체 데이터 수: {len(df):,}개")
    st.write(f"전처리 후 데이터 수: {len(analysis_df):,}개")
    st.write(f"제거된 데이터 수: {len(df) - len(analysis_df):,}개")
    
    # 분석 방법 선택
    analysis_method = st.radio(
        "분석 방법 선택",
        ["이진 분류 (높음/낮음)", "3단계 분류 (상/중/하)", "회귀 분석 (1-5점)"],
        index=0,
        help="스트레스 예측을 위한 분석 방법을 선택하세요."
    )
    
    # 예측 변수 선택
    st.subheader("예측 변수 선택")
    selected_features = st.multiselect(
        "예측에 사용할 변수 선택",
        options=independent_vars,
        default=[var for var in independent_vars if abs(corr_df[corr_df['변수명']==var]['상관계수'].values[0]) >= 0.1],
        format_func=lambda x: f"{variable_labels[x]} (상관계수: {corr_df[corr_df['변수명']==x]['상관계수'].values[0]:.3f})"
    )
    
    if len(selected_features) > 0:
        # 클린 데이터 얻기
        data_for_ml = get_clean_data(df, clean_data, selected_features, dependent_var)
        
        # 선택된 분석 방법에 따라 종속변수 환
        if analysis_method == "이진 분류 (높음/낮음)":
            data_for_ml[dependent_var] = (data_for_ml[dependent_var] >= 3).astype(int)
            st.info("스트레스 수준을 이진 분류로 변환: 0(낮음: 1-2점), 1(높음: 3-5점)")
            
        elif analysis_method == "3단계 분류 (상/중/하)":
            data_for_ml[dependent_var] = data_for_ml[dependent_var].apply(
                lambda x: 0 if x <= 2 else (2 if x >= 4 else 1)
            )
            st.info("스트레스 수준을 3단계 분류: 0(하: 1-2점), 1(중: 3점), 2(상: 4-5점)")
        
        # 결측치 처리
        data_for_ml = data_for_ml.dropna()
        
        if len(data_for_ml) > 0:
            X = data_for_ml[selected_features]
            y = data_for_ml[dependent_var]
            
            # 모델 선택 (분석 방법에 따라 다른 모델 제공)
            if analysis_method == "회귀 분석 (1-5점)":
                models = {
                    "선형 회귀": LinearRegression(),
                    "랜덤 포레스트": RandomForestRegressor(n_estimators=100, random_state=42),
                    "XGBoost": XGBRegressor(random_state=42)
                }
            else:
                models = {
                    "로지스틱 회귀": LogisticRegression(random_state=42),
                    "랜덤 포레스트": RandomForestClassifier(n_estimators=100, random_state=42),
                    "XGBoost": XGBClassifier(random_state=42)
                }
            
            model_type = st.selectbox("머신러닝 모델 선택", list(models.keys()))
            # 랜덤 포레스트 선택 시 안내 메시지
            if model_type == "랜덤 포레스트":
                st.warning("⚠️ 랜덤 포레스트 모델은 다른 모델들에 비해 학습 시간이 오래 걸릴 수 있습니다. " 
                          "특히 하이퍼파라미터 튜닝을 사용할 경우 수 분이 소요될 수 있습니다.")
            
            # 하이퍼파라미터 튜닝 여부 선택
            use_hyperparameter_tuning = st.checkbox("하이퍼파라미터 튜닝 사용", value=True,
                                                  help="체크를 해제하면 기본 하이퍼파라미터를 사용합니다.")

            
            if use_hyperparameter_tuning:
                if analysis_method == "회귀 분석 (1-5점)":
                    if model_type == "선형 회귀":
                        param_grid = {
                            'fit_intercept': [True, False],
                            'positive': [True, False]
                        }
                    elif model_type == "랜덤 포레스트":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20, 30],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                    else:  # XGBoost
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.3],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                else:  # 분류 분석
                    if model_type == "로지스틱 회귀":
                        param_grid = {
                            'C': [0.001, 0.01, 0.1, 1, 10],
                            'penalty': ['l1', 'l2'],
                            'solver': ['liblinear', 'saga']
                        }
                    elif model_type == "랜덤 포레스트":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20, 30],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                    else:  # XGBoost
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.3],
                            'subsample': [0.8, 0.9, 1.0]
                        }
            else:
                # 기본 하이퍼파라미터 설정
                if analysis_method == "회귀 분석 (1-5점)":
                    if model_type == "선형 회귀":
                        param_grid = {
                            'fit_intercept': [True],
                            'positive': [False]
                        }
                    elif model_type == "랜덤 포레스트":
                        param_grid = {
                            'n_estimators': [100],
                            'max_depth': [None],
                            'min_samples_split': [2],
                            'min_samples_leaf': [1]
                        }
                    else:  # XGBoost
                        param_grid = {
                            'n_estimators': [100],
                            'max_depth': [3],
                            'learning_rate': [0.1],
                            'subsample': [1.0]
                        }
                else:  # 분류 분석
                    if model_type == "로지스틱 회귀":
                        param_grid = {
                            'C': [1.0],
                            'penalty': ['l2'],
                            'solver': ['liblinear']
                        }
                    elif model_type == "랜덤 포레스트":
                        param_grid = {
                            'n_estimators': [100],
                            'max_depth': [None],
                            'min_samples_split': [2],
                            'min_samples_leaf': [1]
                        }
                    else:  # XGBoost
                        param_grid = {
                            'n_estimators': [100],
                            'max_depth': [3],
                            'learning_rate': [0.1],
                            'subsample': [1.0]
                        }

            # 교차 검증 폴드 수 설정
            cv_folds = st.slider("교차 검증 폴드 수", 
                        min_value=2, 
                        max_value=5,  # 최대값을 5로 변경
                        value=3,  # 기본값 설정
                        help="2-5 사이의 교차 검증 폴드 수를 선택하세요.")
            
            # 훈련/테스트 세트 분할
            test_size = st.slider("테스트 세트 비율", 0.1, 0.4, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if st.button("모델 학습 시작"):
                with st.spinner("모델 학습 중..."):
                    try:
                        # 진행 상황 표시를 위한 컨테이너
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # GridSearchCV의 n_splits 계산
                        n_splits = cv_folds
                        
                        # 모델 타입별 total_iterations 계산
                        if model_type == "선형 회귀":
                            total_iterations = len(param_grid['fit_intercept']) * len(param_grid['positive']) * n_splits
                        elif model_type == "로지스틱 회귀":
                            total_iterations = len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * n_splits
                        elif model_type == "랜덤 포레스트":
                            total_iterations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * n_splits
                        else:  # XGBoost
                            total_iterations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * n_splits
                        
                        # 진행 상황을 추적하기 위한 커스텀 scorer
                        current_iteration = [0]
                        def custom_scorer(estimator, X, y):
                            current_iteration[0] += 1
                            progress = int((current_iteration[0] / total_iterations) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"진행률: {progress}% ({current_iteration[0]}/{total_iterations} 완료)")
                            if analysis_method == "회귀 분석 (1-5점)":
                                return r2_score(y, estimator.predict(X))
                            else:
                                return accuracy_score(y, estimator.predict(X))

                        # GridSearchCV 객체 생성 및 학습
                        grid_search = GridSearchCV(
                            estimator=models[model_type],
                            param_grid=param_grid,
                            cv=cv_folds,
                            scoring=custom_scorer,
                            n_jobs=1  # 진행률 표시를 위해 단일 프로세스로 실행
                        )
                        
                        grid_search.fit(X_train, y_train)
                        
                        # 진행바 완료 표시
                        progress_bar.progress(100)
                        status_text.text("모델 학습이 완료되었습니다!")
                        
                        # 하이퍼파라미터 튜닝 여부에 따른 다른 메시지 표시
                        if use_hyperparameter_tuning:
                            st.subheader("최적 하이퍼파라미터")
                            st.write(grid_search.best_params_)
                        else:
                            st.subheader("사용된 하이퍼파라미터")
                            st.write("기본 하이퍼파라미터를 사용하여 모델을 학습했습니다.")
                            st.write(param_grid)
                        
                        # 최적 모델로 예측
                        y_pred = grid_search.predict(X_test)
                        
                        # 성능 평가
                        if analysis_method == "회귀 분석 (1-5점)":
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"{rmse:.3f}")
                            with col2:
                                st.metric("R²", f"{r2:.3f}")
                            with col3:
                                st.metric("MSE", f"{mse:.3f}")
                            
                            # 실제값 vs 예측값 산점도
                            fig = px.scatter(
                                x=y_test, y=y_pred,
                                labels={'x': '실제 스트레스', 'y': '예측 스트레스'},
                                title="실제 vs 예측값"
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=[y_test.min(), y_test.max()],
                                    y=[y_test.min(), y_test.max()],
                                    mode='lines',
                                    name='완벽한 예측',
                                    line=dict(color='red', dash='dash')
                                )
                            )
                            st.plotly_chart(fig)
                            
                            # 변수 중요도 시각화
                            st.subheader("변수 중요도")
                            
                            # 모델별 중요도 계산 방식 설명
                            if model_type == "선형 회귀":
                                st.write("""
                                **선형 회귀 모델의 변수 중요도:**
                                - 각 변수의 회귀 계수의 절댓값을 기준으로 산출
                                - 계수의 절댓값이 클수록 해당 변수가 예측에 미치는 영향이 큼
                                """)
                                importance = abs(grid_search.best_estimator_.coef_)
                            elif model_type == "로지스틱 회귀":
                                st.write("""
                                **로지스틱 회귀 모델의 변수 중요도:**
                                - 각 변수의 회귀 계수의 절댓값을 기준으로 산출
                                - 계수의 절댓값이 클수록 해당 변수가 분류에 미치는 영향이 큼
                                """)
                                importance = abs(grid_search.best_estimator_.coef_[0])
                            elif model_type == "랜덤 포레스트":
                                st.write("""
                                **랜덤 포레스트 모델의 변수 중요도:**
                                - 각 변수가 불순도(Impurity) 감소에 기여한 정도를 평균하여 산출
                                - 특정 변수로 인한 불순도 감소가 클수록 해당 변수의 중요도가 높음
                                - 0~1 사이의 값으로 정규화되며, 모든 변수의 중요도 합은 1
                                """)
                                importance = grid_search.best_estimator_.feature_importances_
                            else:  # XGBoost
                                st.write("""
                                **XGBoost 모델의 변수 중요도:**
                                - 각 변수가 트리 구성에 사용된 횟수와 성능 향상에 기여한 정도를 종합하여 산출
                                - 게인(Gain) 기준: 각 변수가 분기 기준으로 선택될 때 발생한 손실 감소량의 평균
                                - 0~1 사이의 값으로 정규화되며, 모든 변수의 중요도 합은 1
                                """)
                                importance = grid_search.best_estimator_.feature_importances_
                            
                            # 변수 중요도 데이터프레임 생성
                            importance_df = pd.DataFrame({
                                '변수': [variable_labels[feat] for feat in selected_features],
                                '중요도': importance
                            })
                            
                            # 중요도를 0-1 사이로 정규화 (선형/로지스틱 회귀의 경우)
                            if model_type in ["선형 회귀", "로지스틱 회귀"]:
                                importance_df['중요도'] = importance_df['중요도'] / importance_df['중요도'].sum()
                            
                            importance_df = importance_df.sort_values('중요도', ascending=True)
                            
                            # 가로 막대 그래프로 시각화
                            fig = px.bar(
                                importance_df,
                                x='중요도',
                                y='변수',
                                orientation='h',
                                title=f"변수 중요도 ({model_type})"
                            )
                            
                            fig.update_layout(
                                xaxis_title="중요도 (0~1 사이 정규화)",
                                yaxis_title="변수",
                                height=max(400, len(selected_features) * 30),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig)
                            
                            # 중요도 해석 추가
                            st.write("### 변수 중요도 해석")
                            top_3_vars = importance_df.sort_values('중요도', ascending=False).head(3)
                            
                            st.write("**상위 3개 중요 변수:**")
                            for idx, row in top_3_vars.iterrows():
                                st.write(f"- **{row['변수']}**: {row['중요도']:.3f} ({row['중요도']*100:.1f}%)")
                            
                            st.write("""
                            **참고사항:**
                            - 중요도는 0~1 사이로 정규화되어 있으며, 모든 변수의 중요도 합은 1입니다.
                            - 중요도가 높을수록 해당 변수가 예측/분류에 미치는 영향이 큽니다.
                            - 이 결과는 현재 선택된 데이터와 모델을 기준으로 산출된 것입니다.
                            """)
                        
                        else:  # 분류 분석
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("정확도", f"{accuracy:.3f}")
                            with col2:
                                st.metric("정밀도", f"{precision:.3f}")
                            with col3:
                                st.metric("재현율", f"{recall:.3f}")
                            with col4:
                                st.metric("F1 점수", f"{f1:.3f}")
                            
                            # 혼동 행렬
                            cm = confusion_matrix(y_test, y_pred)
                            if analysis_method == "이진 분류 (높음/낮음)":
                                labels = ['낮음', '높음']
                                labels_y = labels[::-1]
                                cm = np.flipud(cm)
                            else:  # 3단계 분류
                                labels = ['하', '중', '상']
                                labels_y = labels[::-1]
                                cm = np.flipud(cm)
                            
                            fig = px.imshow(
                                cm,
                                labels=dict(x="예측값", y="실제값", color="빈도"),
                                x=labels,
                                y=labels_y,
                                aspect='auto',
                                title="혼동 행렬"
                            )
                            
                            # 각 셀에 값 표시
                            for i in range(len(cm)):
                                for j in range(len(cm)):
                                    fig.add_annotation(
                                        x=j,
                                        y=i,
                                        text=str(cm[i, j]),
                                        showarrow=False,
                                        font=dict(color='white' if cm[i, j] > cm.max()/2 else 'black')
                                    )
                            
                            st.plotly_chart(fig)
                            
                            # 변수 중요도 시각화
                            st.subheader("변수 중요도")
                            
                            # 모델별 중요도 계산 방식 설명
                            if model_type == "선형 회귀":
                                st.write("""
                                **선형 회귀 모델의 변수 중요도:**
                                - 각 변수의 회귀 계수의 절댓값을 기준으로 산출
                                - 계수의 절댓값이 클수록 해당 변수가 예측에 미치는 영향이 큼
                                """)
                                importance = abs(grid_search.best_estimator_.coef_)
                            elif model_type == "로지스틱 회귀":
                                st.write("""
                                **로지스틱 회귀 모델의 변수 중요도:**
                                - 각 변수의 회귀 계수의 절댓값을 기준으로 산출
                                - 계수의 절댓값이 클수록 해당 변수가 분류에 미치는 영향이 큼
                                """)
                                importance = abs(grid_search.best_estimator_.coef_[0])
                            elif model_type == "랜덤 포레스트":
                                st.write("""
                                **랜덤 포레스트 모델의 변수 중요도:**
                                - 각 변수가 불순도(Impurity) 감소에 기여한 정도를 평균하여 산출
                                - 특정 변수로 인한 불순도 감소가 클수록 해당 변수의 중요도가 높음
                                - 0~1 사이의 값으로 정규화되며, 모든 변수의 중요도 합은 1
                                """)
                                importance = grid_search.best_estimator_.feature_importances_
                            else:  # XGBoost
                                st.write("""
                                **XGBoost 모델의 변수 중요도:**
                                - 각 변수가 트리 구성에 사용된 횟수와 성능 향상에 기여한 정도를 종합하여 산출
                                - 게인(Gain) 기준: 각 변수가 분기 기준으로 선택될 때 발생한 손실 감소량의 평균
                                - 0~1 사이의 값으로 정규화되며, 모든 변수의 중요도 합은 1
                                """)
                                importance = grid_search.best_estimator_.feature_importances_
                            
                            # 변수 중요도 데이터프레임 생성
                            importance_df = pd.DataFrame({
                                '변수': [variable_labels[feat] for feat in selected_features],
                                '중요도': importance
                            })
                            
                            # 중요도를 0-1 사이로 정규화 (선형/로지스틱 회귀의 경우)
                            if model_type in ["선형 회귀", "로지스틱 회귀"]:
                                importance_df['중요도'] = importance_df['중요도'] / importance_df['중요도'].sum()
                            
                            importance_df = importance_df.sort_values('중요도', ascending=True)
                            
                            # 가로 막대 그래프로 시각화
                            fig = px.bar(
                                importance_df,
                                x='중요도',
                                y='변수',
                                orientation='h',
                                title=f"변수 중요도 ({model_type})"
                            )
                            
                            fig.update_layout(
                                xaxis_title="중요도 (0~1 사이 정규화)",
                                yaxis_title="변수",
                                height=max(400, len(selected_features) * 30),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig)
                            
                            # 중요도 해석 추가
                            st.write("### 변수 중요도 해석")
                            top_3_vars = importance_df.sort_values('중요도', ascending=False).head(3)
                            
                            st.write("**상위 3개 중요 변수:**")
                            for idx, row in top_3_vars.iterrows():
                                st.write(f"- **{row['변수']}**: {row['중요도']:.3f} ({row['중요도']*100:.1f}%)")
                            
                            st.write("""
                            **참고사항:**
                            - 중요도는 0~1 사이로 정규화되어 있으며, 모든 변수의 중요도 합은 1입니다.
                            - 중요도가 높을수록 해당 변수가 예측/분류에 미치는 영향이 큽니다.
                            - 이 결과는 현재 선택된 데이터와 모델을 기준으로 산출된 것입니다.
                            """)
                        
                    except Exception as e:
                        st.error(f"모델 학습 중 오류가 발생했습니다: {str(e)}")
                        st.write("데이터를 확인하고 다시 시도해주세요.")
            
    else:
        st.warning("예측에 사용할 변수를 하나 이상 선택해주세요.")
