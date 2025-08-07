import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Konfiguracja strony
st.set_page_config(
    page_title="Find Friends",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stałe
MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        animation: fadeIn 0.6s ease-in;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .sidebar-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def check_files_exist():
    """Sprawdza czy wymagane pliki istnieją"""
    missing_files = []
    files_to_check = [MODEL_NAME, DATA, CLUSTER_NAMES_AND_DESCRIPTIONS]
    
    for file_name in files_to_check:
        if file_name == MODEL_NAME:
            if not (os.path.exists(f"{file_name}.pkl") or 
                   os.path.exists(f"{file_name}.joblib") or 
                   os.path.exists(file_name)):
                missing_files.append(file_name)
        else:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
    
    return missing_files

@st.cache_data
def get_model():
    """Ładowanie modelu ML z cache"""
    try:
        return load_model(MODEL_NAME)
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {str(e)}")
        return None

@st.cache_data
def get_cluster_names_and_descriptions():
    """Ładowanie opisów klastrów z cache"""
    try:
        with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
            return json.loads(f.read())
    except Exception as e:
        st.error(f"Błąd ładowania opisów klastrów: {str(e)}")
        return {
            "0": {"name": "Grupa 1", "description": "Opis grupy 1"},
            "1": {"name": "Grupa 2", "description": "Opis grupy 2"},
            "2": {"name": "Grupa 3", "description": "Opis grupy 3"}
        }

@st.cache_data
def get_all_participants():
    """Ładowanie i predykcja dla wszystkich uczestników"""
    try:
        all_df = pd.read_csv(DATA, sep=';')
        model = get_model()
        if model is not None:
            df_with_clusters = predict_model(model, data=all_df)
            return df_with_clusters
        else:
            return None
    except Exception as e:
        st.error(f"Błąd ładowania danych uczestników: {str(e)}")
        return None

def create_advanced_visualizations(same_cluster_df, all_df, predicted_cluster_id, cluster_names_and_descriptions):
    """Tworzenie zaawansowanych wizualizacji"""
    
    # 1. Dashboard metryczki
    st.subheader("📊 Statystyki Twojej grupy")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Twoja grupa</h3>
            <h2 style="color: #667eea;">{len(same_cluster_df)}</h2>
            <p>osób w grupie</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if not same_cluster_df.empty and 'age' in same_cluster_df.columns:
            avg_age_range = same_cluster_df['age'].mode()[0]
        else:
            avg_age_range = 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Dominujący wiek</h3>
            <h2 style="color: #667eea;">{avg_age_range}</h2>
            <p>najczęstsza grupa wiekowa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if not same_cluster_df.empty and 'edu_level' in same_cluster_df.columns:
            dominant_education = same_cluster_df['edu_level'].mode()[0]
        else:
            dominant_education = 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎓 Wykształcenie</h3>
            <h2 style="color: #667eea;">{dominant_education}</h2>
            <p>dominujący poziom</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if not same_cluster_df.empty and 'gender' in same_cluster_df.columns:
            gender_ratio = same_cluster_df['gender'].value_counts()
            dominant_gender = gender_ratio.index[0] if not gender_ratio.empty else 'N/A'
        else:
            dominant_gender = 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Dominująca płeć</h3>
            <h2 style="color: #667eea;">{dominant_gender}</h2>
            <p>najczęstsza płeć</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Porównanie z innymi grupami
    st.subheader("📊 Porównanie wszystkich grup")
    
    try:
        cluster_stats = all_df.groupby('Cluster').size().reset_index(name='count')
        cluster_stats['cluster_name'] = cluster_stats['Cluster'].apply(
            lambda x: cluster_names_and_descriptions.get(str(x), {}).get('name', f'Grupa {x}')
        )
        
        fig = px.bar(
            cluster_stats, 
            x='cluster_name', 
            y='count',
            color='count',
            color_continuous_scale='viridis',
            title="Rozmiary wszystkich grup",
            text='count'
        )
        fig.update_layout(
            xaxis_title="Nazwa grupy",
            yaxis_title="Liczba osób",
            showlegend=False,
            height=400,
            xaxis={'tickangle': 45}
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        # Podświetl aktualną grupę
        colors = ['#FF6B6B' if row['Cluster'] == predicted_cluster_id else '#4ECDC4' 
                 for _, row in cluster_stats.iterrows()]
        fig.update_traces(marker_color=colors)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Błąd podczas tworzenia wykresu porównawczego: {str(e)}")

def create_detailed_charts(same_cluster_df):
    """Tworzenie szczegółowych wykresów dla grupy"""
    
    if same_cluster_df.empty:
        st.warning("Brak danych do wyświetlenia szczegółowych wykresów.")
        return
    
    st.subheader("🔍 Szczegółowa analiza Twojej grupy")
    
    available_columns = same_cluster_df.columns.tolist()
    charts_to_create = []
    
    if 'age' in available_columns:
        charts_to_create.append(('age', 'Rozkład wieku', '#667eea'))
    if 'edu_level' in available_columns:
        charts_to_create.append(('edu_level', 'Wykształcenie', '#764ba2'))
    if 'fav_animals' in available_columns:
        charts_to_create.append(('fav_animals', 'Ulubione zwierzęta', '#f093fb'))
    if 'fav_place' in available_columns:
        charts_to_create.append(('fav_place', 'Ulubione miejsca', '#f5576c'))
    
    if len(charts_to_create) >= 2:
        rows = (len(charts_to_create) + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=[chart[1] for chart in charts_to_create[:4]],
            specs=[[{"type": "xy"}, {"type": "xy"}] for _ in range(rows)]
        )
        
        for i, (col_name, title, color) in enumerate(charts_to_create[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            try:
                counts = same_cluster_df[col_name].value_counts()
                fig.add_trace(
                    go.Bar(x=counts.index, y=counts.values, name=title, marker_color=color),
                    row=row, col=col
                )
            except Exception as e:
                st.error(f"Błąd podczas tworzenia wykresu dla {col_name}: {str(e)}")
        
        fig.update_layout(
            height=300 * rows,
            showlegend=False,
            title_text="Charakterystyka grupy - wszystkie aspekty"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        for col_name, title, color in charts_to_create:
            try:
                counts = same_cluster_df[col_name].value_counts()
                fig = px.bar(x=counts.index, y=counts.values, title=title)
                fig.update_traces(marker_color=color)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Błąd podczas tworzenia wykresu dla {col_name}: {str(e)}")

def show_cluster_recommendations(predicted_cluster_data, same_cluster_df):
    """Pokazuje rekomendacje dla grupy"""
    st.subheader("💡 Rekomendacje dla Twojej grupy")
    
    recommendations = []
    
    try:
        if 'fav_place' in same_cluster_df.columns and not same_cluster_df.empty:
            dominant_place = same_cluster_df['fav_place'].mode()[0]
            
            if dominant_place == "W górach":
                recommendations.extend([
                    "🏔️ Organizuj wspólne wyprawy górskie",
                    "🥾 Rozważ założenie klubu turystycznego"
                ])
            elif dominant_place == "Nad wodą":
                recommendations.extend([
                    "🏊‍♂️ Organizuj spotkania nad jeziorem/morzem",
                    "🚤 Rozważ wspólne aktywności wodne"
                ])
            elif dominant_place == "W lesie":
                recommendations.extend([
                    "🌲 Organizuj spacery po lesie",
                    "🍄 Rozważ grzybobranie w grupie"
                ])
        
        if 'fav_animals' in same_cluster_df.columns and not same_cluster_df.empty:
            dominant_animal = same_cluster_df['fav_animals'].mode()[0]
            
            if dominant_animal == "Psy":
                recommendations.extend([
                    "🐕 Organizuj spotkania właścicieli psów",
                    "🦴 Stwórz grupę miłośników psów"
                ])
            elif dominant_animal == "Koty":
                recommendations.extend([
                    "🐱 Dziel się zdjęciami kotów",
                    "😻 Organizuj adopcje kotów"
                ])

        if not recommendations:
            recommendations = [
                "👥 Organizuj regularne spotkania grupy",
                "💬 Stwórz grupę komunikacyjną online",
                "🎯 Znajdź wspólne hobby do rozwijania",
                "📅 Planuj wspólne wydarzenia i aktywności"
            ]
        
        for i, rec in enumerate(recommendations[:6], 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}. {rec}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Błąd podczas generowania rekomendacji: {str(e)}")
        fallback_recs = [
            "👥 Organizuj regularne spotkania grupy",
            "💬 Stwórz grupę komunikacyjną online",
            "🎯 Znajdź wspólne hobby do rozwijania"
        ]
        for i, rec in enumerate(fallback_recs, 1):
            st.write(f"{i}. {rec}")

def show_demo_mode():
    """Tryb demo gdy brak plików"""
    st.warning("🚧 Aplikacja działa w trybie demo - brak wymaganych plików danych")
    
    demo_cluster_data = {
        'name': 'Grupa Demo - Miłośnicy Przyrody',
        'description': 'Grupa osób ceniących naturę, góry i aktywny wypoczynek na świeżym powietrzu.'
    }
    
    st.markdown(f"""
    <div class="cluster-card">
        <h2>🎯 Twoja grupa: {demo_cluster_data['name']}</h2>
        <p style="font-size: 1.1em;">{demo_cluster_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo metryki
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👥 Twoja grupa</h3>
            <h2 style="color: #667eea;">42</h2>
            <p>osób w grupie</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dominujący wiek</h3>
            <h2 style="color: #667eea;">25-34</h2>
            <p>najczęstsza grupa wiekowa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎓 Wykształcenie</h3>
            <h2 style="color: #667eea;">Wyższe</h2>
            <p>dominujący poziom</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Dominująca płeć</h3>
            <h2 style="color: #667eea;">Kobieta</h2>
            <p>najczęstsza płeć</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💡 Rekomendacje dla Twojej grupy")
    demo_recommendations = [
        "🏔️ Organizuj wspólne wyprawy górskie",
        "🥾 Rozważ założenie klubu turystycznego",
        "📸 Stwórz grupę fotograficzną przyrody",
        "🌿 Organizuj warsztaty ekologiczne"
    ]
    
    for i, rec in enumerate(demo_recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-card">
            <strong>{i}. {rec}</strong>
        </div>
        """, unsafe_allow_html=True)

# Główna aplikacja
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Find Friends</h1>
        <p>Wykorzystaj AI do odkrycia osób o podobnych zainteresowaniach</p>
    </div>
    """, unsafe_allow_html=True)
    
    missing_files = check_files_exist()
    
    if missing_files:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Brakujące pliki:</h4>
            <ul>
                {''.join([f'<li>{file}</li>' for file in missing_files])}
            </ul>
            <p>Aplikacja będzie działać w trybie demo.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>👤 Twój profil</h2>
            <p>Wypełnij informacje o sobie</p>
        </div>
        """, unsafe_allow_html=True)
        
        age = st.selectbox("🎂 Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
        edu_level = st.selectbox("🎓 Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
        fav_animals = st.selectbox("🐾 Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Koty i Psy', 'Inne'])
        fav_place = st.selectbox("🏞️ Ulubione miejsce", ['W górach', 'Nad wodą', 'W lesie', 'Inne'])
        gender = st.radio("⚧ Płeć", ['Kobieta', 'Mężczyzna'])
        
        analyze_button = st.button("🔍 Znajdź moją grupę!", type="primary", use_container_width=True)
    
    if analyze_button or 'show_results' not in st.session_state:
        st.session_state.show_results = True
        st.session_state.user_data = {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    
    if st.session_state.get('show_results', False):
        if missing_files:
            show_demo_mode()
        else:
            try:
                model = get_model()
                all_df = get_all_participants()
                cluster_names_and_descriptions = get_cluster_names_and_descriptions()
                
                if model is None or all_df is None:
                    show_demo_mode()
                    return
                
                user_data = st.session_state.user_data
                person_df = pd.DataFrame([user_data])
                
                prediction_result = predict_model(model, data=person_df)
                predicted_cluster_id = prediction_result["Cluster"].values[0]
                predicted_cluster_data = cluster_names_and_descriptions.get(
                    str(predicted_cluster_id), 
                    {'name': f'Grupa {predicted_cluster_id}', 'description': 'Brak opisu'}
                )
                same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
                
                st.markdown(f"""
                <div class="cluster-card">
                    <h2>🎯 Twoja grupa: {predicted_cluster_data['name']}</h2>
                    <p style="font-size: 1.1em;">{predicted_cluster_data['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                create_advanced_visualizations(same_cluster_df, all_df, predicted_cluster_id, cluster_names_and_descriptions)
                create_detailed_charts(same_cluster_df)
                show_cluster_recommendations(predicted_cluster_data, same_cluster_df)
                
                with st.expander("📈 Zobacz wszystkie grupy"):
                    for cluster_id, cluster_info in cluster_names_and_descriptions.items():
                        try:
                            cluster_size = len(all_df[all_df["Cluster"] == int(cluster_id)])
                            st.write(f"**{cluster_info['name']}** ({cluster_size} osób)")
                            st.write(f"_{cluster_info['description']}_")
                            st.write("---")
                        except:
                            st.write(f"**{cluster_info['name']}**")
                            st.write(f"_{cluster_info['description']}_")
                            st.write("---")
                            
            except Exception as e:
                st.error(f"Wystąpił błąd podczas analizy: {str(e)}")
                st.info("Przełączanie do trybu demo...")
                show_demo_mode()

if __name__ == "__main__":
    main()