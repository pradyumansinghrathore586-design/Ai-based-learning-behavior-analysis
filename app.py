import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor
from model import BehaviorClusteringModel
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Student Behavior Analysis",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Student Learning Behaviour Pattern Analysis")
st.markdown("""
This dashboard uses **Unsupervised Machine Learning (MiniBatch K-Means)** to analyze student engagement 
and academic data. It automatically groups students into meaningful clusters based on their behavioral patterns.
""")

# Setup Sidebar
st.sidebar.header("⚙️ Control Panel")

# Data loading and clustering pipeline
@st.cache_data
def load_and_process_data():
    data_path = 'student_data.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found. Please ensure the dataset exists.")
            
    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    df_scaled = preprocessor.preprocess(df)
    
    # Features for clustering
    features = [
        'Study_Hours_Per_Week', 'Attendance_Rate', 'Assignments_Completed', 
        'Exam_Score', 'Forum_Participation', 'Library_Usage', 'Past_Failures'
    ]
    
    # Train Model
    model = BehaviorClusteringModel(n_clusters=3)
    raw_labels = model.fit_predict(df_scaled[features])
    
    # Interpret Clusters (Top Performer, Average, At Risk)
    category_labels = model.interpret_clusters(df, raw_labels)
    df['Category'] = category_labels
    
    return df, df_scaled[features], category_labels, model

try:
    with st.spinner("Loading models and crunching large student datasets..."):
        df_final, df_scaled, categories, clustering_model = load_and_process_data()
        
    st.sidebar.success("✅ Data & Model Loaded Successfully!")
    
    # Filter by category
    selected_category = st.sidebar.multiselect(
        "Filter by Student Category:",
        options=["Top Performer", "Average", "At Risk"],
        default=["Top Performer", "Average", "At Risk"]
    )
    
    # Apply filter
    df_filtered = df_final[df_final['Category'].isin(selected_category)]
    
    # Summary Statistics UI
    st.header("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students Analyzed", f"{len(df_filtered):,}")
    
    if not df_filtered.empty:
        col2.metric("Avg Study Hours/Week", f"{df_filtered['Study_Hours_Per_Week'].mean():.1f}")
        col3.metric("Avg Attendance Rate", f"{df_filtered['Attendance_Rate'].mean():.1f}%")
        col4.metric("Avg Exam Score", f"{df_filtered['Exam_Score'].mean():.1f}")
    else:
        col2.metric("Avg Study Hours/Week", "0.0")
        col3.metric("Avg Attendance Rate", "0.0%")
        col4.metric("Avg Exam Score", "0.0")
    
    # Visualization UI
    st.header("📈 Behavioral Clusters (PCA Visualization)")
    st.markdown("""
    Since our data has 7 dimensions, we use **Principal Component Analysis (PCA)** to reduce it to 2 dimensions for visualization.
    Each dot represents a student, colored by their algorithmic cluster mapping.
    """)
    
    visualizer = Visualizer()
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("2D Cluster Map")
        fig_2d = visualizer.plot_clusters_2d(df_scaled.values, categories, sample_size=1200)
        st.pyplot(fig_2d)
        
    with col_plot2:
        st.subheader("Relative Feature Profiles")
        fig_features = visualizer.plot_feature_importance(df_final, categories)
        st.pyplot(fig_features)
        
    # Recommendations Table
    st.header("🎯 Student Action Plan & Recommendations")
    
    # Show generalized recommendations
    st.subheader("Automated AI Interventions")
    for cat in ["At Risk", "Average", "Top Performer"]:
        if cat in selected_category:
            st.info(f"**{cat}**: {clustering_model.get_recommendation(cat)}")
            
    # Sample display
    st.subheader("Filtered Student Roster (Sample)")
    st.dataframe(
        df_filtered[['Student_ID', 'Category', 'Exam_Score', 'Attendance_Rate', 'Study_Hours_Per_Week', 'Past_Failures']].head(50),
        use_container_width=True
    )
            
except Exception as e:
    st.error(f"An error occurred: {e}")
