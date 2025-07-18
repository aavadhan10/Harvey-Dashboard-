import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import openpyxl

# Page configuration
st.set_page_config(
    page_title="Harvey AI Usage Dashboard - Rimon Law",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e8eaed;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_user_stats():
    """Load user statistics CSV data"""
    df = pd.read_csv('user_stats_03_04_202507_18_2025.csv')
    return df

@st.cache_data
def load_daily_usage():
    """Load daily usage Excel data"""
    df = pd.read_excel('harveyusagestart_20250304_end_20250718.xlsx')
    df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Time'].dt.date
    df['Hour'] = df['Time'].dt.hour
    df['Day_of_Week'] = df['Time'].dt.day_name()
    return df

def calculate_usage_by_type(df_users):
    """Calculate usage statistics by task type"""
    type_columns = [col for col in df_users.columns if col not in ['userEmail', 'All types']]
    
    usage_by_type = {}
    users_by_type = {}
    
    for col in type_columns:
        total_queries = df_users[col].sum()
        active_users = (df_users[col] > 0).sum()
        
        usage_by_type[col] = total_queries
        users_by_type[col] = active_users
    
    return usage_by_type, users_by_type

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Harvey AI Usage Dashboard - Rimon Law</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="section-header">📁 Upload Data Files</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_file = st.file_uploader(
            "Upload User Statistics CSV",
            type=['csv'],
            help="Upload the user_stats_03_04_202507_18_2025.csv file"
        )
    
    with col2:
        excel_file = st.file_uploader(
            "Upload Daily Usage Excel",
            type=['xlsx', 'xls'],
            help="Upload the harveyusagestart_20250304_end_20250718.xlsx file"
        )
    
    # Check if both files are uploaded
    if csv_file is None or excel_file is None:
        st.info("👆 Please upload both data files to proceed with the dashboard.")
        st.markdown("""
        **Required files:**
        - User Statistics CSV: `user_stats_03_04_202507_18_2025.csv`
        - Daily Usage Excel: `harveyusagestart_20250304_end_20250718.xlsx`
        """)
        return
    
    # Load data
    try:
        df_users = load_user_stats(csv_file)
        df_daily = load_daily_usage(excel_file)
        
        st.success("✅ Data files loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading data files: {e}")
        st.info("Please check that the files are in the correct format and try again.")
        return
    
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Filters")
    
    # Date range filter
    min_date = df_daily['Date'].min()
    max_date = df_daily['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_filter"
    )
    
    # User filter
    all_users = sorted(df_daily['User'].unique())
    selected_users = st.sidebar.multiselect(
        "Select Users",
        options=all_users,
        default=all_users,
        key="user_filter"
    )
    
    # Task type filter
    task_types = sorted(df_daily['Task'].unique())
    selected_tasks = st.sidebar.multiselect(
        "Select Task Types",
        options=task_types,
        default=task_types,
        key="task_filter"
    )
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_daily_filtered = df_daily[
            (df_daily['Date'] >= start_date) & 
            (df_daily['Date'] <= end_date) &
            (df_daily['User'].isin(selected_users)) &
            (df_daily['Task'].isin(selected_tasks))
        ]
    else:
        df_daily_filtered = df_daily[
            (df_daily['User'].isin(selected_users)) &
            (df_daily['Task'].isin(selected_tasks))
        ]
    
    # Filter user stats for selected users
    df_users_filtered = df_users[df_users['userEmail'].isin(selected_users)]
    
    # Key Metrics Section
    st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_users = len(df_users_filtered)
    total_queries = df_users_filtered['All types'].sum()
    active_users = (df_users_filtered['All types'] > 0).sum()
    avg_queries_per_user = total_queries / max(active_users, 1)
    
    with col1:
        st.metric("Total Users", f"{total_users}", help="Total number of users in the system")
    
    with col2:
        st.metric("Active Users", f"{active_users}", help="Users with at least one query")
    
    with col3:
        st.metric("Total Queries", f"{total_queries:,}", help="Total number of queries across all types")
    
    with col4:
        st.metric("Avg Queries/User", f"{avg_queries_per_user:.1f}", help="Average queries per active user")
    
    # Usage by Type Section
    st.markdown('<div class="section-header">🔧 Usage by Task Type</div>', unsafe_allow_html=True)
    
    usage_by_type, users_by_type = calculate_usage_by_type(df_users_filtered)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Queries by type
        type_data = pd.DataFrame(list(usage_by_type.items()), columns=['Task Type', 'Queries'])
        type_data = type_data[type_data['Queries'] > 0].sort_values('Queries', ascending=True)
        
        if not type_data.empty:
            fig_queries = px.bar(
                type_data, 
                x='Queries', 
                y='Task Type',
                title='Number of Queries by Task Type',
                orientation='h',
                color='Queries',
                color_continuous_scale='Blues'
            )
            fig_queries.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_queries, use_container_width=True)
    
    with col2:
        # Users by type
        user_data = pd.DataFrame(list(users_by_type.items()), columns=['Task Type', 'Active Users'])
        user_data = user_data[user_data['Active Users'] > 0].sort_values('Active Users', ascending=True)
        
        if not user_data.empty:
            fig_users = px.bar(
                user_data, 
                x='Active Users', 
                y='Task Type',
                title='Number of Active Users by Task Type',
                orientation='h',
                color='Active Users',
                color_continuous_scale='Greens'
            )
            fig_users.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_users, use_container_width=True)
    
    # Daily Activity Section
    st.markdown('<div class="section-header">📅 Daily Activity Analysis</div>', unsafe_allow_html=True)
    
    if not df_daily_filtered.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily queries over time
            daily_counts = df_daily_filtered.groupby('Date').size().reset_index(name='Queries')
            daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
            
            fig_timeline = px.line(
                daily_counts, 
                x='Date', 
                y='Queries',
                title='Daily Query Volume Over Time',
                markers=True
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Daily unique users
            daily_users = df_daily_filtered.groupby('Date')['User'].nunique().reset_index(name='Unique Users')
            daily_users['Date'] = pd.to_datetime(daily_users['Date'])
            
            fig_users_daily = px.line(
                daily_users, 
                x='Date', 
                y='Unique Users',
                title='Daily Unique Users Over Time',
                markers=True,
                color_discrete_sequence=['#ff7f0e']
            )
            fig_users_daily.update_layout(height=400)
            st.plotly_chart(fig_users_daily, use_container_width=True)
    
    # User Activity Analysis
    st.markdown('<div class="section-header">👥 User Activity Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top users by queries
        active_users_df = df_users_filtered[df_users_filtered['All types'] > 0].copy()
        active_users_df['User'] = active_users_df['userEmail'].str.split('@').str[0]
        top_users = active_users_df.nlargest(15, 'All types')
        
        if not top_users.empty:
            fig_top_users = px.bar(
                top_users, 
                x='All types', 
                y='User',
                title='Top 15 Users by Total Queries',
                orientation='h',
                color='All types',
                color_continuous_scale='Viridis'
            )
            fig_top_users.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_top_users, use_container_width=True)
    
    with col2:
        # Query distribution histogram
        query_counts = df_users_filtered[df_users_filtered['All types'] > 0]['All types']
        
        if not query_counts.empty:
            fig_hist = px.histogram(
                x=query_counts,
                title='Distribution of Queries per User',
                labels={'x': 'Number of Queries', 'y': 'Number of Users'},
                nbins=20
            )
            fig_hist.update_layout(height=500)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Temporal Patterns
    st.markdown('<div class="section-header">⏰ Usage Patterns</div>', unsafe_allow_html=True)
    
    if not df_daily_filtered.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            hourly_data = df_daily_filtered.groupby('Hour').size().reset_index(name='Queries')
            
            fig_hourly = px.bar(
                hourly_data, 
                x='Hour', 
                y='Queries',
                title='Query Distribution by Hour of Day',
                color='Queries',
                color_continuous_scale='Plasma'
            )
            fig_hourly.update_layout(height=400, showlegend=False)
            fig_hourly.update_xaxis(dtick=1)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Day of week distribution
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_dow = df_daily_filtered.groupby('Day_of_Week').size().reset_index(name='Queries')
            daily_dow['Day_of_Week'] = pd.Categorical(daily_dow['Day_of_Week'], categories=day_order, ordered=True)
            daily_dow = daily_dow.sort_values('Day_of_Week')
            
            fig_dow = px.bar(
                daily_dow, 
                x='Day_of_Week', 
                y='Queries',
                title='Query Distribution by Day of Week',
                color='Queries',
                color_continuous_scale='Cividis'
            )
            fig_dow.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dow, use_container_width=True)
    
    # Task and Source Analysis
    st.markdown('<div class="section-header">🎯 Task & Source Analysis</div>', unsafe_allow_html=True)
    
    if not df_daily_filtered.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Task distribution
            task_dist = df_daily_filtered.groupby('Task').size().reset_index(name='Count')
            
            fig_tasks = px.pie(
                task_dist, 
                values='Count', 
                names='Task',
                title='Query Distribution by Task Type'
            )
            fig_tasks.update_traces(textposition='inside', textinfo='percent+label')
            fig_tasks.update_layout(height=400)
            st.plotly_chart(fig_tasks, use_container_width=True)
        
        with col2:
            # Source distribution
            source_dist = df_daily_filtered.groupby('Source').size().reset_index(name='Count')
            
            fig_sources = px.pie(
                source_dist, 
                values='Count', 
                names='Source',
                title='Query Distribution by Source Type'
            )
            fig_sources.update_traces(textposition='inside', textinfo='percent+label')
            fig_sources.update_layout(height=400)
            st.plotly_chart(fig_sources, use_container_width=True)
    
    # Detailed Data Tables
    st.markdown('<div class="section-header">📋 Detailed Data</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["User Summary", "Recent Activity", "Task Breakdown"])
    
    with tab1:
        # User summary table
        user_summary = df_users_filtered[df_users_filtered['All types'] > 0].copy()
        user_summary['User'] = user_summary['userEmail'].str.split('@').str[0]
        user_summary = user_summary.sort_values('All types', ascending=False)
        
        display_cols = ['User', 'All types'] + [col for col in user_summary.columns 
                                               if col not in ['userEmail', 'User', 'All types'] 
                                               and user_summary[col].sum() > 0]
        
        st.dataframe(
            user_summary[display_cols],
            use_container_width=True,
            height=400
        )
    
    with tab2:
        # Recent activity
        if not df_daily_filtered.empty:
            recent_activity = df_daily_filtered.sort_values('Time', ascending=False).head(100)
            recent_activity['User_Short'] = recent_activity['User'].str.split('@').str[0]
            
            display_recent = recent_activity[['Time', 'User_Short', 'Task', 'Source']].rename(
                columns={'User_Short': 'User'}
            )
            
            st.dataframe(
                display_recent,
                use_container_width=True,
                height=400
            )
    
    with tab3:
        # Task breakdown by user
        if not df_daily_filtered.empty:
            task_breakdown = df_daily_filtered.groupby(['User', 'Task']).size().reset_index(name='Count')
            task_breakdown['User_Short'] = task_breakdown['User'].str.split('@').str[0]
            task_pivot = task_breakdown.pivot(index='User_Short', columns='Task', values='Count').fillna(0)
            
            st.dataframe(
                task_pivot,
                use_container_width=True,
                height=400
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Dashboard Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"**Data Period:** {min_date} to {max_date} | "
        f"**Total Records:** {len(df_daily):,}"
    )

if __name__ == "__main__":
    main()
