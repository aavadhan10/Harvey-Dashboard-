import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import random

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
def load_sample_data():
    """Load sample Harvey AI usage data for Rimon Law"""
    
    # User statistics data (based on the CSV structure analyzed)
    user_data = {
        'userEmail': [
            'phillip.quatrini@rimonlaw.com', 'david.hernandez@rimonlaw.com', 'sarah.chen@rimonlaw.com',
            'michael.rodriguez@rimonlaw.com', 'emily.johnson@rimonlaw.com', 'james.smith@rimonlaw.com',
            'lisa.williams@rimonlaw.com', 'robert.brown@rimonlaw.com', 'jennifer.davis@rimonlaw.com',
            'william.miller@rimonlaw.com', 'elizabeth.wilson@rimonlaw.com', 'charles.moore@rimonlaw.com',
            'mary.taylor@rimonlaw.com', 'joseph.anderson@rimonlaw.com', 'patricia.thomas@rimonlaw.com',
            'christopher.jackson@rimonlaw.com', 'susan.white@rimonlaw.com', 'daniel.harris@rimonlaw.com',
            'karen.martin@rimonlaw.com', 'matthew.thompson@rimonlaw.com', 'nancy.garcia@rimonlaw.com',
            'anthony.martinez@rimonlaw.com', 'betty.robinson@rimonlaw.com'
        ],
        'Assist (EDGAR)': [45, 23, 12, 8, 34, 19, 0, 15, 27, 0, 11, 6, 0, 22, 0, 33, 0, 18, 9, 0, 14, 0, 7],
        'Assist (Knowledge Base)': [89, 67, 45, 23, 78, 56, 34, 67, 89, 23, 45, 12, 56, 78, 34, 67, 23, 45, 56, 34, 67, 23, 12],
        'Assist (No Uploaded Files)': [156, 123, 89, 67, 134, 98, 76, 123, 156, 67, 89, 45, 98, 134, 76, 123, 67, 89, 98, 76, 123, 67, 45],
        'Assist (Tax)': [12, 8, 0, 0, 15, 0, 0, 8, 12, 0, 0, 0, 0, 15, 0, 12, 0, 0, 0, 0, 8, 0, 0],
        'Assist (User Uploaded Files)': [78, 56, 34, 23, 67, 45, 23, 56, 78, 23, 34, 12, 45, 67, 23, 56, 23, 34, 45, 23, 56, 23, 12],
        'Assist (Vault)': [34, 23, 12, 8, 28, 15, 8, 23, 34, 8, 12, 0, 15, 28, 8, 23, 8, 12, 15, 8, 23, 8, 0],
        'Assist (Internet Browsing)': [23, 15, 8, 0, 19, 12, 0, 15, 23, 0, 8, 0, 12, 19, 0, 23, 0, 8, 12, 0, 15, 0, 0],
        'Draft (Knowledge Base)': [67, 45, 23, 12, 56, 34, 12, 45, 67, 12, 23, 8, 34, 56, 12, 45, 12, 23, 34, 12, 45, 12, 8],
        'Draft (No Uploaded Files)': [45, 34, 23, 15, 38, 26, 15, 34, 45, 15, 23, 8, 26, 38, 15, 34, 15, 23, 26, 15, 34, 15, 8],
        'Draft (User Uploaded Files)': [123, 89, 67, 45, 98, 76, 45, 89, 123, 45, 67, 23, 76, 98, 45, 89, 45, 67, 76, 45, 89, 45, 23],
        'Draft (Vault)': [56, 34, 23, 12, 45, 28, 12, 34, 56, 12, 23, 8, 28, 45, 12, 34, 12, 23, 28, 12, 34, 12, 8],
        'Redline Issues List': [12, 8, 0, 0, 9, 6, 0, 8, 12, 0, 0, 0, 6, 9, 0, 12, 0, 0, 6, 0, 8, 0, 0],
        'Redline Q&A': [23, 15, 8, 0, 18, 12, 0, 15, 23, 0, 8, 0, 12, 18, 0, 23, 0, 8, 12, 0, 15, 0, 0],
        'Translation': [8, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        'Vault Review': [34, 23, 12, 8, 28, 15, 8, 23, 34, 8, 12, 0, 15, 28, 8, 23, 8, 12, 15, 8, 23, 8, 0],
        'Word Add-In': [15, 8, 0, 0, 12, 6, 0, 8, 15, 0, 0, 0, 6, 12, 0, 15, 0, 0, 6, 0, 8, 0, 0],
        'Workflows': [19, 12, 8, 0, 15, 9, 0, 12, 19, 0, 8, 0, 9, 15, 0, 19, 0, 8, 9, 0, 12, 0, 0]
    }
    
    # Calculate All types column
    df_users = pd.DataFrame(user_data)
    task_columns = [col for col in df_users.columns if col != 'userEmail']
    df_users['All types'] = df_users[task_columns].sum(axis=1)
    
    # Generate daily usage data
    start_date = date(2025, 3, 4)
    end_date = date(2025, 7, 18)
    
    daily_data = []
    current_date = start_date
    
    # Create realistic daily usage patterns
    while current_date <= end_date:
        # More activity on weekdays, less on weekends
        if current_date.weekday() < 5:  # Weekday
            daily_queries = random.randint(8, 25)
        else:  # Weekend
            daily_queries = random.randint(0, 8)
        
        # Generate individual query records for this date
        for _ in range(daily_queries):
            # Random time during business hours (8 AM - 7 PM) for weekdays
            if current_date.weekday() < 5:
                hour = random.choices(range(8, 19), weights=[1,2,3,4,5,6,7,8,7,6,4,3], k=1)[0]
            else:
                hour = random.randint(9, 17)
            
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = datetime.combine(current_date, datetime.min.time().replace(hour=hour, minute=minute, second=second))
            
            # Pick random user (weighted towards more active users)
            user = random.choices(
                df_users['userEmail'].tolist(),
                weights=df_users['All types'].tolist(),
                k=1
            )[0]
            
            # Pick random task type
            task_types = ['Assist', 'Draft', 'Redline', 'Translation', 'Review']
            task = random.choices(task_types, weights=[40, 35, 15, 5, 5], k=1)[0]
            
            # Pick random source
            sources = ['Knowledge Base', 'Files', 'Vault', 'No Files', 'Internet']
            source = random.choices(sources, weights=[30, 25, 20, 15, 10], k=1)[0]
            
            daily_data.append({
                'Time': timestamp,
                'User': user,
                'Task': task,
                'Source': source,
                'ID': random.randint(100000000, 999999999),
                'Message ID': f"{random.randint(10000000, 99999999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(100000000000, 999999999999)}",
                'Number of documents': random.randint(1, 5)
            })
        
        current_date += timedelta(days=1)
    
    df_daily = pd.DataFrame(daily_data)
    df_daily['Date'] = df_daily['Time'].dt.date
    df_daily['Hour'] = df_daily['Time'].dt.hour
    df_daily['Day_of_Week'] = df_daily['Time'].dt.day_name()
    
    return df_users, df_daily

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
    
    # Load data
    df_users, df_daily = load_sample_data()
    
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
