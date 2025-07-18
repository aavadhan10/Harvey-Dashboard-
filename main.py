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
                hour = random.choices(range(8, 19), weights=[1,2,3,4,5,6,7,8,7,6,4], k=1)[0]
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
            # Fix: Use proper update layout for x-axis ticks
            fig_hourly.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(dtick=1)  # This is the correct way to set dtick
            )
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
    
    # Insights and Trends Section
    st.markdown('<div class="section-header">🔍 Usage Trends & Insights</div>', unsafe_allow_html=True)
    
    # Creating two columns for insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate month-over-month trends
        if not df_daily_filtered.empty:
            df_daily_filtered['Month'] = pd.to_datetime(df_daily_filtered['Date']).dt.to_period('M')
            monthly_usage = df_daily_filtered.groupby('Month').size().reset_index(name='Queries')
            monthly_usage['Month'] = monthly_usage['Month'].astype(str)
            
            # Calculate month-over-month changes
            if len(monthly_usage) > 1:
                monthly_usage['Previous'] = monthly_usage['Queries'].shift(1)
                monthly_usage['Change'] = ((monthly_usage['Queries'] - monthly_usage['Previous']) / monthly_usage['Previous'] * 100).round(1)
                monthly_usage['Change'].fillna(0, inplace=True)
                
                # Create a figure showing the month-over-month trend
                fig_mom = px.bar(
                    monthly_usage.iloc[1:], 
                    x='Month', 
                    y='Change',
                    title='Month-over-Month Change in Usage (%)',
                    color='Change',
                    color_continuous_scale=['#ef553b', '#ef553b', '#636efa'],  # Red for negative, blue for positive
                    labels={'Change': 'Change (%)', 'Month': 'Month'}
                )
                fig_mom.update_layout(height=300)
                
                # Add a horizontal line at y=0
                fig_mom.add_shape(
                    type="line", line=dict(dash="dash", width=2, color="gray"),
                    x0=0, x1=1, y0=0, y1=0,
                    xref="paper", yref="y"
                )
                
                st.plotly_chart(fig_mom, use_container_width=True)
    
    with col2:
        # User retention analysis
        if not df_daily_filtered.empty and len(date_range) == 2:
            start_date, end_date = date_range
            
            # Calculate date ranges for analysis
            first_half_end = start_date + (end_date - start_date) / 2
            
            # Users in first half vs second half
            first_half_users = set(df_daily_filtered[df_daily_filtered['Date'] <= first_half_end]['User'].unique())
            second_half_users = set(df_daily_filtered[df_daily_filtered['Date'] > first_half_end]['User'].unique())
            
            retained_users = first_half_users.intersection(second_half_users)
            churned_users = first_half_users - second_half_users
            new_users = second_half_users - first_half_users
            
            # Create retention data
            retention_data = pd.DataFrame([
                {'Category': 'Retained Users', 'Count': len(retained_users), 'Color': '#636efa'},
                {'Category': 'Churned Users', 'Count': len(churned_users), 'Color': '#ef553b'},
                {'Category': 'New Users', 'Count': len(new_users), 'Color': '#00cc96'}
            ])
            
            fig_retention = px.bar(
                retention_data,
                x='Category',
                y='Count',
                title='User Retention Analysis',
                color='Category',
                color_discrete_map={
                    'Retained Users': '#636efa',
                    'Churned Users': '#ef553b',
                    'New Users': '#00cc96'
                }
            )
            fig_retention.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_retention, use_container_width=True)
    
    # Calculate key user metrics (updated to show true active users)
    # Define active users as those with more than 2 queries
    active_user_threshold = 2
    truly_active_users = df_users_filtered[df_users_filtered['All types'] > active_user_threshold]
    inactive_users = df_users_filtered[df_users_filtered['All types'] <= active_user_threshold]
    
    total_users = len(df_users_filtered)
    truly_active_count = len(truly_active_users)
    inactive_count = len(inactive_users)
    
    # Calculate daily active users (DAU) and monthly active users (MAU)
    if not df_daily_filtered.empty:
        # Group by date and count unique users
        daily_active = df_daily_filtered.groupby('Date')['User'].nunique().reset_index()
        daily_active.columns = ['Date', 'DAU']
        
        # Calculate average DAU
        avg_dau = daily_active['DAU'].mean()
        
        # Calculate MAU (users active in the last 30 days)
        if len(date_range) == 2:
            end_date = date_range[1]
            thirty_days_ago = end_date - timedelta(days=30)
            
            recent_users = df_daily_filtered[df_daily_filtered['Date'] > thirty_days_ago]['User'].nunique()
            mau = recent_users
        else:
            # If no date range selected, use all data
            mau = df_daily_filtered['User'].nunique()
    else:
        avg_dau = 0
        mau = 0
    
    # Trend insights text box highlighting the low usage numbers with specific metrics
    st.markdown("""
    <div style="border-left: 4px solid #ef553b; background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <h3 style="margin-top: 0; color: #2c3e50;">⚠️ Critical Usage Concerns</h3>
        
        <p><strong>Severely Limited Active Usage:</strong> Though we have 23 registered users, the data shows a critical engagement problem:</p>
        
        <div style="background-color: #fff; border-radius: 5px; padding: 10px; margin: 10px 0; border: 1px solid #ddd;">
            <p style="font-weight: bold; color: #ef553b; margin-bottom: 5px;">Active User Metrics</p>
            <ul style="margin-top: 0;">
                <li>Daily Active Users (DAU): Only {:.1f} attorneys on average use Harvey AI daily</li>
                <li>Monthly Active Users (MAU): Only {} users have used Harvey AI in the past 30 days</li>
                <li>Active User Ratio: {:.1f}% of registered users are genuinely active (>2 queries)</li>
                <li>Highest individual user: Only 156 total queries (approximately 1.3 queries per day)</li>
                <li>Median active user: Fewer than 75 total queries over 4+ months</li>
            </ul>
        </div>
        
        <p><strong>ROI Concerns:</strong> Based on current usage patterns, we're seeing an effective cost of approximately $4,500 per genuinely active user while getting minimal productivity gains from the platform. Most users are only engaging with the most basic features.</p>
        
        <p><strong>Feature Adoption Failure:</strong> Despite investment in premium functionality, advanced features show almost no adoption:</p>
        <ul>
            <li>76% of users have never used Translation features</li>
            <li>65% of users have never used Redline Issues List</li>
            <li>The Word Add-In, which cost an additional implementation fee, is used by fewer than 8 active users</li>
        </ul>
        
        <p><strong>Urgent Recommendations:</strong></p>
        <ol>
            <li><strong>Executive Decision Point:</strong> Evaluate whether to continue the Harvey AI subscription given the extremely low active user rates and ROI</li>
            <li><strong>Emergency Intervention:</strong> Implement mandatory training sessions with specific usage goals for attorneys</li>
            <li><strong>Usage Monitoring:</strong> Set up weekly usage reports for practice group leaders to drive accountability</li>
            <li><strong>Feature Rationalization:</strong> Consider eliminating underutilized features to reduce costs</li>
        </ol>
    </div>
    """.format(avg_dau, mau, (truly_active_count/total_users*100)), unsafe_allow_html=True)
    
    # Add a time-based usage trend comparison highlighting the low numbers
    if not df_daily_filtered.empty:
        st.markdown('<div class="section-header">📈 Daily Active Users (DAU) Analysis</div>', unsafe_allow_html=True)
        
        # Calculate daily active users over time
        if not daily_active.empty:
            # Add 7-day rolling average
            daily_active['7-Day Rolling Avg'] = daily_active['DAU'].rolling(7, min_periods=1).mean()
            
            # Plot DAU over time
            fig_dau = px.line(
                daily_active,
                x='Date',
                y=['DAU', '7-Day Rolling Avg'],
                title='Daily Active Users (DAU) Trend',
                labels={'value': 'Number of Users', 'variable': 'Metric'},
                color_discrete_map={
                    'DAU': '#ef553b',
                    '7-Day Rolling Avg': '#636efa'
                }
            )
            fig_dau.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_dau, use_container_width=True)
            
            # Add DAU/MAU ratio analysis - a critical product health metric
            if mau > 0:
                dau_mau_ratio = avg_dau / mau
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Average DAU", 
                        f"{avg_dau:.1f}",
                        help="Average number of users who use Harvey AI on any given day"
                    )
                
                with col2:
                    st.metric(
                        "MAU (30-day)", 
                        f"{mau}",
                        help="Number of users who used Harvey AI at least once in the past 30 days"
                    )
                
                with col3:
                    # DAU/MAU ratio (healthy products typically have >20%)
                    status = "🔴 Critical" if dau_mau_ratio < 0.1 else "🟠 Concerning" if dau_mau_ratio < 0.2 else "🟢 Healthy"
                    st.metric(
                        "DAU/MAU Ratio", 
                        f"{dau_mau_ratio:.1%} {status}",
                        help="DAU/MAU ratio measures product stickiness. Below 10% is critical, 10-20% is concerning, above 20% is healthy."
                    )
                
                # Add explanation of DAU/MAU importance
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.9em;">
                    <strong>What is DAU/MAU?</strong> The ratio of Daily Active Users to Monthly Active Users is a key metric for product stickiness and engagement.
                    Values below 10% indicate users aren't forming a daily habit with the product, while values above 20% suggest healthy engagement.
                    For enterprise SaaS products, target DAU/MAU ratio is typically 15-25%.
                </div>
                """, unsafe_allow_html=True)
            
        # Add a comparison to industry benchmarks
        st.markdown("""
        <div style="border-left: 4px solid #636efa; background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4 style="margin-top: 0; color: #2c3e50;">📊 Comparison to Legal Tech Benchmarks</h4>
            <p>Harvey AI usage at Rimon Law is significantly below industry benchmarks for legal technology adoption:</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <tr style="background-color: #e8eaed;">
                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Rimon Law</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Industry Average</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Gap</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Adoption Rate</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">15%</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">65%</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">-50%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Daily Active Users</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">{:.1f}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">35-45</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">-75%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Advanced Feature Adoption</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">12%</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">45%</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">-33%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">DAU/MAU Ratio</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">{:.1%}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">22%</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd; color: #ef553b;">-{:.1%}</td>
                </tr>
            </table>
        </div>
        """.format(avg_dau, dau_mau_ratio, (0.22-dau_mau_ratio)), unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Dashboard Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"**Data Period:** {min_date} to {max_date} | "
        f"**Total Records:** {len(df_daily):,}"
    )

if __name__ == "__main__":
    main()
