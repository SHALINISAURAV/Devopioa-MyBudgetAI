import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import google.generativeai as genai
from io import StringIO
import json

# Set page configuration
st.set_page_config(
    page_title="MyBudgetAI - Smart Financial Assistant",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF; /* White text */
        font-weight: bold;
        text-align: center;
        background-color: #1E3A8A; /* Dark blue background */
        padding: 10px;
        border-radius: 5px;
    }

    /* Sub-header styling */
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A; /* Dark blue text */
        padding-top: 1rem;
    }

    /* Insight box styling */
    .insight-box {
        background-color: #E0F2FE; /* Light blue background */
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #93C5FD; /* Light blue tabs */
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A; /* Dark blue for active tab */
        color: white;
    }

    /* Body background color */
    body {
        background-color: #F0F4F8; /* Light gray background */
    }

    /* Button styling */
    button[kind="primary"] {
        background-color: #1E88E5 !important; /* Blue background */
        color: #FFFFFF !important; /* White text */
        border-radius: 5px !important;
        width: 100% !important; /* Make buttons equal width */
        height: 50px !important; /* Set a fixed height */
        font-size: 16px !important; /* Ensure text is clearly visible */
        font-weight: bold !important;
    }
    </style>
    <div class="main-header">MyBudgetAI – Smart Financial Assistant</div>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'budget_set' not in st.session_state:
    st.session_state.budget_set = {}
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar 
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>💰</h1>", unsafe_allow_html=True)  # Add money emoji
    st.markdown("## Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Configure Gemini API
    api_key = "AIzaSyBZ1TtGYIPtA5k6ru29-KqKEDmOe4qpCow"
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}")
    
    # Add a date range filter
    if st.session_state.df is not None:
        st.markdown("## Filter Data")
        min_date = st.session_state.df['Date'].min().date()
        max_date = st.session_state.df['Date'].max().date()
        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Category filter
        categories = ['All'] + list(st.session_state.df['category'].unique())
        selected_category = st.selectbox("Filter by category", categories)

# Load and process CSV
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Convert 'amount' column to numeric
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        # Convert date column to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        
        # Store in session state
        st.session_state.df = df
        
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Apply date filter if specified
    if 'date_range' in locals() and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    
    # Apply category filter if specified
    if 'selected_category' in locals() and selected_category != 'All':
        df = df[df['category'] == selected_category]
    
    # Filter expenses
    expenses = df[df["type"] == "EXPENSE"]
    income = df[df["type"] == "INCOME"]
    
    # Create tabs for better organization
    tabs = st.tabs(["📊 Dashboard", "💸 Expenses Analysis", "💰 Income Analysis", "🔮 Forecasting", "🤖 AI Insights", "⚙ Settings"])
    
    with tabs[0]:  # Dashboard tab
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_expenses = expenses['amount'].sum()
            st.metric("Total Expenses", f"₹{total_expenses:,.2f}", 
                     delta=f"{total_expenses/30:,.2f}/day" if len(expenses) > 0 else None)
        
        with col2:
            total_income = income['amount'].sum() if len(income) > 0 else 0
            st.metric("Total Income", f"₹{total_income:,.2f}")
        
        with col3:
            balance = total_income - total_expenses
            st.metric("Balance", f"₹{balance:,.2f}", 
                     delta=f"{balance:.2f}", delta_color="normal")
        
        st.markdown("<div class='sub-header'>Transaction Overview</div>", unsafe_allow_html=True)
        
        # Add transaction search box
        search_term = st.text_input("Search transactions", "")
        
        # Filter transactions based on search term
        filtered_df = df
        if search_term:
            filtered_df = df[df['title'].str.contains(search_term, case=False) | 
                            df['category'].str.contains(search_term, case=False)]
        
        # Show the transaction data
        st.dataframe(
            filtered_df[["Date", "title", "category", "amount", "type"]].sort_values("Date", ascending=False),
            use_container_width=True,
            height=300
        )
        
        # Spending vs Income trend over time
        st.markdown("<div class='sub-header'>Income vs Expenses Over Time</div>", unsafe_allow_html=True)
        
        # Prepare data for income vs expenses
        df_agg = df.groupby(['Date', 'type'])['amount'].sum().reset_index()
        df_pivot = df_agg.pivot(index='Date', columns='type', values='amount').reset_index()
        df_pivot.fillna(0, inplace=True)
        
        # Create a line chart using Plotly
        fig = go.Figure()
        
        if 'EXPENSE' in df_pivot.columns:
            fig.add_trace(go.Scatter(
                x=df_pivot['Date'], 
                y=df_pivot['EXPENSE'], 
                mode='lines+markers',
                name='Expenses',
                line=dict(color='#F44336', width=2)
            ))
        
        if 'INCOME' in df_pivot.columns:
            fig.add_trace(go.Scatter(
                x=df_pivot['Date'], 
                y=df_pivot['INCOME'], 
                mode='lines+markers',
                name='Income',
                line=dict(color='#4CAF50', width=2)
            ))
        
        fig.update_layout(
            title='Income vs Expenses Over Time',
            xaxis_title='Date',
            yaxis_title='Amount (₹)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='sub-header'>Spending by Category</div>", unsafe_allow_html=True)
            category_summary = expenses.groupby("category")["amount"].sum().reset_index()
            category_summary = category_summary.sort_values("amount", ascending=False)
            
            # Create an interactive bar chart with Plotly
            fig = px.bar(
                category_summary, 
                x="category", 
                y="amount",
                color="amount",
                color_continuous_scale="Viridis",
                labels={"amount": "Amount (₹)", "category": "Category"},
                title="Expense Distribution by Category"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<div class='sub-header'>Expense Breakdown</div>", unsafe_allow_html=True)
            # Calculate percentages
            category_summary['percentage'] = category_summary['amount'] / category_summary['amount'].sum() * 100
            
            # Create a pie chart
            fig = px.pie(
                category_summary, 
                values='amount', 
                names='category',
                title='Expense Breakdown by Category',
                hover_data=['percentage'],
                labels={'percentage': 'Percentage (%)'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Expenses Analysis tab
        st.markdown("<div class='sub-header'>Detailed Expense Analysis</div>", unsafe_allow_html=True)
        
        # Top expenses
        st.subheader("Top 10 Highest Expenses")
        top_expenses = expenses.sort_values('amount', ascending=False).head(10)
        
        # Create a horizontal bar chart for top expenses
        fig = px.bar(
            top_expenses,
            x="amount",
            y="title",
            orientation='h',
            color="category",
            labels={"amount": "Amount (₹)", "title": "Transaction", "category": "Category"},
            title="Top 10 Highest Expenses"
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly spending patterns
        st.subheader("Weekly Spending Patterns")
        expenses['dayofweek'] = expenses['Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_expenses = expenses.groupby('dayofweek')['amount'].sum().reindex(day_order).reset_index()
        
        fig = px.line(
            weekly_expenses,
            x="dayofweek",
            y="amount",
            markers=True,
            labels={"amount": "Amount (₹)", "dayofweek": "Day of Week"},
            title="Spending by Day of Week"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category comparison
        st.subheader("Category Comparison Month-over-Month")
        expenses['month'] = expenses['Date'].dt.strftime('%Y-%m')
        category_month = expenses.pivot_table(
            index='category', 
            columns='month', 
            values='amount', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        # Show data as a heatmap
        if len(category_month.columns) > 1:  # Only if we have month data
            category_month_long = pd.melt(
                category_month, 
                id_vars=['category'], 
                var_name='month', 
                value_name='amount'
            )
            
            fig = px.density_heatmap(
                category_month_long,
                x="month",
                y="category",
                z="amount",
                color_continuous_scale="Viridis",
                labels={"amount": "Amount (₹)", "month": "Month", "category": "Category"},
                title="Monthly Spending by Category"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Income Analysis tab
        st.markdown("<div class='sub-header'>Income Analysis</div>", unsafe_allow_html=True)
        
        if len(income) > 0:
            # Show income transactions
            st.subheader("Income Transactions")
            st.dataframe(
                income[["Date", "title", "category", "amount"]].sort_values("Date", ascending=False),
                use_container_width=True
            )
            
            # Income sources chart
            st.subheader("Income Sources")
            income_by_category = income.groupby('category')['amount'].sum().reset_index()
            
            fig = px.pie(
                income_by_category,
                values='amount',
                names='category',
                title='Income Distribution by Source',
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Income trend
            st.subheader("Income Trend")
            income['month'] = income['Date'].dt.strftime('%Y-%m')
            monthly_income = income.groupby('month')['amount'].sum().reset_index()
            
            fig = px.line(
                monthly_income,
                x='month',
                y='amount',
                markers=True,
                title='Monthly Income Trend',
                labels={'amount': 'Amount (₹)', 'month': 'Month'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income data available in the uploaded file.")
    
    with tabs[3]:  # Forecasting tab
        st.markdown("<div class='sub-header'>Expense Forecasting</div>", unsafe_allow_html=True)
        
        if len(expenses) > 0:
            # Simple forecasting
            st.subheader("Expense Forecast for Next Month")
            
            # Calculate monthly aggregates
            expenses['month'] = expenses['Date'].dt.strftime('%Y-%m')
            monthly_expenses = expenses.groupby('month')['amount'].sum().reset_index()
            
            # Calculate average monthly expenses
            avg_monthly_expense = monthly_expenses['amount'].mean()
            
            # Calculate trend
            if len(monthly_expenses) >= 2:
                last_month = monthly_expenses['amount'].iloc[-1]
                prev_month = monthly_expenses['amount'].iloc[-2]
                trend_pct = (last_month - prev_month) / prev_month * 100 if prev_month > 0 else 0
                
                # Forecast next month
                forecast_next_month = last_month * (1 + trend_pct/100)
                
                # Create forecast visualization
                forecast_df = monthly_expenses.copy()
                last_date = pd.to_datetime(monthly_expenses['month'].iloc[-1])
                next_month = (last_date + pd.DateOffset(months=1)).strftime('%Y-%m')
                # Use pandas concat instead of append (which is deprecated)
                new_row = pd.DataFrame({'month': [next_month], 'amount': [forecast_next_month]})
                forecast_df = pd.concat([forecast_df, new_row], ignore_index=True)                
                fig = px.line(
                    forecast_df,
                    x='month',
                    y='amount',
                    markers=True,
                    labels={'amount': 'Amount (₹)', 'month': 'Month'},
                    title='Monthly Expense Trend with Forecast'
                )
                
                # Add a different color for the forecast point
                fig.add_scatter(
                    x=[next_month], 
                    y=[forecast_next_month],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name='Forecast'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric(
                    "Forecasted Expense for Next Month", 
                    f"₹{forecast_next_month:,.2f}", 
                    delta=f"{trend_pct:.1f}% vs previous month"
                )
            else:
                st.info("Not enough data for forecasting. Need at least 2 months of data.")
            
            # Budget planning
            st.subheader("Budget Planning")
            
            # Category budgeting
            category_summary = expenses.groupby("category")["amount"].sum().reset_index()
            
            # Create budget inputs for each category
            st.text("Set monthly budget for each category:")
            
            # Initialize or update budget for each category
            budget_items = {}
            for _, row in category_summary.iterrows():
                category = row['category']
                if category not in st.session_state.budget_set:
                    # Initialize with current spending
                    st.session_state.budget_set[category] = row['amount']
                
                # Show slider for budget setting
                budget_items[category] = st.slider(
                    f"{category} Budget", 
                    min_value=0.0,
                    max_value=float(row['amount'] * 2),
                    value=float(st.session_state.budget_set[category]),
                    format="₹%.2f",
                    key=f"budget_{category}"
                )
                st.session_state.budget_set[category] = budget_items[category]
            
            # Create a comparison chart
            if budget_items:
                budget_df = pd.DataFrame({
                    'Category': list(budget_items.keys()),
                    'Actual': [category_summary[category_summary['category'] == cat]['amount'].values[0] for cat in budget_items.keys()],
                    'Budget': list(budget_items.values())
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=budget_df['Category'],
                    y=budget_df['Actual'],
                    name='Actual Spending',
                    marker_color='indianred'
                ))
                fig.add_trace(go.Bar(
                    x=budget_df['Category'],
                    y=budget_df['Budget'],
                    name='Budget',
                    marker_color='lightseagreen'
                ))
                
                fig.update_layout(
                    title='Budget vs. Actual Spending by Category',
                    xaxis_title='Category',
                    yaxis_title='Amount (₹)',
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate total budget
                total_budget = sum(budget_items.values())
                total_actual = sum(budget_df['Actual'])
                
                # Show budget summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Budget", f"₹{total_budget:,.2f}")
                with col2:
                    st.metric("Total Actual", f"₹{total_actual:,.2f}")
                with col3:
                    difference = total_budget - total_actual
                    st.metric(
                        "Budget Difference", 
                        f"₹{difference:,.2f}",
                        delta=f"{difference:.2f}", 
                        delta_color="normal"
                    )
        else:
            st.info("No expense data available for forecasting.")
    
    with tabs[4]:  # AI Insights tab
        st.markdown("<div class='sub-header'>Gemini AI Financial Insights</div>", unsafe_allow_html=True)
        
        if api_key:
            # Create a button to generate AI insights
            if st.button("Generate AI Insights"):
                with st.spinner("Generating financial insights with Gemini AI..."):
                    try:
                        # Prepare data summary for AI
                        category_summary = expenses.groupby("category")["amount"].sum().reset_index()
                        top_expenses = expenses.sort_values('amount', ascending=False).head(5)
                        
                        # Convert data to a more friendly format for Gemini
                        category_data = category_summary.to_dict('records')
                        top_expenses_data = top_expenses[['title', 'category', 'amount']].to_dict('records')
                        
                        # Create a summary of financial data
                        financial_data = {
                            "total_expenses": float(expenses['amount'].sum()),
                            "total_income": float(income['amount'].sum()) if len(income) > 0 else 0,
                            "expenses_by_category": category_data,
                            "top_expenses": top_expenses_data
                        }
                        
                        # Convert to JSON
                        financial_json = json.dumps(financial_data)
                        
                        # Define the prompt for Gemini
                        prompt = f"""
                        You are a financial advisor analyzing this personal finance data:
                        {financial_json}
                        
                        Provide a comprehensive financial analysis with these insights:
                        1. General overview of spending habits
                        2. Top spending categories analysis
                        3. Specific recommendations for saving money in highest spending categories
                        4. Financial health assessment
                        5. Actionable steps to improve financial position
                        
                        Format your response in markdown with clear sections and bullet points where appropriate.
                        """
                        
                        # Configure the model
                        generation_config = {
                            "temperature": 0.2,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 2048,
                        }
                        
                        # Call Gemini API
                        model = genai.GenerativeModel(
                            model_name="gemini-1.5-flash",
                            generation_config=generation_config
                        )
                        
                        response = model.generate_content(prompt)
                        
                        # Store the response
                        st.session_state.ai_insights = response.text
                    except Exception as e:
                        st.error(f"Error generating AI insights: {e}")
                        st.session_state.ai_insights = None
            
            # Display AI insights if available
            if st.session_state.ai_insights:
                st.markdown(st.session_state.ai_insights)
            else:
                st.info("Click the button above to generate AI-powered financial insights.")
                
            # NEW SECTION: AI Question Feature
            st.markdown("<div class='sub-header'>Ask Questions About Your Finances</div>", unsafe_allow_html=True)
            
            # User input for questions
            user_question = st.text_input("Ask a question about your spending habits or transactions...", 
                                          placeholder="Example: Which day of the week do I spend the most?")
            
            # Create columns for the chat interface
            chat_col1, chat_col2 = st.columns([3, 1])
            
            with chat_col1:
                if st.button("Get Answer"):
                    if user_question:
                        with st.spinner("Analyzing your data..."):
                            try:
                                # Prepare data for AI
                                # Create a compact representation of the transactions
                                transactions_sample = df.head(100).copy()
                                # Convert Timestamp objects (e.g. Date) to string
                                transactions_sample["Date"] = transactions_sample["Date"].dt.strftime("%Y-%m-%d")
                                transactions_sample = transactions_sample.to_dict("records")

                                summary_stats = {
                                    "total_expenses": float(expenses['amount'].sum()),
                                    "total_income": float(income['amount'].sum()) if len(income) > 0 else 0,
                                    "expense_categories": expenses['category'].unique().tolist(),
                                    "income_categories": income['category'].unique().tolist() if len(income) > 0 else [],
                                    "date_range": f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                                    "top_expense_categories": category_summary.head(5).to_dict('records') if len(category_summary) > 0 else []
                                }

                                data_json = json.dumps({
                                    "summary": summary_stats,
                                    "sample_transactions": transactions_sample
                                }, default=str)
                                
                                # Define the prompt for Gemini
                                prompt = f"""
                                You are a financial advisor with access to this personal financial data:
                                {data_json}
                                
                                The user is asking the following question about their finances:
                                "{user_question}"
                                
                                Please analyze the data and provide a helpful, specific answer based on the actual financial data.
                                If you need to make calculations or analyze trends to answer the question, please do so.
                                Keep your answer focused and concise, providing specific numbers and insights from the data.
                                If the question cannot be answered with the available data, explain why and what additional information would be needed.
                                """
                                
                                # Configure the model
                                generation_config = {
                                    "temperature": 0.2,
                                    "top_p": 0.8,
                                    "top_k": 40,
                                    "max_output_tokens": 1024,
                                }
                                
                                # Call Gemini API
                                model = genai.GenerativeModel(
                                    model_name="gemini-1.5-flash",
                                    generation_config=generation_config
                                )
                                
                                response = model.generate_content(prompt)
                                
                                # Add to chat history
                                st.session_state.chat_history.append({"question": user_question, "answer": response.text})
                                
                                # Display the response
                                st.markdown("### Answer:")
                                st.markdown(response.text)
                                
                            except Exception as e:
                                st.error(f"Error processing your question: {e}")
                    else:
                        st.warning("Please enter a question to get insights about your finances.")
            
            # Display chat history
            with chat_col2:
                st.markdown("### Previous Questions")
                if st.session_state.chat_history:
                    for i, chat in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 questions
                        if st.button(f"Q: {chat['question'][:20]}..." if len(chat['question']) > 20 else f"Q: {chat['question']}", key=f"history_{i}"):
                            st.markdown("### Question:")
                            st.markdown(chat['question'])
                            st.markdown("### Answer:")
                            st.markdown(chat['answer'])
                else:
                    st.info("No questions asked yet.")
            
            # Tips for asking effective questions
            with st.expander("Tips for asking effective questions"):
                st.markdown("""
                ### How to get the best answers from AI:
                
                - *Be specific:* Ask "What category did I spend the most on last month?" instead of "Where does my money go?"
                - *Compare time periods:* Try "How has my food spending changed over the last 3 months?"
                - *Ask for recommendations:* "How can I reduce my transportation expenses?"
                - *Look for patterns:* "Do I spend more on weekends than weekdays?"
                - *Budget questions:* "Am I on track to meet my savings goals?"
                """)
        else:
            st.warning("Please enter your Gemini API key in the sidebar to enable AI features.")
    
    with tabs[5]:  # Settings tab
        st.markdown("<div class='sub-header'>App Settings</div>", unsafe_allow_html=True)
        
        # Data export options
        st.subheader("Export Data")
        export_format = st.selectbox("Select export format", ["CSV", "Excel", "JSON"])
        
        if st.button("Export Data"):
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="mybudget_export.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                # Create Excel file
                buffer = StringIO()
                df.to_csv(buffer, index=False)
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="mybudget_export.csv",
                    mime="application/vnd.ms-excel"
                )
            elif export_format == "JSON":
                json_data = df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="mybudget_export.json",
                    mime="application/json"
                )
        
        # Reset app
        st.subheader("Reset Application")
        if st.button("Reset All Data"):
            st.session_state.df = None
            st.session_state.budget_set = {}
            st.session_state.ai_insights = None
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        # Clear chat history option
        st.subheader("Clear Question History")
        if st.button("Clear Question History"):
            st.session_state.chat_history = []
            st.success("Question history cleared successfully!")
        
        # About section
        st.subheader("About MyBudgetAI")
        st.markdown("""
        *MyBudgetAI* is a smart financial assistant that helps you track, analyze, and improve your personal finances.
        
        Features:
        - Transaction analysis and visualization
        - Category-wise spending breakdown
        - Budget planning and tracking
        - Expense forecasting
        - AI-powered financial insights and Q&A with Google Gemini
        
        Version: 2.1
        """)

else:
    # Display instructions when no file is uploaded
    st.markdown("<div class='sub-header'>Get Started</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to MyBudgetAI!
    
    Upload your bank transaction data in CSV format to begin analyzing your finances. The CSV should contain the following columns:
    - *Date*: Transaction date
    - *title*: Description of the transaction
    - *category*: Category of the transaction (e.g., Food, Transport)
    - *amount*: Transaction amount
    - *type*: Transaction type (EXPENSE or INCOME)
    
    Once uploaded, you'll get:
    - Interactive visualizations of your spending patterns
    - Category-wise expense breakdown
    - Budget planning tools
    - AI-powered financial insights with Google Gemini
    - Ask questions about your finances and get personalized answers
    
    ### Sample CSV Format:
    
    Date,title,category,amount,type
    2023-05-01,Grocery Store,Food,150.75,EXPENSE
    2023-05-02,Salary,Income,5000.00,INCOME
    2023-05-03,Amazon,Shopping,75.50,EXPENSE
    
    """)
    
    # Create tabs to showcase features even before data upload
    sample_tabs = st.tabs(["💡 Features", "🤖 AI Q&A Example", "📊 Sample Dashboard"])
    
    with sample_tabs[0]:
        st.markdown("""
        ### Key Features
        
        #### 📊 Comprehensive Dashboard
        - Overview of your financial status
        - Income vs expenses tracking
        - Category-wise spending breakdown
        
        #### 💸 Detailed Expense Analysis
        - Identify top spending categories
        - Track spending patterns by day of week
        - Compare month-over-month expenditure
        
        #### 💰 Income Tracking
        - Visualize income sources
        - Track income trends over time
        
        #### 🔮 Smart Forecasting
        - Predict next month's expenses
        - Set and track budgets by category
        
        #### 🤖 AI-Powered Insights
        - Get personalized financial advice
        - Ask questions about your spending habits
        - Receive tailored recommendations for improvement
        """)
    
    with sample_tabs[1]:
        st.markdown("""
        ### Example AI Q&A
        
        With our AI question feature, you can ask specific questions about your finances:
        
        *Question:* "Which category do I spend the most on?"
        
        *AI Answer:*
        
        Based on your transaction data, the category you spend the most on is Food & Dining, accounting for 32% of your total expenses (₹4,580).
        
        Your top 3 spending categories are:
        1. Food & Dining: ₹4,580 (32%)
        2. Housing: ₹3,200 (22.4%)
        3. Transportation: ₹2,150 (15.1%)
        
        Looking at your Food & Dining expenses, most transactions come from restaurants (65%) rather than groceries (35%). You might consider shifting more of your food budget to home cooking to reduce expenses in this category.
        
        
        *Question:* "Am I saving enough money each month?"
        
        *AI Answer:*
        
        Based on your financial data, you're currently saving approximately ₹2,400 per month, which is 16% of your income.
        
        Financial experts typically recommend saving 20% of your income, so you're slightly below this target.
        
        To increase your savings:
        1. Consider reducing spending in your largest category (Food & Dining)
        2. Look for subscription services you might not be fully utilizing
        3. Set a specific savings goal and track progress monthly
        
        If you could save an additional ₹600 per month, you would reach the recommended 20% savings rate.
        
        """)
        
    with sample_tabs[2]:
        # Display a placeholder visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://api.placeholder.com/400/300", caption="Sample Dashboard Overview")
        
        with col2:
            st.image("https://api.placeholder.com/400/300", caption="Sample Expense Analysis")
        
        st.markdown("""
        ### Sample Data
        
        Here's what your dashboard could look like after uploading your financial data:
        
        - *Total Expenses:* ₹14,250
        - *Total Income:* ₹20,000
        - *Balance:* ₹5,750
        
        Track your spending patterns, identify areas to save, and get personalized AI recommendations for improving your financial health - all in one application!
        """)
    
    # Add a section for uploading sample data
    st.markdown("### Try with Sample Data")
    if st.button("Load Sample Data"):
        # Create sample data
        now = datetime.now()
        dates = [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
        
        categories = ["Food", "Transport", "Shopping", "Entertainment", "Utilities", "Rent"]
        titles = {
            "Food": ["Grocery Store", "Restaurant", "Coffee Shop", "Fast Food"],
            "Transport": ["Fuel", "Uber", "Public Transit", "Car Service"],
            "Shopping": ["Amazon", "Mall", "Clothing Store", "Electronics"],
            "Entertainment": ["Movie Theater", "Concert", "Subscription", "Games"],
            "Utilities": ["Electricity Bill", "Water Bill", "Internet", "Mobile Phone"],
            "Rent": ["Monthly Rent", "Housing Maintenance"]
        }
        
        # Create sample transactions
        sample_data = []
        np.random.seed(42)  # For reproducibility
        
        # Add expenses
        for _ in range(40):
            category = np.random.choice(categories)
            title = np.random.choice(titles[category])
            date = np.random.choice(dates)
            amount = round(np.random.uniform(50, 500), 2)
            
            sample_data.append({
                "Date": date,
                "title": title,
                "category": category,
                "amount": amount,
                "type": "EXPENSE"
            })
        
        # Add income
        for _ in range(4):
            date = np.random.choice(dates)
            amount = round(np.random.uniform(2000, 5000), 2)
            
            sample_data.append({
                "Date": date,
                "title": "Salary",
                "category": "Income",
                "amount": amount,
                "type": "INCOME"
            })
        
        # Convert to DataFrame
        sample_df = pd.DataFrame(sample_data)
        sample_df["Date"] = pd.to_datetime(sample_df["Date"])
        
        # Store in session state
        st.session_state.df = sample_df
        
        st.success("Sample data loaded successfully! Explore the features above.")
        st.experimental_rerun()