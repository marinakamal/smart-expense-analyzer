"""
Smart Expense Analyzer - Complete Application with ML
Author: Nik Marina
CAIE Project - Batch 3
Features: LLM (Gemini) + ML (K-Means Clustering & Frequency Analysis)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.parser import (
    parse_maybank_csv, 
    parse_maybank_pdf
)
from utils.categorizer import (
    categorize_dataframe, 
    get_category_breakdown, 
    CATEGORY_ICONS,
    CATEGORIES,
    analyze_transaction_frequency
)
from utils.llm_insights import (
    chat_with_gemini, 
    create_chatbot_context
)

# Page configuration
st.set_page_config(
    page_title="Smart Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Smart Expense Analyzer")
st.markdown("Analyze your spending with ML clustering and AI chatbot")

# Sidebar for user inputs
with st.sidebar:
    st.header("📊 Your Information")
    income = st.number_input(
        "Monthly Income (RM)", 
        min_value=0, 
        value=4500,
        step=100
    )
    savings = st.number_input(
        "Already Saved/Invested (RM)", 
        min_value=0, 
        value=800,
        step=100
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.caption("Created by Nik Marina Kamal")
    st.page_link("https://www.linkedin.com/in/nikmarinakamal", label = "Linkedin")


# Main content area
st.header("📄 Upload Your Bank Statement")


#File upload section
uploaded_file = st.file_uploader(
    "Upload Maybank Statement",
    type=['csv', 'pdf'],
    help="Supports CSV and PDF formats"
)

# Initialize dataframe
df = None

# Handle file upload
if uploaded_file:
    with st.spinner("📄 Parsing your statement..."):
        if uploaded_file.type == 'text/csv':
            df = parse_maybank_csv(uploaded_file)
        elif uploaded_file.type == 'application/pdf':
            df = parse_maybank_pdf(uploaded_file)
    
    if df is not None:
        st.success(f"✅ Parsed {len(df)} transactions successfully!")

# Process data if available
if df is not None:
    
    # AI Categorization
    st.markdown("---")
    with st.spinner("Categorizing your expenses..."):
        df = categorize_dataframe(df, show_progress=True)
    
    st.success("✅ Categorization complete!")
    
    # Calculate metrics
    total_spent = df['amount_abs'].sum()
    spending_rate = (total_spent / income) * 100 if income > 0 else 0
    remaining = income - total_spent - savings
    savings_rate = (savings / income) * 100 if income > 0 else 0
    
    # Display summary metrics
    st.markdown("---")
    st.subheader("📊 Financial Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Spent", 
            f"RM {total_spent:,.2f}",
            delta=f"{spending_rate:.1f}% of income"
        )
    
    with col2:
        st.metric(
            "Transactions", 
            len(df),
            delta=f"Avg: RM {total_spent/len(df):.2f}"
        )
    
    with col3:
        st.metric(
            "Savings Rate", 
            f"{savings_rate:.1f}%",
            delta=f"RM {savings:,.2f}"
        )
    
    with col4:
        st.metric(
            "Remaining", 
            f"RM {remaining:,.2f}",
            delta="After expenses & savings"
        )
    
    # Category Breakdown
    st.markdown("---")
    st.subheader("📈 Spending by Category")
    
    category_summary = get_category_breakdown(df)
    
    if category_summary is not None:
        # Create two columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                category_summary,
                values='total',
                names='category',
                title='Spending Distribution',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                category_summary,
                x='category',
                y='total',
                title='Spending by Category (RM)',
                labels={'total': 'Amount (RM)', 'category': 'Category'},
                color='total',
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Category breakdown table
        st.markdown("#### Category Details")
        
        for _, row in category_summary.iterrows():
            icon = CATEGORY_ICONS.get(row['category'], '❓')
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"{icon} **{row['category']}**")
            with col2:
                st.write(f"RM {row['total']:,.2f}")
            with col3:
                st.write(f"{row['percentage']:.1f}%")
            with col4:
                st.write(f"{int(row['count'])} transactions")
    

    
    # ========== ML TRANSACTION FREQUENCY ANALYSIS (AUTO-RUN) ==========
    st.markdown("---")
    st.subheader("Transaction Frequency Prediction")
    st.markdown("Analyzing when you typically spend in each category")
    
    with st.spinner("🤖 Analyzing spending frequency patterns..."):
        frequency_results = analyze_transaction_frequency(df)
        
        if frequency_results:
            # Filter out categories with insufficient data
            valid_freq = [f for f in frequency_results if f['avg_days_between'] is not None]
            
            if valid_freq:
                freq_df = pd.DataFrame(valid_freq)
                
                # Create frequency chart
                fig_freq = px.bar(
                    freq_df,
                    x='category',
                    y='avg_days_between',
                    title='Average Days Between Purchases by Category',
                    labels={'avg_days_between': 'Days', 'category': 'Category'},
                    color='avg_days_between',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_freq, use_container_width=True)
                
                # Key insights
                most_frequent = min(valid_freq, key=lambda x: x['avg_days_between'])
                least_frequent = max(valid_freq, key=lambda x: x['avg_days_between'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 **Most Frequent:** {most_frequent['category']} - {most_frequent['frequency_pattern']}")
                with col2:
                    st.info(f"💤 **Least Frequent:** {least_frequent['category']} - {least_frequent['frequency_pattern']}")

    
    
    # ========== ENHANCED CHATBOT WITH QUICK ACTIONS ==========
    st.markdown("---")
    st.subheader("💬 Financial Chatbot")
    st.markdown("Ask questions about your spending based on your uploaded transactions!")
    
    # Quick action buttons
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 How to save RM 500?", key="quick1"):
            st.session_state.quick_question = "How can I save RM 500 next month?"
    
    with col2:
        if st.button("🍔 Is my food spending high?", key="quick2"):
            st.session_state.quick_question = "Is my food spending too high compared to my income?"
    
    with col3:
        if st.button("✂️ Where should I cut?", key="quick3"):
            st.session_state.quick_question = "What category should I cut spending on to save more?"
    
    with col4:
        if st.button("📊 Create a budget", key="quick4"):
            st.session_state.quick_question = "Help me create a realistic monthly budget"
    
    st.markdown("---")
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Handle quick question
    if 'quick_question' in st.session_state:
        prompt = st.session_state.quick_question
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate AI response
        context = create_chatbot_context(df, income, savings)
        response = chat_with_gemini(
            prompt, 
            st.session_state.messages,
            context
        )
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear quick question
        del st.session_state.quick_question
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💭 Ask a question about your finances..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                context = create_chatbot_context(df, income, savings)
                response = chat_with_gemini(
                    prompt, 
                    st.session_state.messages,
                    context
                )
                st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload your bank statement to get started!")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Getting Started
        
        **Step 1:** Upload your Maybank bank statement (CSV or PDF format)
        
        **Step 2:** Enter your monthly income and savings amount in the sidebar
        
        **Step 3:** View categorized expenses, analysis and chat with AI!
           
        ### Supported Format
        Your bank statement should have these columns:
        - Date
        - Description
        - Amount
        - Balance (Optional)
        
        ### Features
        - 🤖 **Rule-Based Categorization**: Automatic expense classification
        - ⏱️ **Frequency Prediction**: Predicts next purchase dates per category
        - 📊 **Visual Analytics**: Interactive charts and spending breakdown
        - 💬 **LLM Financial Chatbot**: Conversational advise using Google Gemini
        
        ### Technologies Used
        - **Machine Learning**: Rule-based classification, scikit-learn (Time Series Analysis)
        - **LLM**: Google Gemini API
        - **Visualization**: Plotly, Streamlit
        
        ### Privacy
        - Your data is processed locally and not stored
        - Files are deleted after analysis
        - No account required
        """)
    
    # Show example use cases
    st.markdown("---")
    st.subheader("🎯 What You Can Do")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Track Spending**
        - Category breakdown
        - Visual analytics
        - Pattern detection
        """)
    
    with col2:
        st.markdown("""
        **💬 AI Chat**
        - Ask questions about budget planning or savings goals
        - Get advice
        - Financial tips
        """)

# Footer
st.markdown("---")
st.caption("© 2026 Smart Expense Analyzer | CAIE Batch 3")
st.caption("Powered by: Google Gemini (LLM) + Streamlit")