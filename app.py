"""
Smart Expense Analyzer - Main Application
Author: Nik Marina
"""

import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Smart Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Smart Expense Analyzer")
st.markdown("Get AI-powered insights on your spending in seconds")

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

# Main content area
st.header("Upload Your Bank Statement")

# Two columns for upload options
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Maybank Statement",
        type=['csv', 'pdf'],
        help="Supports CSV and PDF formats"
    )

with col2:
    if st.button("🎲 Try Sample Data", use_container_width=True):
        st.info("Sample data feature coming soon!")

# Process uploaded file
if uploaded_file:
    st.success("✅ File uploaded successfully!")
    
    # For now, just show file details
    st.write("**File Details:**")
    st.write(f"- Filename: {uploaded_file.name}")
    st.write(f"- File type: {uploaded_file.type}")
    st.write(f"- File size: {uploaded_file.size} bytes")
    
    # Placeholder for future functionality
    with st.spinner("Analyzing transactions..."):
        st.info("🚧 Analysis feature coming next!")
        
    # Show what will come next
    st.markdown("---")
    st.subheader("Coming Soon:")
    st.markdown("""
    - 📊 **Spending Dashboard** - Visual breakdown by category
    - 🤖 **AI Categorization** - Automatic expense classification
    - 💡 **Personalized Insights** - LLM-powered recommendations
    - 💬 **Financial Chatbot** - Ask questions about your spending
    """)

else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload your bank statement to get started, or try sample data!")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Step 1:** Upload your Maybank bank statement (CSV or PDF)
        
        **Step 2:** Enter your monthly income and savings amount
        
        **Step 3:** Get instant AI-powered insights on your spending!
        
        **Supported Format:**
        - Entry Date | Value Date | Description | Amount | Balance
        """)

# Footer
st.markdown("---")
st.caption("Made with ❤️ for CAIE Project | Powered by Streamlit")

import streamlit as st
import pandas as pd
from utils.parser import (
    parse_maybank_csv, 
    parse_maybank_pdf,
    filter_expenses_only,
    get_transaction_summary,
    generate_sample_data
)

# ... your existing code ...

# When user uploads file:
if uploaded_file:
    # Parse based on file type
    if uploaded_file.type == 'text/csv':
        df = parse_maybank_csv(uploaded_file)
    elif uploaded_file.type == 'application/pdf':
        df = parse_maybank_pdf(uploaded_file)
    
    if df is not None:
        st.success(f"✅ Parsed {len(df)} transactions successfully!")
        
        # Get summary
        summary = get_transaction_summary(df)
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spent", f"RM {summary['total_spent']:,.2f}")
        col2.metric("Transactions", summary['total_expenses'])
        col3.metric("Avg per Transaction", f"RM {summary['average_transaction']:,.2f}")
        
        # Show data
        st.dataframe(df)

# Sample data button
if st.button("🎲 Try Sample Data"):
    df = generate_sample_data()
    st.success("✅ Loaded sample data!")
    # ... display data ...

import streamlit as st
from utils.parser import parse_maybank_csv, parse_maybank_pdf, generate_sample_data
from utils.categorizer import categorize_dataframe, get_category_breakdown, CATEGORY_ICONS

st.title("💰 Smart Expense Analyzer")

# ... your existing upload code ...

if uploaded_file:
    # Parse the file
    if uploaded_file.type == 'text/csv':
        df = parse_maybank_csv(uploaded_file)
    else:
        df = parse_maybank_pdf(uploaded_file)
    
    if df is not None:
        st.success(f"✅ Parsed {len(df)} transactions")
        
        # NEW: AI Categorization
        with st.spinner("🤖 AI is categorizing your expenses..."):
            df = categorize_dataframe(df)
        
        st.success("✅ Categorization complete!")
        
        # Show category breakdown
        category_summary = get_category_breakdown(df)
        
        st.subheader("📊 Spending by Category")
        
        # Display with icons
        for _, row in category_summary.iterrows():
            icon = CATEGORY_ICONS.get(row['category'], '❓')
            st.write(f"{icon} **{row['category']}**: RM {row['total']:,.2f} ({row['percentage']}%)")
        
        # Show categorized transactions
        st.subheader("📋 Categorized Transactions")
        st.dataframe(df[['date', 'description', 'category', 'amount_abs', 'confidence']])

import streamlit as st
from utils.parser import parse_maybank_csv, generate_sample_data
from utils.categorizer import categorize_dataframe, get_category_breakdown
from utils.llm_insights import generate_financial_insights, chat_with_gemini, create_chatbot_context

st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")
st.title("💰 Smart Expense Analyzer")

# Sidebar
with st.sidebar:
    income = st.number_input("Monthly Income (RM)", value=4500, step=100)
    savings = st.number_input("Already Saved (RM)", value=800, step=100)

# Upload
uploaded_file = st.file_uploader("Upload Statement", type=['csv'])
use_sample = st.button("🎲 Try Sample Data")

df = None
if uploaded_file:
    df = parse_maybank_csv(uploaded_file)
elif use_sample:
    df = generate_sample_data()

if df is not None:
    # Categorize
    with st.spinner("🤖 AI Categorization..."):
        df = categorize_dataframe(df)
    
    st.success("✅ Analysis complete!")
    
    # Metrics
    total_spent = df['amount_abs'].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"RM {total_spent:,.2f}")
    col2.metric("Income Spent", f"{(total_spent/income*100):.1f}%")
    col3.metric("Savings Rate", f"{((income-total_spent-savings)/income*100):.1f}%")
    
    # ===== NEW: LLM INSIGHTS =====
    st.markdown("---")
    st.subheader("💡 AI Financial Insights")
    
    with st.spinner("🧠 Generating personalized insights..."):
        insights = generate_financial_insights(df, income, savings)
    
    st.markdown(insights)
    
    # ===== NEW: CHATBOT =====
    st.markdown("---")
    st.subheader("💬 Ask Your Financial Coach")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your finances... (e.g., 'How can I save RM 500 next month?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = create_chatbot_context(df, income, savings)
                response = chat_with_gemini(
                    prompt, 
                    st.session_state.messages,
                    context
                )
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Category breakdown (existing code)
    st.markdown("---")
    st.subheader("📊 Spending Breakdown")
    category_summary = get_category_breakdown(df)
    st.dataframe(category_summary)