"""
Smart Expense Analyzer - Main Application
Author: Nik Marina
CAIE Project - Batch 3
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.parser import (
    parse_maybank_csv, 
    parse_maybank_pdf,
    get_transaction_summary,
    generate_sample_data
)
from utils.categorizer import (
    categorize_dataframe, 
    get_category_breakdown, 
    CATEGORY_ICONS
)
from utils.llm_insights import (
    #generate_financial_insights, 
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
    
    st.markdown("---")
    st.markdown("### About")
    st.caption("AI-powered financial analysis using HuggingFace and Google Gemini")

# Main content area
st.header("📁 Upload Your Bank Statement")


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

# # Handle sample data
# elif use_sample:
#     with st.spinner("📄 Loading sample data..."):
#         df = generate_sample_data()
    
#     if df is not None:
#         st.success(f"✅ Loaded {len(df)} sample transactions!")

# Process data if available
if df is not None:
    
    # AI Categorization
    st.markdown("---")
    with st.spinner("🤖 AI is categorizing your expenses..."):
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
    
    # # LLM Insights Generation
    # st.markdown("---")
    # st.subheader("💡 AI Financial Insights")
    
    # with st.spinner("🧠 Generating personalized insights..."):
    #     insights = generate_financial_insights(df, income, savings)
    
    # # Display insights in a nice box
    # st.markdown(f"""
    # <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
    # {insights}
    # </div>
    # """, unsafe_allow_html=True)
    
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
    
       
    # Transaction Details
    st.markdown("---")
    st.subheader("📋 All Transactions")
    
    # Add filter options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_category = st.selectbox(
            "Filter by category:",
            ["All"] + list(df['category'].unique())
        )
    
    # Filter dataframe
    if filter_category != "All":
        filtered_df = df[df['category'] == filter_category]
    else:
        filtered_df = df
    
    # Display transactions
    st.dataframe(
        filtered_df[['date', 'description', 'category', 'amount_abs', 'confidence']].style.format({
            'amount_abs': 'RM {:.2f}',
            'confidence': '{:.2%}'
        }),
        width='stretch',
        hide_index=True
    )
    # Interactive Chatbot
    st.markdown("---")
    st.subheader("💬 Ask Your Financial Coach")
    st.markdown("Ask me anything about your spending, savings goals, or budget planning!")
    st.markdown("""
            **Example questions:**
            - "How can I save RM 500 next month?"
            - "What category should I cut spending on?"
            - "Help me create a realistic budget"
            - "Is my food spending too high?"
                """)
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    # Chat input
    if prompt := st.chat_input("💭 Ask a question (e.g., 'How can I save RM 500 next month?')"):
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
        
        **Step 3:** Get instant AI-powered insights on your spending!
        
        ### Supported Format
        Your bank statement should have these columns:
        - Date
        - Description
        - Amount
        - Balance (Optional)
        
        ### Features
        - 🤖 **AI Categorization**: Automatic expense classification using HuggingFace transformers
        - 💡 **Personalized Insights**: LLM-powered recommendations using Google Gemini
        - 📊 **Visual Analytics**: Interactive charts and spending breakdown
        - 💬 **Financial Chatbot**: Ask questions about your budget and get personalized advice
        
        ### Privacy
        - Your data is processed locally and not stored
        - Files are deleted after analysis
        - No account required
        """)
    
    # Show example use cases
    st.markdown("---")
    st.subheader("🎯 What You Can Do")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Track Spending**
        - See where your money goes
        - Identify spending patterns
        - Compare to income
        """)
    
    with col2:
        st.markdown("""
        **💰 Save More**
        - Get personalized savings tips
        - Set realistic budget targets
        - Track progress monthly
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI Advisor**
        - Ask budget questions
        - Plan savings goals
        - Get actionable advice
        """)

# Footer
st.markdown("---")
st.caption("© 2024 Smart Expense Analyzer | Batch 3")

# Cache the HuggingFace model (loads once)
#@st.cache_resource
# def load_categorizer():
#     from utils.categorizer import ExpenseCategorizer
#     return ExpenseCategorizer()

# Cache expensive computations
#@st.cache_data
# def categorize_transactions(df):
#     categorizer = load_categorizer()
#     return categorizer.categorize_dataframe(df)

# Use session state to avoid re-processing
# if 'categorized_df' not in st.session_state:
#     st.session_state.categorized_df = categorize_transactions(df)