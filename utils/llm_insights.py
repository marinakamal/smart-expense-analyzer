"""
llm_insights.py - LLM-Powered Financial Insights
Uses Google Gemini API for personalized analysis and chatbot
"""

import google.generativeai as genai
import streamlit as st
import pandas as pd
from datetime import datetime

google_gemini_model = 'gemini-2.5-flash'

def configure_gemini(api_key=None):
    """
    Configure Google Gemini API
    
    Args:
        api_key (str): Google Gemini API key (if not using Streamlit secrets)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Try to get API key from Streamlit secrets first
        if api_key is None:
            if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
            else:
                st.error("⚠️ Gemini API key not found. Please add it to .streamlit/secrets.toml")
                return False
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        return True
        
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False


@st.cache_data(ttl=3600)

def create_chatbot_context(_df, income, savings):
    """
    Create context string for chatbot to understand user's financial data
    
    Args:
        _df (DataFrame): Categorized transaction data
        income (float): Monthly income
        savings (float): Amount saved/invested
        
    Returns:
        str: Formatted context for chatbot
    """
    total_spent = _df['amount_abs'].sum()
    
    # Category breakdown
    category_summary = _df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
    category_text = "\n".join([
        f"  - {cat}: RM {amount:,.2f}"
        for cat, amount in category_summary.items()
    ])
    
    context = f"""USER'S FINANCIAL CONTEXT:

OVERVIEW:
- Monthly Income: RM {income:,.2f}
- Total Spent: RM {total_spent:,.2f}
- Already Saved/Invested: RM {savings:,.2f}
- Spending Rate: {(total_spent/income*100):.1f}% of income
- Number of Transactions: {len(_df)}

SPENDING BY CATEGORY:
{category_text}

You are a friendly financial advisor helping this user with their budget planning and savings goals.
Answer their questions based on this data. Be specific, actionable, and encouraging.
"""
    
    return context


def chat_with_gemini(user_message, conversation_history, context):
    """
    Chat with Gemini about finances, maintaining conversation context
    
    Args:
        user_message (str): User's question
        conversation_history (list): List of previous messages
        context (str): User's financial data context
        
    Returns:
        str: Gemini's response


    LLM Integration with Context-Aware Prompting

    Approach: Prompt-based (not RAG or fine-tuning) because:
    1. Financial data is small and structured (fits in context window)
    2. User data changes with each upload (no persistent storage)
    3. Session-specific context more appropriate than document retrieval

    Prompt Strategy:
    - Inject comprehensive financial context (income, spending breakdown)
    - Maintain conversation history (last 6 messages)
    - Provide specific instructions for actionable advice
    - Cache context for 1 hour to optimize API usage
    """

    if not configure_gemini():
        return "Unable to respond: API key not configured"
    
    try:
        # Build conversation prompt
        conversation_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[-6:]  # Last 3 exchanges
        ])
        
        prompt = f"""{context}

CONVERSATION HISTORY:
{conversation_text}

USER'S CURRENT QUESTION:
{user_message}

INSTRUCTIONS:
- Answer based on their specific financial data above
- Be conversational and friendly
- Provide actionable advice with specific numbers
- If they ask "how to save X amount", break it down across categories
- If they ask about budgets, suggest realistic percentages based on their income
- Keep responses concise (150-200 words max)

YOUR RESPONSE:"""

        # Generate response
        model = genai.GenerativeModel(google_gemini_model)
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


# Quick response templates for common questions
QUICK_RESPONSES = {
    "reduce food": "To reduce food costs: 1) Cook at home 3-4x/week (save ~RM 200), 2) Pack lunch for work (save ~RM 150), 3) Limit eating out to weekends only (save ~RM 100). Total potential savings: RM 450/month!",
    
    "save more": "To boost savings: 1) Set up auto-transfer on payday (RM {amount}), 2) Apply the 24-hour rule for purchases over RM 100, 3) Review and cancel unused subscriptions. Start small and increase gradually!",
    
    "budget food": "A healthy food budget is typically 15-25% of income. For RM {income}, aim for RM {budget_low}-{budget_high}/month. This includes groceries AND dining out.",
    
    "reduce transport": "To cut transport costs: 1) Carpool 2-3x/week (save RM 80), 2) Use public transport when possible (save RM 60), 3) Plan errands to minimize trips (save RM 40). Total savings: RM 180/month!",
    
    "emergency fund": "Financial experts recommend 3-6 months of expenses as emergency fund. Based on your spending of RM {spending}/month, aim for RM {emergency_low}-{emergency_high}. Start with 1 month first!"
}


if __name__ == "__main__":
    print("LLM Insights module loaded successfully!")
    print("Available functions:")
    print("- generate_financial_insights(df, income, savings)")
    print("- chat_with_gemini(message, history, context)")
    print("- generate_savings_plan(df, income, target)")
    print("- compare_spending_benchmarks(df, income)")
    print("\nMake sure to add your GEMINI_API_KEY to .streamlit/secrets.toml")