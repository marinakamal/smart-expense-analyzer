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
def generate_financial_insights(_df, income, savings):
    """
    Generate personalized financial insights using Google Gemini
    
    Args:
        _df (DataFrame): Categorized transaction data (underscore for Streamlit caching)
        income (float): Monthly income
        savings (float): Amount already saved/invested
    Returns:
        str: Generated insights text
    """
    if not configure_gemini():
        return "Unable to generate insights: API key not configured"
    
    try:
        # Prepare data summary
        total_spent = _df['amount_abs'].sum()
        spending_rate = (total_spent / income) * 100 if income > 0 else 0
        
        # Get category breakdown
        category_summary = _df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
        top_categories = category_summary.head(5)
        
        # Format category breakdown for prompt
        category_text = "\n".join([
            f"  - {cat}: RM {amount:,.2f} ({amount/total_spent*100:.1f}%)"
            for cat, amount in top_categories.items()
        ])
        
        # Get top 5 individual expenses
        top_expenses = _df.nlargest(5, 'amount_abs')[['description', 'amount_abs', 'category']]
        expenses_text = "\n".join([
            f"  - RM {row['amount_abs']:,.2f}: {row['description']} ({row['category']})"
            for _, row in top_expenses.iterrows()
        ])
        
        # Build prompt
        prompt = f"""You are a friendly financial coach analyzing a user's monthly spending.

USER'S FINANCIAL DATA:
- Monthly Income: RM {income:,.2f}
- Total Spent This Month: RM {total_spent:,.2f}
- Already Saved/Invested: RM {savings:,.2f}
- Spending Rate: {spending_rate:.1f}% of income
- Remaining After Expenses & Savings: RM {(income - total_spent - savings):,.2f}

SPENDING BY CATEGORY (Top 5):
{category_text}

LARGEST INDIVIDUAL EXPENSES (Top 5):
{expenses_text}

TASK:
Provide personalized financial insights in a friendly, encouraging tone. Your response should:

1. START with a warm greeting and spending summary (2-3 sentences)
   - Example: "Hey! I analyzed your spending for this month. You spent RM {total_spent:,.2f}, which is {spending_rate:.1f}% of your RM {income:,.2f} income."

2. IDENTIFY 2-3 key observations about their spending patterns
   - Point out the highest spending categories
   - Note any concerning patterns (e.g., high food delivery, unused subscriptions)
   - Celebrate positive habits if any (e.g., good savings rate, low transport costs)

3. PROVIDE 3 specific, actionable savings recommendations
   - Each recommendation should include:
     * What to do (be specific and practical)
     * Estimated monthly savings amount
     * Why this will help
   - Examples:
     * "Cook 2 more meals at home per week instead of ordering delivery. You could save around RM 150/month!"
     * "Review your subscriptions - canceling unused ones could free up RM 50/month."
     * "Use public transport 2 days a week instead of Grab - save RM 80/month on transport."

4. SUGGEST realistic budget targets for top 2 categories
   - Based on their income level
   - Example: "Try to keep Food & Dining under RM 600 (20% of income) next month."

TONE & STYLE:
- Friendly and conversational (like talking to a supportive friend)
- Encouraging and positive (no judgment!)
- Specific with numbers and percentages
- Action-oriented (tell them exactly what to do)
- Use Malaysian context (RM currency, local places/brands)

Keep your response focused and helpful - around 250-300 words.
Start with "Hey! I analyzed your spending..."
"""

        # Generate insights using Gemini
        model = genai.GenerativeModel(google_gemini_model)
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating insights: {str(e)}\n\nPlease check your API key and internet connection."


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


def generate_savings_plan(_df, income, target_savings):
    """
    Generate a specific savings plan to reach a target amount
    
    Args:
        _df (DataFrame): Categorized transaction data
        income (float): Monthly income
        target_savings (float): Target amount to save
        
    Returns:
        str: Detailed savings plan
    """
    if not configure_gemini():
        return "Unable to generate plan: API key not configured"
    
    try:
        total_spent = _df['amount_abs'].sum()
        current_disposable = income - total_spent
        
        # Category spending
        category_summary = _df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
        category_text = "\n".join([
            f"  - {cat}: RM {amount:,.2f}"
            for cat, amount in category_summary.items()
        ])
        
        prompt = f"""You are a financial advisor creating a savings plan.

USER'S SITUATION:
- Monthly Income: RM {income:,.2f}
- Current Spending: RM {total_spent:,.2f}
- Currently Available: RM {current_disposable:,.2f}
- TARGET SAVINGS GOAL: RM {target_savings:,.2f}
- Gap to Fill: RM {max(0, target_savings - current_disposable):,.2f}

CURRENT SPENDING BY CATEGORY:
{category_text}

CREATE A SAVINGS PLAN:
1. Assess if the target is realistic given their income
2. If they need to cut more expenses, suggest specific reductions per category
3. Provide a step-by-step monthly action plan
4. Be encouraging but realistic

Format as:
**Savings Plan to Reach RM {target_savings:,.2f}/month:**

[Your analysis and plan here]
"""

        model = genai.GenerativeModel(google_gemini_model)
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating savings plan: {str(e)}"


def compare_spending_benchmarks(_df, income):
    """
    Compare user's spending to general budget benchmarks
    
    Args:
        _df (DataFrame): Categorized transaction data
        income (float): Monthly income
        
    Returns:
        str: Comparison analysis
    """
    if not configure_gemini():
        return "Unable to generate comparison: API key not configured"
    
    try:
        # Calculate actual percentages
        total_spent = _df['amount_abs'].sum()
        category_summary = _df.groupby('category')['amount_abs'].sum()
        
        category_percentages = {}
        for cat, amount in category_summary.items():
            percentage = (amount / income) * 100
            category_percentages[cat] = percentage
        
        # Format for prompt
        spending_text = "\n".join([
            f"  - {cat}: {pct:.1f}% of income (RM {category_summary[cat]:,.2f})"
            for cat, pct in sorted(category_percentages.items(), key=lambda x: x[1], reverse=True)
        ])
        
        prompt = f"""You are a financial advisor comparing spending to budget guidelines.

USER'S SPENDING:
Income: RM {income:,.2f}
{spending_text}

GENERAL BUDGET GUIDELINES (50/30/20 rule adapted):
- Necessities (Food, Transport, Bills): 50-60% of income
- Wants (Entertainment, Shopping): 20-30% of income  
- Savings/Investments: 20% of income minimum

TASK:
Compare their spending to these benchmarks:
1. Are they overspending in any category?
2. Which percentages are healthy vs concerning?
3. Specific recommendations to align with guidelines
4. Celebrate what they're doing well!

Keep it encouraging and actionable.
"""

        model = genai.GenerativeModel(google_gemini_model)
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating comparison: {str(e)}"


def suggest_budget_optimization(_df, income):
    """
    Suggest optimal budget allocation
    
    Args:
        _df (DataFrame): Categorized transaction data
        income (float): Monthly income
        
    Returns:
        dict: Suggested budget per category
    """
    # Simple budget suggestions based on income level
    if income < 2000:
        # Tight budget
        suggested_budget = {
            "Food & Dining": income * 0.25,
            "Groceries": income * 0.15,
            "Transport": income * 0.10,
            "Bills & Utilities": income * 0.15,
            "Entertainment": income * 0.05,
            "Shopping": income * 0.05,
            "Healthcare": income * 0.05,
            "Savings": income * 0.20
        }
    elif income < 4000:
        # Moderate budget
        suggested_budget = {
            "Food & Dining": income * 0.20,
            "Groceries": income * 0.12,
            "Transport": income * 0.12,
            "Bills & Utilities": income * 0.15,
            "Entertainment": income * 0.08,
            "Shopping": income * 0.08,
            "Healthcare": income * 0.05,
            "Savings": income * 0.20
        }
    else:
        # Comfortable budget
        suggested_budget = {
            "Food & Dining": income * 0.18,
            "Groceries": income * 0.10,
            "Transport": income * 0.10,
            "Bills & Utilities": income * 0.12,
            "Entertainment": income * 0.10,
            "Shopping": income * 0.10,
            "Healthcare": income * 0.05,
            "Savings": income * 0.25
        }
    
    return suggested_budget


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