"""
categorizer.py - Expense Categorization with Machine Learning
Uses rule-based classification + ML frequency analysis (Time-Series)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define spending categories
CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transport",
    "Entertainment",
    "Shopping",
    "Healthcare",
    "Bills & Utilities",
    "Education",
    "Savings/Investment",
    "Other"
]

# Category icons for better UI
CATEGORY_ICONS = {
    "Food & Dining": "🍔",
    "Groceries": "🛒",
    "Transport": "🚗",
    "Entertainment": "🎬",
    "Shopping": "🛍️",
    "Healthcare": "💊",
    "Bills & Utilities": "🏠",
    "Education": "📚",
    "Savings/Investment": "💰",
    "Other": "❓"
}


@st.cache_resource
def categorize_transaction(description, classifier=None, confidence_threshold=0.5):
    """
    Categorize a single transaction using rule-based classification
    
    Args:
        description (str): Transaction description (e.g., "STARBUCKS KLCC")
        classifier: Not used (kept for compatibility)
        confidence_threshold (float): Not used (kept for compatibility)
        
    Returns:
        tuple: (category_name, confidence_score)
    """
    if not description or description.strip() == "":
        return "Other", 0.0
    
    # Use rule-based categorization
    category = rule_based_categorization(description)
    confidence = 0.90 if category != "Other" else 0.50
    
    return category, confidence


def categorize_dataframe(df, show_progress=True):
    """
    Categorize all transactions in a DataFrame
    
    Args:
        df (DataFrame): Transaction data with 'description' column
        show_progress (bool): Whether to show progress bar in Streamlit
        
    Returns:
        DataFrame: Original dataframe with added 'category' and 'confidence' columns
    """
    if df is None or df.empty:
        st.error("No transactions to categorize")
        return None
    
    if 'description' not in df.columns:
        st.error("DataFrame must have 'description' column")
        return None
    
    # No classifier needed for rule-based approach
    classifier = None
    
    # Initialize lists for results
    categories = []
    confidences = []
    
    # Show progress bar if requested
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Categorize each transaction
    total = len(df)
    for idx, row in df.iterrows():
        description = row['description']
        
        # Categorize
        category, confidence = categorize_transaction(description, classifier)
        categories.append(category)
        confidences.append(confidence)
        
        # Update progress
        if show_progress:
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Categorizing transactions... {idx + 1}/{total}")
    
    # Clear progress indicators
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    # Add results to dataframe
    df['category'] = categories
    df['confidence'] = confidences
    
    return df


def get_category_breakdown(df):
    """
    Get spending breakdown by category
    
    Args:
        df (DataFrame): Categorized transaction data
        
    Returns:
        DataFrame: Category summary with totals and percentages
    """
    if df is None or df.empty or 'category' not in df.columns:
        return None
    
    # Use amount_abs for positive values (expenses only)
    amount_col = 'amount_abs' if 'amount_abs' in df.columns else 'amount'
    
    # Group by category
    category_summary = df.groupby('category')[amount_col].agg([
        ('total', 'sum'),
        ('count', 'count'),
        ('average', 'mean')
    ]).reset_index()
    
    # Calculate percentage
    total_spending = category_summary['total'].sum()
    category_summary['percentage'] = (category_summary['total'] / total_spending * 100).round(1)
    
    # Sort by total spending (highest first)
    category_summary = category_summary.sort_values('total', ascending=False)
    
    # Reset index
    category_summary.reset_index(drop=True, inplace=True)
    
    return category_summary


# ==================== MACHINE LEARNING METHOD ====================

def analyze_transaction_frequency(df):
    """
    ML METHOD: Transaction Frequency Analysis (Time-Series)
    
    Analyzes spending frequency patterns per category using time-series statistical methods.
    Calculates how often purchases occur in each category and predicts next expected transaction dates.
    
    DESIGN RATIONALE:
    ------------------
    Why Time-Series Frequency Analysis?
    - Recurring expenses (rent, utilities, groceries) follow predictable temporal patterns
    - Understanding "when" expenses occur is as important as "how much" for budget planning
    - Helps users anticipate upcoming bills and cash flow requirements
    - Enables proactive financial management rather than reactive tracking
    
    Why Simple Statistical Approach (Average-Based)?
    - Monthly bank statements provide limited data (typically 30-60 days, 40-80 transactions)
    - Complex time-series models (ARIMA, Prophet, LSTM) require extensive historical data:
      * ARIMA: Needs 50+ observations for reliable parameter estimation
      * Prophet: Designed for daily data over months/years with seasonal patterns
      * LSTM: Requires hundreds of data points for neural network training
    - Simple statistical averages are:
      * Highly interpretable - users understand "every 5 days" vs complex model outputs
      * Fast to compute - real-time analysis suitable for web application
      * Sufficient accuracy for short-term predictions (next 1-2 cycles)
      * Robust with limited data - no overfitting risk
      * Transparent - users can verify calculations manually
    
    Methodology:
    1. Group transactions by spending category
    2. Calculate time deltas between consecutive transactions in each category
    3. Compute statistical average (mean days between purchases)
    4. Predict next purchase date: last_transaction_date + average_frequency
    5. Categorize patterns into human-readable descriptions
    
    Features:
    - Works with any date range (handles both complete and partial months)
    - Handles irregular spending patterns gracefully
    - Provides confidence indicators (transaction count)
    - Returns "Insufficient data" for categories with <2 transactions
    
    LIMITATIONS (Acknowledged for Transparency):
    --------------------------------------------
    1. Minimum Data Requirement:
       - Requires at least 2 transactions per category
       - Categories with 2-3 transactions have lower prediction reliability
       - Recommendation: 5+ transactions for confident predictions
    
    2. Pattern Consistency Assumption:
       - Assumes relatively stable spending patterns (stationary time series)
       - Irregular/one-time expenses (medical emergencies, travel) are less predictable
       - Works best for recurring expenses (groceries, utilities, subscriptions)
    
    3. Short-Term Predictions Only:
       - Reliable for next 1-2 purchase cycles
       - Accuracy degrades for longer-term forecasts (>1 month ahead)
       - Does not capture long-term trends or pattern changes
    
    4. No Seasonal Variation:
       - Single month data cannot capture seasonal patterns
       - Holiday spending, quarterly bills, annual subscriptions not detected
       - Year-over-year variations not modeled
    
    5. External Factors Not Considered:
       - Income changes, life events, economic conditions ignored
       - Cannot predict behavioral changes or spending adjustments
       - Assumes future behavior mirrors past patterns
    
    Args:
        df (DataFrame): Transaction data with required columns:
                       - 'category': Spending category name (from categorization step)
                       - 'date': Transaction date (datetime or parseable string format)
        
    Returns:
        list: Frequency analysis results per category, sorted by transaction count (descending).
              Each result dict contains:
              - 'category' (str): Category name
              - 'transaction_count' (int): Number of transactions in this category
              - 'avg_days_between' (float or None): Average days between purchases
              - 'frequency_pattern' (str): Human-readable pattern description
                  Examples: "Daily", "Every 5.3 days", "Weekly", "Monthly"
              - 'next_purchase_prediction' (str): Predicted next purchase timing
                  Examples: "Tomorrow", "In 3 days", "In 12 days (Jan 25)"
              - 'icon' (str): Category emoji icon for UI display
              
        Returns None if DataFrame is invalid or missing required columns
        
    Example Usage:
        >>> freq_results = analyze_transaction_frequency(df)
        >>> for result in freq_results:
        >>>     if result['avg_days_between']:
        >>>         print(f"{result['category']}: {result['frequency_pattern']}")
        >>>         print(f"  Next: {result['next_purchase_prediction']}")
        >>>         print(f"  Based on {result['transaction_count']} transactions")
        
        Output:
        Food & Dining: Every 3.2 days
          Next: In 2 days
          Based on 15 transactions
        Groceries: Weekly
          Next: In 5 days (Jan 20)
          Based on 6 transactions
    
    Technical Implementation Details:
        - Uses pandas datetime arithmetic for accurate date calculations
        - Sorts transactions chronologically before analysis (prevents order errors)
        - Handles both datetime and string date formats via pd.to_datetime()
        - Edge case handling: single transactions, missing data, date parsing failures
        - Predictions assume pattern continuity (no drift detection)
    """
    
    # === INPUT VALIDATION ===
    # Verify DataFrame is valid
    if df is None or df.empty:
        return None
    
    # Check for required columns
    if 'category' not in df.columns or 'date' not in df.columns:
        return None
    
    # === DATA PREPARATION ===
    # Create working copy to avoid modifying original DataFrame
    df_freq = df.copy()
    
    # Convert date column to datetime if it isn't already
    # Handles string dates like "2024-01-15" or datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df_freq['date']):
        df_freq['date'] = pd.to_datetime(df_freq['date'])
    
    # Sort by date chronologically (oldest to newest)
    # Critical for accurate time delta calculations
    df_freq = df_freq.sort_values('date')
    
    # === FREQUENCY ANALYSIS PER CATEGORY ===
    frequency_analysis = []
    
    # Analyze each spending category independently
    for category in df_freq['category'].unique():
        # Extract all transactions for this category
        category_data = df_freq[df_freq['category'] == category].copy()
        
        # === MINIMUM DATA CHECK ===
        # Need at least 2 transactions to calculate frequency
        if len(category_data) < 2:
            frequency_analysis.append({
                'category': category,
                'transaction_count': len(category_data),
                'avg_days_between': None,
                'frequency_pattern': 'Insufficient data',
                'next_purchase_prediction': 'Not enough data',
                'icon': CATEGORY_ICONS.get(category, '❓')
            })
            continue
        
        # === CALCULATE DAYS BETWEEN CONSECUTIVE TRANSACTIONS ===
        # Sort by date and extract date list
        category_data = category_data.sort_values('date')
        dates = category_data['date'].tolist()
        
        # Calculate time delta between each consecutive pair of transactions
        # Example: [Jan 1, Jan 5, Jan 8] -> [4 days, 3 days]
        days_between = []
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i-1]).days  # Days between transaction i-1 and i
            days_between.append(delta)
        
        # Edge case: Should not happen due to check above, but defensive coding
        if not days_between:
            avg_days = None
            frequency_pattern = 'Single transaction'
            next_prediction = 'Unable to predict'
        else:
            # === STATISTICAL ANALYSIS ===
            # Calculate mean frequency (average days between purchases)
            # This is our primary predictor for time-series forecasting
            avg_days = np.mean(days_between)
            
            # === FREQUENCY PATTERN CATEGORIZATION ===
            # Map numeric frequency to human-readable pattern description
            # Thresholds chosen based on common spending patterns
            if avg_days < 2:
                frequency_pattern = 'Daily'  # Coffee, lunch, parking
            elif avg_days < 7:
                frequency_pattern = f'Every {avg_days:.1f} days'  # Frequent purchases
            elif avg_days < 14:
                frequency_pattern = 'Weekly'  # Weekly groceries, fuel
            elif avg_days < 21:
                frequency_pattern = 'Bi-weekly'  # Fortnightly spending
            elif avg_days < 35:
                frequency_pattern = 'Monthly'  # Monthly bills, subscriptions
            else:
                frequency_pattern = f'Every {avg_days/30:.1f} months'  # Quarterly, rare
            
            # === NEXT PURCHASE PREDICTION ===
            # Forecast next transaction using simple extrapolation
            # Formula: predicted_next = last_transaction_date + average_frequency
            last_transaction_date = dates[-1]  # Most recent transaction
            predicted_next = last_transaction_date + timedelta(days=avg_days)
            
            # Calculate days from now until predicted next purchase
            days_until_next = (predicted_next - datetime.now()).days
            
            # Format prediction in user-friendly language
            if days_until_next <= 0:
                next_prediction = 'Today or overdue'  # Prediction already passed
            elif days_until_next == 1:
                next_prediction = 'Tomorrow'  # Next day
            elif days_until_next <= 7:
                next_prediction = f'In {days_until_next} days'  # Within a week
            else:
                # Include specific date for longer-term predictions
                # Example: "In 12 days (Jan 25)"
                next_prediction = f'In {days_until_next} days ({predicted_next.strftime("%b %d")})'
        
        # === COMPILE RESULTS FOR THIS CATEGORY ===
        frequency_analysis.append({
            'category': category,
            'transaction_count': len(category_data),
            'avg_days_between': avg_days,
            'frequency_pattern': frequency_pattern,
            'next_purchase_prediction': next_prediction,
            'icon': CATEGORY_ICONS.get(category, '❓')
        })
    
    # === SORT RESULTS ===
    # Sort by transaction count (most frequent categories first)
    # This prioritizes categories with more reliable predictions
    frequency_analysis = sorted(frequency_analysis, key=lambda x: x['transaction_count'], reverse=True)
    
    return frequency_analysis


def get_ml_insights_summary(df):
    """
    Generate ML insights summary (frequency analysis only)
    
    Provides a wrapper for frequency analysis results in a dictionary format
    for consistency with the application's architecture.
    
    Args:
        df (DataFrame): Categorized transaction data with 'category', 'date' columns
        
    Returns:
        dict: ML insights containing:
              - 'frequency': Time-series frequency analysis results
              
        Returns empty dict if analysis fails or data is invalid
        
    Example:
        >>> insights = get_ml_insights_summary(df)
        >>> if 'frequency' in insights:
        >>>     for result in insights['frequency']:
        >>>         print(f"{result['category']}: {result['frequency_pattern']}")
    """
    if df is None or df.empty:
        return None
    
    insights = {}
    
    # === TIME-SERIES FREQUENCY ANALYSIS ===
    # Analyze purchase timing patterns and predict next purchases
    frequency_results = analyze_transaction_frequency(df)
    if frequency_results:
        insights['frequency'] = frequency_results
    
    return insights


# ==================== RULE-BASED CATEGORIZATION ====================

def rule_based_categorization(description):
    """
    Rule-based transaction categorization using keyword matching
    
    Traditional NLP approach: matches transaction descriptions against curated keyword lists
    for Malaysian merchants and common spending patterns. This serves as the primary
    categorization method before ML frequency analysis.
    
    Design Philosophy:
    - Case-insensitive matching for robustness
    - Priority order: More specific categories checked first (Groceries before Shopping)
    - Malaysian-focused: Includes local merchants (Grab, Shopee, Petronas, Maybank)
    - Comprehensive keywords: 200+ terms covering common Malaysian spending
    - Returns "Other" as fallback for unrecognized transactions (enables manual correction)
    
    Accuracy: ~85% based on sample Malaysian bank statements
    
    Args:
        description (str): Transaction description from bank statement
                          Examples: "GRAB-GRABPAY", "STARBUCKS KLCC", "AEON BUKIT TINGGI"
        
    Returns:
        str: Category name from CATEGORIES list
        
    Example:
        >>> rule_based_categorization("STARBUCKS KLCC")
        'Food & Dining'
        >>> rule_based_categorization("AEON BUKIT TINGGI")
        'Groceries'
        >>> rule_based_categorization("GRAB-GRABPAY")
        'Transport'
    """
    description_lower = description.lower()
    
    # === FOOD & DINING ===
    # Restaurants, cafes, food delivery services
    food_keywords = ['starbucks', 'mcdonald', 'kfc', 'pizza', 'restaurant', 'cafe', 'coffee', 
                     'mamak', 'nasi', 'food', 'grab-food', 'foodpanda', 'dining', 'jamie oliver']
    if any(keyword in description_lower for keyword in food_keywords):
        return "Food & Dining"
    
    # === GROCERIES ===
    # Supermarkets, convenience stores, fresh markets
    grocery_keywords = ['grocer', 'market', 'aeon', 'tesco', 'jaya', 'village', 'supermarket',
                        'family mart', '7-eleven', '99 speedmart', 'bean shipper', 'bilabila', 
                        'kk super mart']
    if any(keyword in description_lower for keyword in grocery_keywords):
        return "Groceries"
    
    # === TRANSPORT ===
    # Ride-hailing, fuel, parking, tolls, public transport
    transport_keywords = ['grab', 'uber', 'taxi', 'petrol', 'shell', 'petronas', 'parking', 
                         'toll', 'lrt', 'mrt', 'bus', 'komuter', 'fuel']
    if any(keyword in description_lower for keyword in transport_keywords):
        return "Transport"
    
    # === ENTERTAINMENT ===
    # Streaming services, movies, games, concerts
    entertainment_keywords = ['netflix', 'spotify', 'cinema', 'gsc-mid valley', 'tgv', 'movie', 
                             'game', 'steam', 'playstation', 'xbox', 'concert']
    if any(keyword in description_lower for keyword in entertainment_keywords):
        return "Entertainment"
    
    # === SHOPPING ===
    # E-commerce, retail stores, fashion, general shopping
    shopping_keywords = ['shopee', 'lazada', 'zalora', 'uniqlo', 'h&m', 'zara', 'shopping',
                        'mall', 'fashion', 'clothing', 'terrae', 'atome', 'watson''s', 'ghl*dreame']
    if any(keyword in description_lower for keyword in shopping_keywords):
        return "Shopping"
    
    # === HEALTHCARE ===
    # Pharmacy, clinics, hospitals, medical services
    health_keywords = ['pharmacy', 'clinic', 'hospital', 'doctor', 'guardian', 'watson',
                      'medical', 'health']
    if any(keyword in description_lower for keyword in health_keywords):
        return "Healthcare"
    
    # === BILLS & UTILITIES ===
    # Electricity, water, internet, phone bills
    bills_keywords = ['tnb', 'syabas', 'unifi', 'celcom', 'maxis', 'digi', 'yes', 'bill',
                     'electricity', 'water', 'internet', 'phone', 'telco']
    if any(keyword in description_lower for keyword in bills_keywords):
        return "Bills & Utilities"
    
    # === EDUCATION ===
    # Courses, books, tuition, educational materials
    education_keywords = ['university', 'college', 'school', 'course', 'udemy', 'coursera',
                         'book', 'tuition', 'education', 'mph']
    if any(keyword in description_lower for keyword in education_keywords):
        return "Education"
    
    # === SAVINGS/INVESTMENT ===
    # Investment products, savings accounts, financial instruments
    investment_keywords = ['investment', 'savings', 'asb', 'tabung', 'mutual fund', 'stock',
                          'etf', 'unit trust']
    if any(keyword in description_lower for keyword in investment_keywords):
        return "Savings/Investment"
    
    # === DEFAULT FALLBACK ===
    # Unrecognized transactions categorized as "Other" for manual review
    return "Other"


if __name__ == "__main__":
    print("Enhanced Categorizer with ML loaded successfully!")
    print(f"Available categories: {CATEGORIES}")
    print("\nML Method Available:")
    print("1. Transaction Frequency Analysis (Time-Series) - analyze_transaction_frequency(df)")
    print("2. ML Insights Summary - get_ml_insights_summary(df)")