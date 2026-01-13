"""
categorizer.py - Expense Categorization with Machine Learning
Uses rule-based classification + ML analysis (K-Means Clustering & Frequency Analysis)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

# Category icons for better UI (optional)
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

# Cluster profile descriptions
CLUSTER_PROFILES = {
    0: {"name": "Daily Small Purchases", "icon": "🟢", "description": "Frequent low-value transactions"},
    1: {"name": "Weekly Mid-Range", "icon": "🟡", "description": "Regular moderate spending"},
    2: {"name": "Monthly Large Bills", "icon": "🔴", "description": "Infrequent high-value payments"}
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


# ==================== MACHINE LEARNING METHODS ====================

def perform_clustering_analysis(df, n_clusters=3):
    """
    ML METHOD 1: K-Means Clustering Analysis
    Groups transactions into spending patterns based on amount and frequency
    
    Args:
        df (DataFrame): Transaction data with 'amount_abs' and 'date' columns
        n_clusters (int): Number of clusters (default: 3)
        
    Returns:
        dict: Clustering results with cluster assignments and analysis
    """
    if df is None or df.empty:
        return None
    
    if 'amount_abs' not in df.columns:
        return None
    
    # Prepare features for clustering
    # Feature 1: Transaction amount
    # Feature 2: Day of month (to capture timing patterns)
    features = []
    
    for idx, row in df.iterrows():
        amount = row['amount_abs']
        
        # Extract day of month if date column exists
        if 'date' in df.columns:
            try:
                if isinstance(row['date'], str):
                    date_obj = pd.to_datetime(row['date'])
                else:
                    date_obj = row['date']
                day_of_month = date_obj.day
            except:
                day_of_month = 15  # Default to mid-month
        else:
            day_of_month = 15
        
        features.append([amount, day_of_month])
    
    features_array = np.array(features)
    
    # Standardize features (important for K-Means)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Analyze each cluster
    cluster_analysis = []
    
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Calculate cluster statistics
        avg_amount = cluster_data['amount_abs'].mean()
        total_amount = cluster_data['amount_abs'].sum()
        transaction_count = len(cluster_data)
        
        # Get top categories in this cluster
        if 'category' in cluster_data.columns:
            top_categories = cluster_data['category'].value_counts().head(3).to_dict()
        else:
            top_categories = {}
        
        # Assign cluster profile based on average amount
        if avg_amount < 50:
            profile_id = 0  # Small purchases
        elif avg_amount < 200:
            profile_id = 1  # Mid-range
        else:
            profile_id = 2  # Large bills
        
        cluster_info = {
            'cluster_id': cluster_id,
            'profile': CLUSTER_PROFILES[profile_id],
            'transaction_count': transaction_count,
            'avg_amount': avg_amount,
            'total_amount': total_amount,
            'top_categories': top_categories,
            'percentage_of_transactions': (transaction_count / len(df)) * 100
        }
        
        cluster_analysis.append(cluster_info)
    
    # Sort clusters by average amount (ascending)
    cluster_analysis = sorted(cluster_analysis, key=lambda x: x['avg_amount'])
    
    return {
        'df_with_clusters': df_clustered,
        'cluster_analysis': cluster_analysis,
        'n_clusters': n_clusters
    }


def analyze_transaction_frequency(df):
    """
    ML METHOD 2: Transaction Frequency Analysis
    Analyzes spending frequency patterns per category using time series analysis
    
    Args:
        df (DataFrame): Transaction data with 'category' and 'date' columns
        
    Returns:
        dict: Frequency analysis results per category
    """
    if df is None or df.empty:
        return None
    
    if 'category' not in df.columns or 'date' not in df.columns:
        return None
    
    # Convert date column to datetime if it isn't already
    df_freq = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_freq['date']):
        df_freq['date'] = pd.to_datetime(df_freq['date'])
    
    # Sort by date
    df_freq = df_freq.sort_values('date')
    
    frequency_analysis = []
    
    # Analyze each category
    for category in df_freq['category'].unique():
        category_data = df_freq[df_freq['category'] == category].copy()
        
        if len(category_data) < 2:
            # Not enough data for frequency analysis
            frequency_analysis.append({
                'category': category,
                'transaction_count': len(category_data),
                'avg_days_between': None,
                'frequency_pattern': 'Insufficient data',
                'next_purchase_prediction': 'Not enough data',
                'icon': CATEGORY_ICONS.get(category, '❓')
            })
            continue
        
        # Calculate days between transactions
        category_data = category_data.sort_values('date')
        dates = category_data['date'].tolist()
        
        days_between = []
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i-1]).days
            days_between.append(delta)
        
        if not days_between:
            avg_days = None
            frequency_pattern = 'Single transaction'
            next_prediction = 'Unable to predict'
        else:
            avg_days = np.mean(days_between)
            
            # Categorize frequency pattern
            if avg_days < 2:
                frequency_pattern = 'Daily'
            elif avg_days < 7:
                frequency_pattern = f'Every {avg_days:.1f} days'
            elif avg_days < 14:
                frequency_pattern = 'Weekly'
            elif avg_days < 21:
                frequency_pattern = 'Bi-weekly'
            elif avg_days < 35:
                frequency_pattern = 'Monthly'
            else:
                frequency_pattern = f'Every {avg_days/30:.1f} months'
            
            # Predict next purchase
            last_transaction_date = dates[-1]
            predicted_next = last_transaction_date + timedelta(days=avg_days)
            days_until_next = (predicted_next - datetime.now()).days
            
            if days_until_next <= 0:
                next_prediction = 'Today or overdue'
            elif days_until_next == 1:
                next_prediction = 'Tomorrow'
            elif days_until_next <= 7:
                next_prediction = f'In {days_until_next} days'
            else:
                next_prediction = f'In {days_until_next} days ({predicted_next.strftime("%b %d")})'
        
        frequency_analysis.append({
            'category': category,
            'transaction_count': len(category_data),
            'avg_days_between': avg_days,
            'frequency_pattern': frequency_pattern,
            'next_purchase_prediction': next_prediction,
            'icon': CATEGORY_ICONS.get(category, '❓')
        })
    
    # Sort by transaction count (descending)
    frequency_analysis = sorted(frequency_analysis, key=lambda x: x['transaction_count'], reverse=True)
    
    return frequency_analysis


def get_ml_insights_summary(df):
    """
    Generate a comprehensive ML insights summary combining clustering and frequency analysis
    
    Args:
        df (DataFrame): Categorized transaction data
        
    Returns:
        dict: Combined ML insights
    """
    if df is None or df.empty:
        return None
    
    insights = {}
    
    # Clustering analysis
    clustering_results = perform_clustering_analysis(df, n_clusters=3)
    if clustering_results:
        insights['clustering'] = clustering_results
    
    # Frequency analysis
    frequency_results = analyze_transaction_frequency(df)
    if frequency_results:
        insights['frequency'] = frequency_results
    
    return insights


# ==================== RULE-BASED CATEGORIZATION ====================

def rule_based_categorization(description):
    """
    Simple rule-based categorization as fallback
    Uses keyword matching
    
    Args:
        description (str): Transaction description
        
    Returns:
        str: Category name
    """
    description_lower = description.lower()
    
    # Food & Dining
    food_keywords = ['starbucks', 'mcdonald', 'kfc', 'pizza', 'restaurant', 'cafe', 'coffee', 
                     'mamak', 'nasi', 'food', 'grab-food', 'foodpanda', 'dining','jamie oliver']
    if any(keyword in description_lower for keyword in food_keywords):
        return "Food & Dining"
    
    # Groceries
    grocery_keywords = ['grocer', 'market', 'aeon', 'tesco', 'jaya', 'village', 'supermarket',
                        'family mart', '7-eleven', '99 speedmart', 'bean shipper','bilabila','kk super mart']
    if any(keyword in description_lower for keyword in grocery_keywords):
        return "Groceries"
    
    # Transport
    transport_keywords = ['grab', 'uber', 'taxi', 'petrol', 'shell', 'petronas', 'parking', 
                         'toll', 'lrt', 'mrt', 'bus', 'komuter', 'fuel']
    if any(keyword in description_lower for keyword in transport_keywords):
        return "Transport"
    
    # Entertainment
    entertainment_keywords = ['netflix', 'spotify', 'cinema', 'gsc-mid valley', 'tgv', 'movie', 'game',
                             'steam', 'playstation', 'xbox', 'concert']
    if any(keyword in description_lower for keyword in entertainment_keywords):
        return "Entertainment"
    
    # Shopping
    shopping_keywords = ['shopee', 'lazada', 'zalora', 'uniqlo', 'h&m', 'zara', 'shopping',
                        'mall', 'fashion', 'clothing','terrae','atome','watson''s','ghl*dreame']
    if any(keyword in description_lower for keyword in shopping_keywords):
        return "Shopping"
    
    # Healthcare
    health_keywords = ['pharmacy', 'clinic', 'hospital', 'doctor', 'guardian', 'watson',
                      'medical', 'health']
    if any(keyword in description_lower for keyword in health_keywords):
        return "Healthcare"
    
    # Bills & Utilities
    bills_keywords = ['tnb', 'syabas', 'unifi', 'celcom', 'maxis', 'digi', 'yes', 'bill',
                     'electricity', 'water', 'internet', 'phone', 'telco']
    if any(keyword in description_lower for keyword in bills_keywords):
        return "Bills & Utilities"
    
    # Education
    education_keywords = ['university', 'college', 'school', 'course', 'udemy', 'coursera',
                         'book', 'tuition', 'education','mph']
    if any(keyword in description_lower for keyword in education_keywords):
        return "Education"
    
    # Savings/Investment
    investment_keywords = ['investment', 'savings', 'asb', 'tabung', 'mutual fund', 'stock',
                          'etf', 'unit trust']
    if any(keyword in description_lower for keyword in investment_keywords):
        return "Savings/Investment"
    
    # Default
    return "Other"


if __name__ == "__main__":
    print("Enhanced Categorizer with ML loaded successfully!")
    print(f"Available categories: {CATEGORIES}")
    print("\nML Methods Available:")
    print("1. K-Means Clustering Analysis - perform_clustering_analysis(df)")
    print("2. Transaction Frequency Analysis - analyze_transaction_frequency(df)")
    print("3. Combined ML Insights - get_ml_insights_summary(df)")