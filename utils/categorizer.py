"""
categorizer.py - AI-Powered Expense Categorization
Uses HuggingFace transformers for zero-shot classification
"""

from transformers import pipeline
import streamlit as st
import pandas as pd

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


@st.cache_resource
def load_classifier():
    """
    Load the HuggingFace zero-shot classification model
    
    Uses caching to avoid reloading the model on every run
    This model download happens only once (first time)
    
    Returns:
        HuggingFace pipeline for zero-shot classification
    """
    try:
        # Load zero-shot classification pipeline
        # Using facebook/bart-large-mnli - good balance of speed and accuracy
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli", #Default (500MB)
            #model="typeform/distilbert-base-uncased-mnli",  # Smaller (250MB)
            device=-1  # Use CPU (set to 0 for GPU if available)
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading AI model: {str(e)}")
        return None


def categorize_transaction(description, classifier, confidence_threshold=0.5):
    """
    Categorize a single transaction using zero-shot classification
    
    Args:
        description (str): Transaction description (e.g., "STARBUCKS KLCC")
        classifier: HuggingFace pipeline object
        confidence_threshold (float): Minimum confidence to accept category (0-1)
        
    Returns:
        tuple: (category_name, confidence_score)
    """
    if not description or description.strip() == "":
        return "Other", 0.0
    
    try:
        # Run zero-shot classification
        result = classifier(
            description,
            candidate_labels=CATEGORIES,
            multi_label=False
        )
        
        # Get top prediction
        top_category = result['labels'][0]
        top_score = result['scores'][0]
        
        # If confidence is below threshold, mark as "Other"
        if top_score < confidence_threshold:
            return "Other", top_score
        
        return top_category, top_score
        
    except Exception as e:
        # If categorization fails, return Other
        return "Other", 0.0


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
    
    # Load the classifier
    classifier = load_classifier()
    
    if classifier is None:
        st.error("Failed to load AI classifier")
        return None
    
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


def get_top_expenses_by_category(df, category_name, top_n=5):
    """
    Get top N expenses within a specific category
    
    Args:
        df (DataFrame): Categorized transaction data
        category_name (str): Category to filter by
        top_n (int): Number of top expenses to return
        
    Returns:
        DataFrame: Top expenses in the category
    """
    if df is None or df.empty or 'category' not in df.columns:
        return None
    
    # Filter by category
    category_df = df[df['category'] == category_name].copy()
    
    if category_df.empty:
        return None
    
    # Sort by amount (highest first)
    amount_col = 'amount_abs' if 'amount_abs' in df.columns else 'amount'
    category_df = category_df.nlargest(top_n, amount_col)
    
    return category_df[['date', 'description', amount_col, 'confidence']]


def get_categorization_stats(df):
    """
    Get statistics about the categorization process
    
    Args:
        df (DataFrame): Categorized transaction data
        
    Returns:
        dict: Statistics about categorization quality
    """
    if df is None or df.empty or 'category' not in df.columns:
        return None
    
    stats = {
        'total_transactions': len(df),
        'categorized': len(df[df['category'] != 'Other']),
        'uncategorized': len(df[df['category'] == 'Other']),
        'average_confidence': df['confidence'].mean(),
        'high_confidence': len(df[df['confidence'] > 0.8]),
        'low_confidence': len(df[df['confidence'] < 0.5]),
        'unique_categories': df['category'].nunique()
    }
    
    stats['categorization_rate'] = (stats['categorized'] / stats['total_transactions'] * 100)
    
    return stats


def suggest_recategorization(df, confidence_threshold=0.5):
    """
    Find transactions that might need manual review
    
    Args:
        df (DataFrame): Categorized transaction data
        confidence_threshold (float): Confidence below which to flag
        
    Returns:
        DataFrame: Transactions with low confidence scores
    """
    if df is None or df.empty or 'confidence' not in df.columns:
        return None
    
    # Filter low confidence transactions
    low_confidence = df[df['confidence'] < confidence_threshold].copy()
    
    if low_confidence.empty:
        return None
    
    # Sort by confidence (lowest first)
    low_confidence = low_confidence.sort_values('confidence')
    
    return low_confidence[['date', 'description', 'category', 'confidence', 'amount_abs']]


# Rule-based fallback (optional - for when AI categorization fails)
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
                     'mamak', 'nasi', 'food', 'grab-food', 'foodpanda', 'dining']
    if any(keyword in description_lower for keyword in food_keywords):
        return "Food & Dining"
    
    # Groceries
    grocery_keywords = ['grocer', 'market', 'aeon', 'tesco', 'jaya', 'village', 'supermarket',
                        'family mart', '7-eleven', '99 speedmart', 'bean shipper']
    if any(keyword in description_lower for keyword in grocery_keywords):
        return "Groceries"
    
    # Transport
    transport_keywords = ['grab', 'uber', 'taxi', 'petrol', 'shell', 'petronas', 'parking', 
                         'toll', 'lrt', 'mrt', 'bus', 'komuter', 'fuel']
    if any(keyword in description_lower for keyword in transport_keywords):
        return "Transport"
    
    # Entertainment
    entertainment_keywords = ['netflix', 'spotify', 'cinema', 'gsc', 'tgv', 'movie', 'game',
                             'steam', 'playstation', 'xbox', 'concert']
    if any(keyword in description_lower for keyword in entertainment_keywords):
        return "Entertainment"
    
    # Shopping
    shopping_keywords = ['shopee', 'lazada', 'zalora', 'uniqlo', 'h&m', 'zara', 'shopping',
                        'mall', 'fashion', 'clothing']
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
                         'book', 'tuition', 'education']
    if any(keyword in description_lower for keyword in education_keywords):
        return "Education"
    
    # Savings/Investment
    investment_keywords = ['investment', 'savings', 'asb', 'tabung', 'mutual fund', 'stock',
                          'etf', 'unit trust']
    if any(keyword in description_lower for keyword in investment_keywords):
        return "Savings/Investment"
    
    # Default
    return "Other"


def hybrid_categorize_transaction(description, classifier, use_ai=True):
    """
    Hybrid approach: Try AI first, fallback to rules if confidence is low
    
    Args:
        description (str): Transaction description
        classifier: HuggingFace pipeline
        use_ai (bool): Whether to use AI or just rules
        
    Returns:
        tuple: (category, confidence, method_used)
    """
    if use_ai and classifier is not None:
        # Try AI categorization
        category, confidence = categorize_transaction(description, classifier)
        
        # If high confidence, use AI result
        if confidence > 0.6:
            return category, confidence, "AI"
    
    # Otherwise, use rule-based
    category = rule_based_categorization(description)
    return category, 0.9, "Rules"  # Assume high confidence for rule-based


if __name__ == "__main__":
    # Test the categorizer
    print("Categorizer module loaded successfully!")
    print(f"Available categories: {CATEGORIES}")
    print("\nTest categorization:")
    
    # Test with sample descriptions
    test_descriptions = [
        "STARBUCKS KLCC",
        "GRAB-RIDE JALAN AMPANG",
        "SHOPEE PURCHASE",
        "TNB ELECTRICITY BILL"
    ]
    
    print("\nRule-based categorization test:")
    for desc in test_descriptions:
        category = rule_based_categorization(desc)
        print(f"  {desc} → {category}")