"""
parser.py - Bank Statement Parsing Utilities
Handles CSV and PDF parsing for Maybank statements
"""

import pandas as pd
import pdfplumber
import streamlit as st
from io import StringIO

def parse_maybank_csv(file):
    """
    Parse Maybank CSV bank statement
    
    Expected format:
    Entry Date | Value Date | Transaction Description | Transaction Amount | Statement Balance
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        pandas DataFrame with columns: date, description, amount, balance, transaction_type
    """
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Standardize column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Expected columns (flexible matching)
        # Maybank format: entry_date, value_date, transaction_description, transaction_amount, statement_balance
        
        # Rename columns to standard format
        column_mapping = {
            'entry_date': 'date',
            'transaction_description': 'description',
            'transaction_amount': 'amount',
            'statement_balance': 'balance'
        }
        
        # Apply renaming
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Keep only relevant columns
        required_columns = ['date', 'description', 'amount', 'balance']
        df = df[required_columns]
        
        # Clean amount column (remove commas, convert to float)
        df['amount'] = df['amount'].astype(str).str.replace(',', '').str.replace('RM', '').str.strip()
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Determine transaction type (expense vs income)
        df['transaction_type'] = df['amount'].apply(lambda x: 'expense' if x < 0 else 'income')
        
        # Make all amounts positive for easier display
        df['amount_abs'] = df['amount'].abs()
        
        # Filter out any rows with missing data
        df = df.dropna(subset=['description', 'amount'])
        
        # Sort by date (most recent first)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date', ascending=False)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        st.error("Please make sure your CSV has the format: Entry Date | Value Date | Description | Amount | Balance")
        return None


def parse_maybank_pdf(file):
    """
    Parse Maybank PDF bank statement
    
    Expected format: Table with columns
    Entry Date | Value Date | Transaction Description | Transaction Amount | Statement Balance
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        pandas DataFrame with columns: date, description, amount, balance, transaction_type
    """
    try:
        transactions = []
        
        # Open PDF file
        with pdfplumber.open(file) as pdf:
            # Iterate through all pages
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from the page
                tables = page.extract_tables()
                
                if not tables:
                    continue
                
                # Process each table on the page
                for table in tables:
                    if not table or len(table) < 2:  # Need at least header + 1 row
                        continue
                    
                    # Skip header row (first row)
                    for row in table[1:]:
                        # Skip empty rows or rows with insufficient columns
                        if not row or len(row) < 4:
                            continue
                        
                        # Extract data (adjust indices based on Maybank PDF format)
                        # Typical format: [Entry Date, Value Date, Description, Amount, Balance]
                        try:
                            entry_date = row[0] if row[0] else ""
                            description = row[2] if len(row) > 2 else ""
                            amount = row[3] if len(row) > 3 else "0"
                            balance = row[4] if len(row) > 4 else ""
                            
                            # Skip rows that don't look like transactions
                            if not description or description.strip() == "":
                                continue
                            
                            # Clean amount (remove currency symbols, commas)
                            amount_clean = str(amount).replace('RM', '').replace(',', '').strip()
                            
                            # Skip if amount is not a number
                            try:
                                amount_float = float(amount_clean)
                            except:
                                continue
                            
                            transactions.append({
                                'date': entry_date,
                                'description': description.strip(),
                                'amount': amount_float,
                                'balance': balance
                            })
                            
                        except Exception as row_error:
                            # Skip problematic rows
                            continue
        
        if not transactions:
            st.error("No transactions found in PDF. Please check the file format.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Determine transaction type
        df['transaction_type'] = df['amount'].apply(lambda x: 'expense' if x < 0 else 'income')
        
        # Make all amounts positive for display
        df['amount_abs'] = df['amount'].abs()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Sort by date (most recent first)
        df = df.sort_values('date', ascending=False)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        st.error("Please make sure your PDF is a valid Maybank statement.")
        return None


def filter_expenses_only(df):
    """
    Filter dataframe to show only expenses (negative transactions)
    
    Args:
        df: DataFrame from parse_maybank_csv or parse_maybank_pdf
        
    Returns:
        DataFrame with only expense transactions
    """
    if df is None or df.empty:
        return None
    
    expenses_df = df[df['transaction_type'] == 'expense'].copy()
    return expenses_df


def get_transaction_summary(df):
    """
    Get summary statistics from transactions
    
    Args:
        df: DataFrame from parse_maybank_csv or parse_maybank_pdf
        
    Returns:
        Dictionary with summary stats
    """
    if df is None or df.empty:
        return None
    
    # Filter expenses only
    expenses = df[df['transaction_type'] == 'expense']
    
    summary = {
        'total_transactions': len(df),
        'total_expenses': len(expenses),
        'total_spent': expenses['amount_abs'].sum(),
        'average_transaction': expenses['amount_abs'].mean(),
        'largest_expense': expenses['amount_abs'].max(),
        'date_range_start': df['date'].min(),
        'date_range_end': df['date'].max()
    }
    
    return summary


def validate_statement_format(df):
    """
    Validate that the parsed dataframe has the required columns
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Boolean: True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    required_columns = ['date', 'description', 'amount']
    
    for col in required_columns:
        if col not in df.columns:
            return False
    
    return True


# Sample data generator for demo
def generate_sample_data():
    """
    Generate realistic sample Maybank transaction data for demo purposes
    
    Returns:
        pandas DataFrame with sample transactions
    """
    sample_transactions = [
        {'date': '2024-11-01', 'description': 'STARBUCKS KLCC', 'amount': -18.50, 'balance': 4231.50},
        {'date': '2024-11-02', 'description': 'GRAB-RIDE JALAN AMPANG', 'amount': -12.00, 'balance': 4219.50},
        {'date': '2024-11-02', 'description': 'MCDONALD\'S PAVILION', 'amount': -23.90, 'balance': 4195.60},
        {'date': '2024-11-03', 'description': 'SHOPEE PURCHASE', 'amount': -156.00, 'balance': 4039.60},
        {'date': '2024-11-03', 'description': 'NETFLIX SUBSCRIPTION', 'amount': -49.00, 'balance': 3990.60},
        {'date': '2024-11-04', 'description': 'VILLAGE GROCER', 'amount': -85.30, 'balance': 3905.30},
        {'date': '2024-11-05', 'description': 'GRAB-FOOD DELIVERY', 'amount': -32.50, 'balance': 3872.80},
        {'date': '2024-11-05', 'description': 'SHELL PETROL STATION', 'amount': -95.00, 'balance': 3777.80},
        {'date': '2024-11-06', 'description': 'UNIQLO MID VALLEY', 'amount': -189.00, 'balance': 3588.80},
        {'date': '2024-11-07', 'description': 'STARBUCKS KLCC', 'amount': -21.50, 'balance': 3567.30},
        {'date': '2024-11-08', 'description': 'GRAB-RIDE BANGSAR', 'amount': -15.80, 'balance': 3551.50},
        {'date': '2024-11-09', 'description': 'FAMILY MART', 'amount': -12.50, 'balance': 3539.00},
        {'date': '2024-11-10', 'description': 'FOODPANDA ORDER', 'amount': -45.60, 'balance': 3493.40},
        {'date': '2024-11-11', 'description': 'LAZADA PURCHASE', 'amount': -220.00, 'balance': 3273.40},
        {'date': '2024-11-12', 'description': 'GSC CINEMAS', 'amount': -38.00, 'balance': 3235.40},
        {'date': '2024-11-13', 'description': 'SPOTIFY PREMIUM', 'amount': -17.90, 'balance': 3217.50},
        {'date': '2024-11-14', 'description': 'WATSON\'S PHARMACY', 'amount': -45.80, 'balance': 3171.70},
        {'date': '2024-11-15', 'description': 'TNB ELECTRICITY BILL', 'amount': -125.00, 'balance': 3046.70},
        {'date': '2024-11-15', 'description': 'GRAB-RIDE KLCC', 'amount': -18.50, 'balance': 3028.20},
        {'date': '2024-11-16', 'description': 'MAMAK RESTAURANT', 'amount': -15.00, 'balance': 3013.20},
        {'date': '2024-11-17', 'description': 'AEON JUSCO', 'amount': -156.40, 'balance': 2856.80},
        {'date': '2024-11-18', 'description': 'STARBUCKS PAVILION', 'amount': -19.90, 'balance': 2836.90},
        {'date': '2024-11-19', 'description': 'GRAB-FOOD DELIVERY', 'amount': -38.20, 'balance': 2798.70},
        {'date': '2024-11-20', 'description': 'SHELL PETROL STATION', 'amount': -100.00, 'balance': 2698.70},
        {'date': '2024-11-21', 'description': 'H&M SHOPPING', 'amount': -245.00, 'balance': 2453.70},
        {'date': '2024-11-22', 'description': 'CELCOM BILL', 'amount': -88.00, 'balance': 2365.70},
        {'date': '2024-11-23', 'description': 'MCDONALD\'S', 'amount': -28.50, 'balance': 2337.20},
        {'date': '2024-11-24', 'description': 'FOODPANDA ORDER', 'amount': -52.00, 'balance': 2285.20},
        {'date': '2024-11-25', 'description': 'GUARDIAN PHARMACY', 'amount': -32.40, 'balance': 2252.80},
        {'date': '2024-11-26', 'description': 'GRAB-RIDE BUKIT BINTANG', 'amount': -22.00, 'balance': 2230.80},
        {'date': '2024-11-27', 'description': 'STARBUCKS', 'amount': -17.50, 'balance': 2213.30},
        {'date': '2024-11-28', 'description': 'SHOPEE PURCHASE', 'amount': -89.00, 'balance': 2124.30},
        {'date': '2024-11-29', 'description': 'KFC DRIVE-THRU', 'amount': -34.90, 'balance': 2089.40},
        {'date': '2024-11-30', 'description': 'UNIFI INTERNET BILL', 'amount': -139.00, 'balance': 1950.40},
    ]
    
    df = pd.DataFrame(sample_transactions)
    
    # Process the same way as real data
    df['date'] = pd.to_datetime(df['date'])
    df['transaction_type'] = df['amount'].apply(lambda x: 'expense' if x < 0 else 'income')
    df['amount_abs'] = df['amount'].abs()
    df = df.sort_values('date', ascending=False)
    df.reset_index(drop=True, inplace=True)
    
    return df


if __name__ == "__main__":
    # Test the parser functions
    print("Parser module loaded successfully!")
    print("Available functions:")
    print("- parse_maybank_csv(file)")
    print("- parse_maybank_pdf(file)")
    print("- filter_expenses_only(df)")
    print("- get_transaction_summary(df)")
    print("- generate_sample_data()")