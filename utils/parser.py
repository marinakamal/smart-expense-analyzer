"""
parser.py - Bank Statement Parsing Utilities
Handles CSV and PDF parsing for Maybank statements with flexible column detection
"""

import pandas as pd
import pdfplumber
import streamlit as st
from io import StringIO

def detect_columns(df):
    """
    Intelligently detect which columns represent date, description, and amount
    
    Args:
        df: DataFrame with original column names
        
    Returns:
        dict: Mapping of standard names to actual column names
    """
    columns_lower = [col.lower() for col in df.columns]
    mapping = {}
    
    # Detect DATE column (first one containing "date")
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break  # Take the FIRST date column
    mapping['date'] = date_col
    
    # Detect DESCRIPTION column (any containing "description", "desc", "transaction", "particulars")
    desc_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['description', 'desc', 'particular', 'transaction', 'detail']):
            desc_col = col
            break
    mapping['description'] = desc_col
    
    # Detect AMOUNT column (containing "amount", "debit", "credit", "value")
    amount_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['amount', 'debit', 'credit', 'value', 'sum']):
            amount_col = col
            break
    mapping['amount'] = amount_col
    
    # Detect BALANCE column (optional - containing "balance")
    balance_col = None
    for col in df.columns:
        if 'balance' in col.lower():
            balance_col = col
            break
    mapping['balance'] = balance_col
    
    return mapping


def parse_maybank_csv(file):
    """
    Parse bank statement CSV with flexible column detection
    
    Accepts any format as long as it has columns containing:
    - "date" (any column with "date" in name)
    - "description" or "transaction" (description of transaction)
    - "amount" (transaction amount)
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        pandas DataFrame with columns: date, description, amount, transaction_type, amount_abs
    """
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Detect columns intelligently
        col_mapping = detect_columns(df)
        
        # Validate required columns were found
        if not col_mapping['date']:
            st.error("❌ Could not find a DATE column. Please ensure your CSV has a column with 'date' in the name.")
            return None
        
        if not col_mapping['description']:
            st.error("❌ Could not find a DESCRIPTION column. Please ensure your CSV has a column with 'description' or 'transaction' in the name.")
            return None
        
        if not col_mapping['amount']:
            st.error("❌ Could not find an AMOUNT column. Please ensure your CSV has a column with 'amount', 'debit', or 'credit' in the name.")
            return None
        
        # Create new dataframe with standardized columns
        df_clean = pd.DataFrame()
        df_clean['date'] = df[col_mapping['date']]
        df_clean['description'] = df[col_mapping['description']]
        df_clean['amount'] = df[col_mapping['amount']]
        
        # Add balance if available
        if col_mapping['balance']:
            df_clean['balance'] = df[col_mapping['balance']]
        else:
            df_clean['balance'] = None
        
        # Clean amount column (remove commas, RM, currency symbols, CR suffix)
        df_clean['amount'] = df_clean['amount'].astype(str).str.strip()
        
        # Remove CR, commas, RM, and currency symbols
        df_clean['amount'] = df_clean['amount'].str.replace('CR', '', case=False).str.replace(',', '').str.replace('RM', '').str.replace('$', '').str.strip()
        df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
        
        # Make all amounts negative (expenses)
        df_clean['amount'] = -df_clean['amount'].abs()
        
        # Determine transaction type - exclude payments based on description
        df_clean['transaction_type'] = df_clean['description'].apply(
            lambda x: 'income' if 'PYMT' in str(x).upper() or 'PAYMENT' in str(x).upper() else 'expense'
        )
        
        # Make all amounts positive for easier display
        df_clean['amount_abs'] = df_clean['amount'].abs()
        
        # Filter out payments based on transaction_type (already set by description check above)
        df_clean = df_clean[df_clean['transaction_type'] == 'expense'].copy()
        
        # Check if any expenses remain
        if df_clean.empty:
            st.warning("⚠️ No expenses found. All transactions appear to be payments.")
            return None
        
        # Filter out any rows with missing data
        df_clean = df_clean.dropna(subset=['description', 'amount'])
        
        # Convert date to datetime
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        # Drop rows with invalid dates
        df_clean = df_clean.dropna(subset=['date'])
        
        # Sort by date (most recent first)
        df_clean = df_clean.sort_values('date', ascending=False)
        
        # Reset index
        df_clean.reset_index(drop=True, inplace=True)
        
        # Show success message with detected columns
        st.success(f"✅ Detected columns: Date='{col_mapping['date']}', Description='{col_mapping['description']}', Amount='{col_mapping['amount']}'")
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        st.error("Please make sure your CSV has columns containing: 'Date', 'Description', and 'Amount'")
        return None


def parse_maybank_pdf(file):
    """
    Parse bank statement PDF with flexible column detection
    
    Extracts tables and identifies columns containing:
    - "date" (any column with date-like values)
    - "description" (transaction descriptions)
    - "amount" (transaction amounts)
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        pandas DataFrame with columns: date, description, amount, transaction_type, amount_abs
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
                    
                    # First row is usually headers
                    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                    
                    # SPECIAL HANDLING: Check if data is in one row with newline separators
                    # (Common in Maybank credit card statements)
                    # Check for actual newline character (not string '\n')
                    if len(table) == 2 and '\n' in table[1][0]:  # Check the actual cell content
                        # Split the single row by newlines
                        data_row = table[1]
                        split_data = []
                        
                        # Split each column by newlines
                        max_rows = max(len(col.split('\n')) if col else 0 for col in data_row)
                        
                        for i in range(max_rows):
                            row_data = []
                            for col in data_row:
                                col_values = col.split('\n') if col else []
                                row_data.append(col_values[i] if i < len(col_values) else "")
                            split_data.append(row_data)
                        
                        # Create dataframe from split data
                        temp_df = pd.DataFrame(split_data, columns=headers)
                    else:
                        # Normal multi-row table
                        temp_df = pd.DataFrame(table[1:], columns=headers)
                    
                    # Detect columns in this table
                    col_mapping = detect_columns(temp_df)
                    
                    # Skip if essential columns not found
                    if not col_mapping['date'] or not col_mapping['description'] or not col_mapping['amount']:
                        continue
                    
                    # Extract data from each row
                    for _, row in temp_df.iterrows():
                        try:
                            date_val = row[col_mapping['date']] if col_mapping['date'] else ""
                            desc_val = row[col_mapping['description']] if col_mapping['description'] else ""
                            amount_val = row[col_mapping['amount']] if col_mapping['amount'] else "0"
                            balance_val = row[col_mapping['balance']] if col_mapping['balance'] else ""
                            
                            # Skip rows with empty description
                            if not desc_val or str(desc_val).strip() == "" or str(desc_val).strip().lower() in ['nan', 'none']:
                                continue
                            
                            # Clean amount (remove currency symbols, commas, CR)
                            amount_clean = str(amount_val).strip()
                            
                            # Remove CR, RM, $, commas
                            amount_clean = amount_clean.replace('CR', '').replace('cr', '').replace('RM', '').replace('$', '').replace(',', '').strip()
                            
                            # Skip if amount is not a number
                            try:
                                amount_float = float(amount_clean)
                            except:
                                continue
                            
                            # Make all amounts negative (expenses)
                            amount_float = -abs(amount_float)
                            
                            # Skip zero amounts
                            if amount_float == 0:
                                continue
                            
                            transactions.append({
                                'date': str(date_val).strip(),
                                'description': str(desc_val).strip(),
                                'amount': amount_float,
                                'balance': str(balance_val).strip() if balance_val else None
                            })
                            
                        except Exception as row_error:
                            # Skip problematic rows
                            continue
        
        if not transactions:
            st.error("No transactions found in PDF. Please check the file format.")
            st.info("💡 Tip: Make sure your PDF has a table with columns for Date, Description, and Amount")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Determine transaction type - exclude payments based on description
        df['transaction_type'] = df['description'].apply(
            lambda x: 'income' if 'PYMT' in str(x).upper() or 'PAYMENT' in str(x).upper() else 'expense'
        )
        
        # Make all amounts positive for display
        df['amount_abs'] = df['amount'].abs()
        
        # Filter out payments based on transaction_type (already set by description check above)
        df = df[df['transaction_type'] == 'expense'].copy()
        
        # Check if any expenses remain
        if df.empty:
            st.warning("⚠️ No expenses found. All transactions appear to be payments.")
            return None
        
        # Sort by date (most recent first)
        df = df.sort_values('date', ascending=False)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        st.success(f"✅ Extracted {len(df)} transactions from PDF")
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        st.error("Please make sure your PDF is a valid bank statement with a table format.")
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
        'average_transaction': expenses['amount_abs'].mean() if len(expenses) > 0 else 0,
        'largest_expense': expenses['amount_abs'].max() if len(expenses) > 0 else 0,
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