# Smart Expense Analyzer 💰

Financial analysis tool that categorizes bank transactions and provides spending insights using Machine Learning and LLM chatbot.

## 🎯 Features

- 📋 **Rule-Based Categorization** - Automatic expense classification using keyword matching
- 🤖 **ML Clustering Analysis** - K-Means algorithm groups spending patterns
- ⏱️ **ML Frequency Analysis** - Analyzes purchase frequency per category
- 💬 **LLM Financial Chatbot** - Conversational advice using Google Gemini API
- ✏️ **Manual Categorization** - Correct uncategorized transactions manually
- 📊 **Visual Dashboard** - Interactive charts and spending breakdown

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/marinakamal/smart-expense-analyzer.git
cd smart-expense-analyzer
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up API key:**
- Get a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- Create `.streamlit/secrets.toml` file:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

5. **Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure
```
smart-expense-analyzer/
├── app.py                    # Main Streamlit application
├── utils/
│   ├── parser.py            # CSV/PDF parsing functions
│   ├── categorizer.py       # Rule-based classification + ML analysis
│   └── llm_insights.py      # Gemini LLM integration
├── data/
│   └── sample_statement.csv # Sample data for demo
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 💻 Usage

1. **Upload Statement:** Upload your Maybank bank statement (CSV or PDF format)
2. **Enter Context:** Provide your monthly income and savings amount in sidebar
3. **View Analysis:** 
   - Category breakdown with charts
   - ML clustering patterns (automatically generated)
   - Purchase frequency predictions (automatically generated)
4. **Manual Corrections:** Categorize any uncategorized transactions
5. **Ask Questions:** Use the chatbot for personalized financial advice

### Supported Format
CSV with columns:
```
Date, Description, Amount, Balance
```

## 🛠️ Technologies

- **Frontend:** Streamlit
- **LLM:** Google Gemini API
- **Machine Learning:** scikit-learn (K-Means Clustering, Time Series Analysis)
- **Data Processing:** pandas, numpy
- **Visualization:** Plotly

## 📊 Demo

Try with the included `sample_bank_statement.csv` (50 sample transactions)

## 🎓 CAIE Project

This project is part of the Certified AI Engineer (CAIE) program by USAII, demonstrating:
- **LLM Functionality:** Google Gemini chatbot for financial advice
- **ML Components:** 
  - K-Means clustering for spending pattern analysis
  - Time series analysis for purchase frequency prediction
- **Real-world Use Case:** Personal finance management
- **Working Interface:** Streamlit web application

## 📝 License

MIT License - free to use for learning and portfolio purposes

## 👤 Author

**Nik Marina Kamal**
- LinkedIn: [linkedin.com/in/nikmarinakamal](https://www.linkedin.com/in/nikmarinakamal)
- CAIE Batch 3

---

**Made for CAIE Final Project**