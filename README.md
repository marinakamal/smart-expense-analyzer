# Smart Expense Analyzer 💰

AI-powered financial assistant that analyzes bank statements and provides personalized savings insights.

## 🎯 Features

- 🤖 **AI Expense Categorization** - Automatic classification using HuggingFace transformers
- 💡 **Personalized Insights** - LLM-powered financial advice using Google Gemini
- 💬 **Interactive Chatbot** - Ask questions about your spending
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
│   ├── categorizer.py       # HuggingFace classification
│   └── llm_insights.py      # Gemini LLM integration
├── data/
│   └── sample_statement.csv # Sample data for demo
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 💻 Usage

1. **Upload Statement:** Upload your Maybank bank statement (CSV or PDF format)
2. **Enter Context:** Provide your monthly income and savings amount
3. **Get Insights:** Receive AI-powered analysis and recommendations
4. **Ask Questions:** Use the chatbot for personalized advice

### Supported Format
```
Date | Description | Amount
```

## 🛠️ Technologies

- **Frontend:** Streamlit
- **AI/ML:** HuggingFace Transformers, Google Gemini API
- **Data Processing:** pandas, pdfplumber
- **Visualization:** Plotly

## 📊 Demo

Try the live demo: [Coming Soon]

## 🎓 CAIE Project

This project is part of the Certified AI Engineer (CAIE) program by USAII, demonstrating:
- LLM functionality (Google Gemini)
- Traditional AI component (Transformer-based classification)
- Real-world use case (Personal finance management)
- Full-stack deployment

## 📝 License

MIT License - feel free to use for learning and portfolio purposes!

## 👤 Author

**Nik Marina Binti Nik Ahmad Kamal**
- Email: marinakamal@gmail.com
- Batch: August 2025 / Batch 3

---

**Made for CAIE Final Project**