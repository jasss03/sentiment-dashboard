# 📊 Sentiment Analysis Dashboard

A simple and interactive Sentiment Analysis tool built with **Streamlit**, using **TextBlob** for Natural Language Processing (NLP). This app allows users to analyze sentiment from individual text inputs as well as from uploaded CSV/Excel files.

---

## 🚀 Features

- 🔍 Analyze sentiment of individual text snippets.
- 🧼 Clean text (remove extra spaces, punctuation, numbers, and stopwords).
- 📤 Upload CSV or Excel files to analyze sentiment in bulk.
- 📥 Download analyzed results as a CSV file.
- 📊 Outputs include:
  - **Polarity**: How positive or negative a sentence is (scale: -1 to +1)
  - **Subjectivity**: How subjective or opinion-based the text is (scale: 0 to 1)
  - **Categorical Analysis**: Positive, Neutral, Negative

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/)
- [TextBlob](https://textblob.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)
- [clean-text](https://pypi.org/project/clean-text/)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
