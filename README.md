📄 AI Resume Screening System
🔍 Overview

The AI Resume Screening System is an intelligent application that automates the process of comparing multiple resumes with a job description. It ranks candidates based on how closely their profiles match the job requirements, helping recruiters make faster and more accurate hiring decisions.

💡 Problem Statement

Recruiters often deal with a large number of resumes for a single job opening. Manually reviewing each CV is:

Time-consuming
Inefficient
Prone to human bias
🚀 Solution

This project uses Natural Language Processing (NLP) and machine learning techniques to:

Analyze resumes and job descriptions
Measure similarity between them
Rank candidates based on relevance
⚙️ Features
📂 Upload multiple resumes (PDF/DOCX)
📝 Add job description
🔎 Automatic text extraction and preprocessing
📊 Similarity scoring using NLP techniques
🏆 Ranking of resumes based on job relevance
🌐 Simple interface using Streamlit
🧠 Technologies Used
Python
Streamlit (for frontend UI)
Flask (optional backend handling)
PyPDF2 (PDF text extraction)
python-docx (DOCX parsing)
Scikit-learn (TF-IDF, Cosine Similarity)
🏗️ Project Setup

Follow these steps to run the project locally:

1️⃣ Clone the Repository
git clone https://github.com/your-username/resume-screening.git
cd resume-screening
2️⃣ Create Virtual Environment
python3 -m venv .venv
3️⃣ Activate Virtual Environment
On macOS/Linux:
source .venv/bin/activate
On Windows:
.venv\Scripts\activate
4️⃣ Install Dependencies
pip install --upgrade pip
pip install streamlit flask PyPDF2 python-docx scikit-learn
5️⃣ Run the Application
streamlit run streamlit_app.py
📊 How It Works
Upload multiple resumes
Enter the job description
System extracts text from resumes
Applies preprocessing (tokenization, cleaning, etc.)
Converts text into vectors using TF-IDF
Calculates similarity using cosine similarity
Displays ranked resumes
📈 Example Output
Resume A → 90% match
Resume B → 75% match
Resume C → 60% match
🎯 Use Cases
HR recruitment automation
Resume filtering systems
Applicant Tracking Systems (ATS)
Hiring platforms
🔮 Future Improvements
Integration with advanced NLP models (e.g., BERT)
Web-based dashboard with database support
Real-time resume parsing
Skill gap analysis
API integration for job portals
🤝 Contributing

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

📌 Author
Bisma Riaz
