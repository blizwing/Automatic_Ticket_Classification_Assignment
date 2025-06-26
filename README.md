# 🧾 Automatic Ticket Classification Using Topic Modeling & Supervised Learning

## 📌 Problem Statement

Customer support centers receive thousands of tickets daily, often in unstructured textual form. To improve issue resolution efficiency, this project focuses on **automatically classifying customer complaints** into relevant categories based on their content.

### Objective:
Use **unsupervised topic modeling** to extract latent topics from complaint texts and then use this labeled data to train a **supervised classification model**.

---

## 📂 Dataset

- Format: `.json`
- Contains unstructured customer complaint texts
- No predefined labels

---

## 🔍 Approach

### 🧠 Topic Modeling (NMF)
Used **Non-negative Matrix Factorization (NMF)** to discover hidden topics in the dataset and classify tickets into 5 meaningful clusters:

1. **Credit card / Prepaid card**
2. **Bank account services**
3. **Theft / Dispute reporting**
4. **Mortgages / Loans**
5. **Others**

### ✅ Supervised Learning
Used the topic clusters to train and evaluate classification models:
- Logistic Regression
- Decision Tree
- Random Forest

---

## 🔁 Pipeline

1. Load and preprocess data from JSON
2. Clean and normalize complaint texts
3. Apply TF-IDF vectorization
4. Perform NMF topic modeling
5. Map each ticket to a topic cluster
6. Train classification models using the generated labels
7. Evaluate models using accuracy and classification reports

---

## 🛠️ Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- NLTK
- matplotlib, seaborn

---

## 📈 Results

- Achieved interpretable topic clusters using NMF
- Supervised models showed reasonable accuracy in classifying new tickets
- Project shows the power of combining unsupervised and supervised techniques for real-world text classification

---

## 📎 How to Run

1. Clone the repository or download the notebook
2. Run the notebook in a Python environment (Jupyter/Colab)
3. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn
   ```
4. Follow the pipeline step-by-step

---

📚 Lessons Learned

- Text preprocessing has a significant impact on model performance
- NMF is effective for interpretable topic modeling on customer support data
- Synthetic labeling using unsupervised methods can be useful in absence of labeled data
