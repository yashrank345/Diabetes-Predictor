# 🩺 Diabetes Predictor App  

An interactive **Streamlit web application** that helps users:  
- **Predict diabetes** based on medical parameters using a trained machine learning model.  
- **Visualize diabetes dataset insights** with interactive plots.  

---

## Features  

###  Diabetes Predictor  
- Input **medical parameters** such as Blood Pressure, BMI, Age, etc.  
- Get **instant predictions** (Positive / Negative) using a trained ML pipeline.  
- Displays health categories for **Blood Pressure** and **BMI** to guide inputs.  

###  Visualization  
Explore dataset insights through:  
- Dataset preview & statistics  
- Outcome distribution (Positive vs Negative)  
- Numeric feature distributions (histograms)  
- Correlation heatmap  
- Box plots comparing features by outcome  
- Scatter plot: Age vs Diabetes Pedigree Function  

---

## 📂 Project Structure 
```bash
├── Diabetes Predictor.ipynb   # Jupyter Notebook (model training & analysis)  
├── app.py                     # Streamlit app for prediction & visualization  
├── pipe.pkl                   # Trained ML pipeline  
├── df.pkl                     # Preprocessed dataset  
├── requirements.txt           # Python dependencies  
└── README.md 
```
---


## Installation  

### 1. Clone the repository  
```bash
git clone https://github.com/your-username/diabetes-predictor.git
cd diabetes-predictor 
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
---

## Usage  

1. Ensure that the trained model (`pipe.pkl`) and dataset (`df.pkl`) are in the project folder.  
2. Run the Streamlit app:  
   ```bash
   streamlit run app.py
3. Use the sidebar to switch between:

   Diabetes Predictor → Input parameters & get prediction

   Visualization → Explore dataset insights

---
## Model  

- Built with a **scikit-learn pipeline**.  
- Classification algorithms experimented with include:  
  - Logistic Regression  
  - Random Forest  
  - KNeighborsClassifier  
  - Support Vector Classifier (SVC)  
  - GaussianNB  
  - DecisionTreeClassifier  
  - StackingClassifier  
  - VotingClassifier  
  - GradientBoostingClassifier  
- Model trained on a **preprocessed dataset** and serialized using **pickle**.  

---

## Future Improvements  

- Add support for **live input fields** instead of only dropdowns.  
- Deploy on **Streamlit Cloud**.  
- Add model evaluation metrics such as:  
  - Accuracy  
  - Confusion Matrix  
  - Cross Validation Score  
  - Classification Report  
- Enable **CSV file upload** for batch predictions.  

---
## Author  

Developed by **Yash Rank** ✨  

---
## Deploy Link
```bash
https://diabetes-predictor-64gqpx95qnqdzqmtxtjabq.streamlit.app/
```



