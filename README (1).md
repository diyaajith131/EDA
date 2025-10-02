# 📊 Exploratory Data Analysis - Task 2

## 📌 Objective
Perform **Exploratory Data Analysis (EDA)** on the Titanic dataset (or any dataset) to understand data using **statistics and visualizations**.

## 🛠 Tools Used
- Python  
- Pandas  
- Matplotlib  
- Seaborn  

## 📂 Steps Performed
1. **Data Loading** – Imported Titanic dataset.  
2. **Summary Statistics** – Mean, median, std, min, max, etc.  
3. **Missing Value Analysis** – Checked null values.  
4. **Visualizations**:  
   - Histograms (distribution of numerical features)  
   - Boxplots (outlier detection)  
   - Correlation heatmap (relationships between features)  
   - Pairplots (feature relationships)  

## 📈 Outputs
All plots are saved in the **outputs/** folder:  
- `histograms.png`  
- `boxplots.png`  
- `correlation_heatmap.png`  
- `pairplot.png`  

---

## ❓ Interview Questions & Answers

### 1. What is the purpose of EDA?  
EDA helps to understand the dataset’s structure, detect patterns, identify anomalies, and decide the right preprocessing & modeling techniques.

### 2. How do boxplots help in understanding a dataset?  
Boxplots show the **spread, median, quartiles, and outliers** of numerical data, making it easier to detect skewness and extreme values.

### 3. What is correlation and why is it useful?  
Correlation measures the **linear relationship** between variables. It helps identify redundant features and select important ones.

### 4. How do you detect skewness in data?  
By checking **histograms, skewness value (df.skew())**, or **boxplots**.

### 5. What is multicollinearity?  
When two or more independent variables are **highly correlated**, making it hard to estimate their effects accurately in models.

### 6. What tools do you use for EDA?  
Common tools: **Pandas, Matplotlib, Seaborn, Plotly, Power BI, Excel**.

### 7. Can you explain a time when EDA helped you find a problem?  
Example: In the Titanic dataset, EDA revealed missing **Age** values and outliers in **Fare**, which needed treatment before modeling.

### 8. What is the role of visualization in ML?  
Visualizations simplify data understanding, highlight hidden patterns, and guide **feature engineering** for better ML performance.

---

## 📌 Dataset
- [Titanic Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

---

## 🚀 How to Run
```bash
git clone <your-repo-link>
cd EDA-Task2
pip install -r requirements.txt
python eda_task2.py
```
