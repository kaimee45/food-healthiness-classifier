# 🥗 Food Healthiness Classifier & Statistical Inference

**Statistical Inference Lab · Final Project**  
*Author:* Kaivalya Maske · AIDS (Artificial Intelligence and Data Science)

---

## 📌 Project Overview
An end-to-end Data Science and Statistical Inference project that analyzes nutritional profiles of fast-food items, computes evidence-based **Health Scores (0–6)**, and classifies foods into **Healthy** vs. **Unhealthy** using statistical techniques and Machine Learning algorithms (**Logistic Regression** and **Decision Tree**).

The project is deployed as an interactive **Streamlit Web Application** featuring live predictors, interactive dataset management, statistical inference tools, and exploratory data analysis.

---

## 🔬 Core Statistical Inference & ML Features

- **Nutritional Health Scoring:** Evaluates items based on 6 core nutritional criteria (Calories, Total Fat, Saturated Fat, Sugar, Sodium, Protein) following dietary standards (ICMR/WHO).
- **Outlier Handling (IQR Winsorization):** Detects and caps extreme outliers in Calories and Sodium to prevent regression weight distortion without dropping data.
- **Exploratory Visualizations:** Feature distributions, Boxplots, Scatter relationships (Calories vs Fat/Sugar), and Pearson correlation heatmaps.
- **Normal Distribution Comparison:** Compares empirical feature distributions against Gaussian theoretical PDF curves.
- **Central Limit Theorem (CLT) Simulation:** Simulates repeated sampling distributions ($n = 10, 30, 50$) showing convergence to a normal distribution.
- **Bootstrap Estimation:** Non-parametric resampling ($B = 3000$) generating 95% Confidence Intervals for population parameters.
- **Chi-Square ($\chi^2$) Test of Independence:** Evaluates the statistical association between Menu Categories and Healthiness ($p < 0.001$).
- **Machine Learning Classification:**
  - **Logistic Regression:** Scaled linear classifier with balanced class weights ($\approx 94\%$ accuracy).
  - **Decision Tree Classifier:** Explainable tree-based decision splits ($\approx 98\%$ accuracy).
- **Interactive Live Predictor:** Predicts health labels for existing menu items, external fast-food brands (**Burger King, KFC, Subway, Taco Bell, Domino's**), and custom foods via interactive sliders.

---

## 📂 Repository Structure

```text
├── app.py                                                      # Streamlit interactive web application
├── Menu.csv                                                    # Primary fast-food nutrition dataset (3,380+ items)
├── India_Menu.csv                                              # Original McDonald's India menu dataset
├── SI_Lab_Final_Project_Food_Healthiness_Classification_Final.ipynb  # Main academic lab project notebook
├── FoodHealthClassifier_Colab.ipynb                            # Google Colab notebook for cloud execution
├── requirements.txt                                            # Python dependencies
├── report.pdf                                                  # Formal project report
└── README.md                                                   # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Free)
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New App**, select this repository, set `app.py` as the main file path, and click **Deploy**!
