# DS202W: Data Science for Social Scientists — Complete Course Handout

**LSE | Dr. Ghita Berrada | 2025-2026**

---

# WEEK 02: Foundations of Data Science & Linear Regression

---

## 1. Core Principles

- **"All models are wrong, but some are useful"** (George Box): Models are deliberate simplifications, not exact replicas of reality.
- Data alone do NOT determine a unique model — modelling requires human choices about structure, variables, and objectives.
- **Rule-based systems** use hand-coded expert rules (brittle, domain-specific); **Machine Learning** learns patterns from data (flexible, data-driven).

## 2. Supervised Learning Framework

$$Y = f(X) + \varepsilon$$

- $Y$: outcome (target, response, dependent variable)
- $X = (X_1, \dots, X_p)$: predictors (features, independent variables)
- $f$: unknown true relationship
- $\varepsilon$: irreducible noise (randomness no model can capture)

**Goal:** Estimate $\hat{f}$ from data such that $\hat{Y} = \hat{f}(X)$ generalises well to unseen data.

## 3. Linear Regression

### 3.1 Univariate Linear Regression

$$\hat{y} = \beta_0 + \beta_1 x$$

- $\beta_0$: intercept (predicted $y$ when $x = 0$)
- $\beta_1$: slope (change in $\hat{y}$ per unit increase in $x$)

### 3.2 Multivariate Linear Regression

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

Each $\beta_j$ is a **partial effect**: the association between $x_j$ and $y$, holding all other predictors constant.

### 3.3 Geometric Interpretation

The regression surface is a hyperplane in $(p+1)$-dimensional space. Each coefficient tilts the hyperplane in the direction of the corresponding predictor axis.

### 3.4 OLS (Ordinary Least Squares)

Minimises the sum of squared residuals:

$$\hat{\boldsymbol{\beta}}^{OLS} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j)^2$$

Closed-form solution:

$$\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$$

### 3.5 Five Assumptions of Linear Regression

| # | Assumption | Violation Consequence |
|---|---|---|
| 1 | **Linearity** — mean relationship is linear in parameters | Biased predictions, systematic errors |
| 2 | **Independence** — observations are independent | Underestimated standard errors |
| 3 | **Homoskedasticity** — constant error variance | Inefficient estimates, invalid inference |
| 4 | **No perfect multicollinearity** | Unstable coefficients, inflated standard errors |
| 5 | **Normality of errors** (for inference) | Invalid confidence intervals / p-values |

**Important:** Assumptions are about *errors*, not predictors. Linear regression does NOT require normal predictors.

## 4. Exploratory Data Analysis (EDA) — Four Dimensions

### 4.1 Variable Types
- Continuous (numeric) vs. categorical (nominal/ordinal)
- Check with `.dtypes`, `.info()`

### 4.2 Missingness
- Quantify: `.isnull().sum()`, `.isnull().mean()`
- Visualise: `msno.matrix()`, `msno.heatmap()`

### 4.3 Three Missingness Mechanisms

| Mechanism | Definition | Diagnostic | Treatment |
|---|---|---|---|
| **MCAR** | Missingness unrelated to any variable | Rare in practice | Dropping valid but wasteful |
| **MAR** | Missingness depends on *observed* variables | Most common; check patterns by group | Predictive imputation (KNN, tree-based) |
| **MNAR** | Missingness depends on the *unobserved value itself* | Hardest to detect | No standard imputation fully corrects |

### 4.4 Correlation Structure
```python
corr = df.corr()
corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
```
- Strong correlations → redundancy, multicollinearity risk
- Use VIF (Variance Inflation Factor) to quantify:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### 4.5 Distribution Shape (Skewness)
```python
skewness = df.skew(numeric_only=True).sort_values(ascending=False)
```
- |skew| > 1 → substantially skewed → consider scaling/transformation

## 5. The Leakage-Free Modelling Pipeline

**CRITICAL ORDERING — memorise this:**

1. **Drop missing outcomes** (never impute the target)
2. **Exclude** identifiers and variables with excessive missingness
3. **Train-test split** (e.g., 75/25 with `random_state`)
4. **Fit scaler on training data only** → transform both
5. **Fit imputer on scaled training data only** → transform both
6. **Fit model(s)** on training data
7. **Evaluate** on test data

### 5.1 Why Never Impute the Outcome
Imputing the outcome means training on fabricated labels — this breaks supervised learning logic and inflates performance metrics.

### 5.2 Standardisation (StandardScaler)

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{\sigma_j}$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit AND transform on TRAIN
X_test_scaled = scaler.transform(X_test)         # transform only on TEST
```

**Why mandatory:**
- Regularisation penalties act on coefficient magnitude → incomparable without scaling
- KNN imputation uses distances → dominated by large-scale features
- After scaling, coefficients represent change per 1 standard deviation increase

### 5.3 KNN Imputation

**Core idea:** "If two observations are similar on observed variables, they probably have similar values on the missing one."

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)
```

**Why KNN over mean/median imputation:**
- Mean/median collapses all missing values to one number → distorts distributions
- Mean/median ignores relationships between variables
- KNN preserves multivariate structure and distributional variance

**Why scaling MUST come before KNN:** KNN uses distance calculations; unscaled features with larger ranges dominate distances.

## 6. Evaluation Metrics (Regression)

### 6.1 MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- Average absolute prediction error, same units as outcome
- Robust to outliers; treats all errors equally

### 6.2 RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- Penalises large errors more heavily
- RMSE ≥ MAE always; large gap → outliers present

### 6.3 R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{Residual variance}}{\text{Total variance}}$$

- Share of variance explained
- **Increases mechanically with more predictors** (even irrelevant ones)
- In social science, moderate R² (0.2–0.4) is common
- Very high R² should raise suspicion of leakage

### 6.4 Diagnosing Overfitting/Underfitting

| Pattern | Train metrics | Test metrics | Diagnosis |
|---|---|---|---|
| Underfitting | Poor | Poor | Model too simple |
| Overfitting | Excellent | Poor | Model too complex |
| Good generalisation | Moderate | Similar to train | Appropriate complexity |

```python
def metrics(model, X, y):
    preds = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "R2": r2_score(y, preds)
    }
```

## 7. Residual Diagnostics

### Residuals vs. Fitted Values Plot
- **Want:** No pattern, constant spread, centred around zero
- **Red flags:** Funnel → heteroskedasticity; Curvature → non-linearity; Clusters → omitted variables

### Residual Distribution
- **Want:** Rough symmetry
- **Heavy tails:** Outliers present
- Normality matters for inference, not prediction

```python
residuals = y_test - model.predict(X_test)
sns.scatterplot(x=model.predict(X_test), y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
```

## 8. Regularisation

**Core idea:** OLS minimises Error only. Regularised regression minimises Error + Penalty.

### 8.1 Ridge (L2)

$$\hat{\boldsymbol{\beta}}^{Ridge} = \arg\min \left[\sum(y_i - \beta_0 - \sum x_{ij}\beta_j)^2 + \lambda\sum_{j=1}^{p}\beta_j^2\right]$$

- Shrinks coefficients smoothly toward zero
- **Never sets coefficients exactly to zero**
- Keeps all variables → good when all may contribute
- Handles multicollinearity well

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)  # alpha = λ
```

### 8.2 Lasso (L1)

$$\hat{\boldsymbol{\beta}}^{Lasso} = \arg\min \left[\sum(y_i - \beta_0 - \sum x_{ij}\beta_j)^2 + \lambda\sum_{j=1}^{p}|\beta_j|\right]$$

- Can set coefficients **exactly to zero** → performs variable selection
- Produces **sparse** models
- Unstable when predictors are highly correlated (may drop one arbitrarily)
- **Geometric intuition:** L1 constraint region has sharp corners (diamond shape); solution hits corners where some $\beta_j = 0$

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.05)
```

### 8.3 Elastic Net (L1 + L2)

$$\hat{\boldsymbol{\beta}}^{EN} = \arg\min \left[\sum(y_i - \beta_0 - \sum x_{ij}\beta_j)^2 + \lambda\left(\alpha\sum|\beta_j| + (1-\alpha)\sum\beta_j^2\right)\right]$$

- Two parameters: $\lambda$ (overall strength), $\alpha \in [0,1]$ (L1/L2 balance)
- $\alpha = 0$ → Ridge; $\alpha = 1$ → Lasso
- Combines stability (Ridge) with sparsity (Lasso)

```python
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha=0.05, l1_ratio=0.5)
```

### 8.4 Comparison Summary

| Model | Penalty | Variable Selection | Best When |
|---|---|---|---|
| OLS | None | No | Few predictors, low collinearity |
| Ridge | $\sum\beta_j^2$ | No | Multicollinearity, prediction focus |
| Lasso | $\sum|\beta_j|$ | Yes | Feature selection desired |
| Elastic Net | L1 + L2 | Yes | Correlated predictors + selection |

**Key takeaway:** Regularisation encodes modelling priorities — OLS prioritises unbiased estimation; Ridge prioritises stability; Lasso prioritises simplicity.

---

# WEEK 03: From Regression to Classification

---

## 1. Why Classification?

- **Regression** predicts quantities (continuous outcome)
- **Classification** predicts categories (discrete outcome)

## 2. Why Linear Regression Fails for Classification

Linear regression applied to binary outcomes produces predictions **outside [0,1]** — nonsensical as probabilities. Negative probabilities and probabilities > 1 are undefined.

## 3. Logistic Regression

### 3.1 The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$$

- **Always outputs [0,1]** → interpretable as probability
- S-shaped curve, symmetric around 0.5
- Nonlinear in probabilities, linear in predictors

### 3.2 Log-Odds (Logit) Formulation

$$\log\left(\frac{P(y=1)}{1-P(y=1)}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$$

- Left side: log-odds, ranges $(-\infty, +\infty)$
- Right side: linear combination of predictors

### 3.3 Coefficient Interpretation — Odds Ratios

Each coefficient $\beta_j$ represents change in **log-odds** per one-SD increase (after standardisation).

**To make interpretable:**
1. Exponentiate: $e^{\beta_j}$ = **odds ratio**
2. OR > 1 → odds increase; OR < 1 → odds decrease; OR = 1 → no association

**Example:** $\beta = -0.47 \Rightarrow e^{-0.47} \approx 0.63$ → one-SD increase in this predictor is associated with a **37% decrease in odds**.

**Important:** Odds ratios describe *relative* risk, not absolute probability differences. The same OR translates into different probability changes depending on the baseline probability.

```python
coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
```

### 3.4 Decision Threshold

- Default: predict class 1 if $P(y=1) \geq 0.5$
- The threshold is a **policy choice**, not a mathematical requirement
- Different thresholds yield different precision-recall tradeoffs

## 4. Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN (True Negative) | FP (False Positive, Type I) |
| **Actual Positive** | FN (False Negative, Type II) | TP (True Positive) |

**FP and FN are NOT symmetric** — costs differ by application:
- Medical screening: FN (missing disease) often worse
- Spam filtering: FP (losing important email) often worse
- Criminal justice: depends on values about liberty vs. safety

## 5. Classification Metrics

### 5.1 Accuracy (often misleading)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Danger:** A model predicting the majority class always achieves accuracy = majority proportion. **Never use accuracy alone when classes are imbalanced.**

### 5.2 Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

Average of recall per class. Immune to majority class dominance.

### 5.3 Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

"Of those we predicted positive, how many actually are?" Use when **false alarms are costly**.

### 5.4 Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

"Of all actual positives, how many did we catch?" Use when **missing cases is costly**.

### 5.5 F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Harmonic mean of precision and recall. Balances both; useful when you care about the positive class.

### 5.6 Precision-Recall Tradeoff

Lowering threshold → more positives predicted → higher recall, lower precision.
Raising threshold → fewer positives predicted → higher precision, lower recall.

## 6. The Perfect Separation Problem

### 6.1 What It Is

Perfect separation occurs when predictors completely separate classes with no overlap. Logistic regression coefficients diverge to ±∞.

### 6.2 Detection Signs
1. Convergence warnings
2. Extremely large coefficients
3. Predicted probabilities at exactly 0 or 1
4. Perfect training accuracy

### 6.3 Common Causes
1. **Data leakage** — outcome or near-outcome variables included as predictors
2. **Temporal leakage** — future information used to predict present
3. **Small samples** — too few observations per class
4. **Highly correlated predictors** — jointly encode the outcome

### 6.4 Three Failures
1. Statistical interpretation collapses (odds ratios explode)
2. Generalisation is compromised (brittle deterministic rule)
3. Policy interpretation becomes misleading (false certainty)

### 6.5 Solutions (ranked by priority)
1. **Remove or redesign problematic predictors** (best)
2. **Revisit data construction and timing**
3. **Use regularisation as stabilising safeguard** (not a cure)
4. **Collect more data** (when feasible)

---

# WEEK 04: Classification Evaluation, KNN, Cross-Validation & Fairness

---

## 1. Advanced Evaluation Metrics

### 1.1 ROC Curve and AUC

**ROC curve** plots True Positive Rate (Recall) vs. False Positive Rate across all thresholds.

- Diagonal = random classifier (AUC = 0.5)
- Perfect classifier = top-left corner (AUC = 1.0)
- AUC 0.7–0.8: acceptable; 0.8–0.9: good; >0.9: excellent (rare in social science)

**Probabilistic interpretation:** AUC = probability that model assigns higher score to a random positive than a random negative.

**Limitation:** Can be overly optimistic under class imbalance.

### 1.2 Precision-Recall Curve and Average Precision (AP)

- Baseline = proportion of positive class (not 0.5)
- More informative than ROC under imbalance
- AP summarises the PR curve: $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$

### 1.3 Calibration Curve

Compares predicted probabilities to actual frequencies in bins.

- **On diagonal** = well-calibrated (predicted 60% → 60% truly positive)
- **Above diagonal** = underconfident
- **Below diagonal** = overconfident

Logistic regression is usually well-calibrated by design (MLE). Other methods (RF, SVM, neural networks) often need recalibration.

## 2. Multi-Class Classification

### 2.1 Extension of Logistic Regression
- Multinomial logistic regression (auto-selected when >2 classes in sklearn)
- Outputs probability for each class
- Use **macro-averaging** for metrics: compute metric per class, then average

### 2.2 Encoding Rules
- **2 categories** → single binary indicator
- **>2 categories** → one-hot encoding with `drop_first=True` (reference category)
- **Never encode nominal categories as ordered numbers** unless truly ordinal

### 2.3 Multi-Class Metrics
```python
precision_score(y_true, y_pred, average="macro")
recall_score(y_true, y_pred, average="macro")
f1_score(y_true, y_pred, average="macro")
balanced_accuracy_score(y_true, y_pred)
roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
```

**Key insight:** Good ranking (ROC-AUC) does NOT guarantee good categorisation (F1). Continuous scores may rank well, but converting to labels introduces discretisation error.

## 3. K-Nearest Neighbours (KNN)

### 3.1 How It Works
1. Store all training data (no training phase — "lazy learning")
2. For new point, find K nearest neighbours using distance metric (default: Euclidean)
3. Majority vote among K neighbours → predicted class

$$d(x_i, x_j) = \sqrt{\sum_{k=1}^{p}(x_{ik} - x_{jk})^2}$$

### 3.2 Key Characteristics
- Non-parametric, instance-based
- No explicit decision boundary learned
- Can create **non-linear** decision boundaries
- Memory and prediction time grow with dataset size
- **Requires feature scaling** (distance-based)

### 3.3 Effect of K

| Small K (e.g., K=1) | Large K (e.g., K=200) |
|---|---|
| Flexible, jagged boundary | Smooth boundary |
| Low bias, high variance | High bias, low variance |
| Overfits easily | May underfit |
| K=1: training error = 0 | Approaches majority class |

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
```

## 4. Bias-Variance Tradeoff

$$\mathbb{E}[(Y - \hat{f}(X))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Bias:** Systematic error from overly simple assumptions
- **Variance:** Instability — how much predictions change with different training samples
- **Irreducible noise:** Randomness no model can eliminate

**Tradeoff:** Increasing model complexity → decreases bias but increases variance. Optimal complexity minimises total error.

## 5. Cross-Validation

### 5.1 K-Fold Cross-Validation

1. Divide training data into K folds (typically 5 or 10)
2. For each fold: train on K−1 folds, evaluate on held-out fold
3. Repeat K times → average validation performance

```
Fold 1: [Valid][Train][Train][Train][Train]
Fold 2: [Train][Valid][Train][Train][Train]
...
Fold 5: [Train][Train][Train][Train][Valid]
```

### 5.2 Stratified K-Fold

Preserves class proportions in each fold — **essential for classification**.

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv_results = cross_validate(
    model, X_train_scaled, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    return_train_score=True
)
```

### 5.3 Diagnosing Two Key Gaps

**Gap 1: CV-train − CV-validation**
- Large gap → overfitting / high variance
- Both low → underfitting / high bias

**Gap 2: CV-validation − Final test**
- Test much lower → hyperparameter tuning overfit to CV folds
- Approximately equal → CV estimate was reliable

**Ideal:** Mean Train ≥ Mean CV ≈ Final Test

### 5.4 GridSearchCV — Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 15, 20, 30, 40]}
grid = GridSearchCV(
    KNeighborsClassifier(), param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    return_train_score=True
)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
```

**Model selection overfitting:** Even if individual models don't overfit, *selecting* the best hyperparameter can exploit random fluctuations. **A completely untouched test set is ALWAYS required.**

## 6. Fairness — The Impossibility Theorem

### 6.1 Core Concept

Protected attributes (e.g., race) are used for **evaluation**, NOT prediction. Encode race to measure fairness, but exclude from feature matrix X.

### 6.2 Base Rate

Actual outcome frequency in each group. When base rates differ between groups, fairness tradeoffs become unavoidable.

### 6.3 Three Fairness Definitions

**1. Demographic Parity (Statistical Parity):**

$$P(\hat{Y}=1 | A=a) = P(\hat{Y}=1 | A=b)$$

Equal positive prediction rates across groups. Focuses on equality of treatment.

**2. Equalized Odds (Error Rate Balance):**

$$P(\hat{Y}=1 | Y=y, A=a) = P(\hat{Y}=1 | Y=y, A=b) \quad \forall y$$

Equal TPR AND FPR across groups. Focuses on equality of errors. (ProPublica's argument about COMPAS.)

**3. Predictive Parity (Calibration):**

$$P(Y=1 | \hat{Y}=1, A=a) = P(Y=1 | \hat{Y}=1, A=b)$$

Equal precision across groups. (Northpointe's argument about COMPAS.)

### 6.4 The Impossibility Theorem

**When base rates differ, it is IMPOSSIBLE to simultaneously satisfy all three fairness criteria.** This is a mathematical constraint, not a modelling failure.

- Equal precision ⇒ unequal error rates
- Equal error rates ⇒ unequal precision
- Equal prediction rates ⇒ unequal accuracy

### 6.5 Practical Implications

Fairness is a **values/ethical question**, not a technical one. Which criterion to prioritise depends on context:
- **Criminal justice:** FP (wrongful flagging) severe → equalized odds
- **Medical screening:** FN (missing disease) harmful → recall emphasis
- **Lending:** Regulatory requirements may mandate demographic parity

### 6.6 Additional Bias Sources

- **Proxy variables** (zip code correlated with race)
- **Historical bias** in training data
- **Measurement bias** (arrests ≠ actual crimes)
- **Feedback loops** (predictions influence future outcomes)

---

# WEEK 05: Non-Linear Algorithms

---

## 1. Why Non-Linear Models?

GLMs assume $g(\mathbb{E}[Y|X]) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$ — linear and additive in predictors. Each predictor has a constant marginal effect. Real-world phenomena (e.g., financial regime changes) exhibit non-linear relationships.

## 2. Feature Engineering for Time Series

### 2.1 Lagging Principle

Use only information available at time $t$ to predict time $t+1$. Prevents **data leakage**.

```python
variable.shift(1)  # Uses last period's value
```

### 2.2 Common Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| Return_lag1 | `pct_change(1).shift(1)` | Short-term momentum |
| Moving Average | `shift(1).rolling(k).mean()` | Smoothed trend |
| Interest Rate change | `Rate.diff().shift(1)` | Policy shocks |
| Inflation | `CPI.pct_change(12).shift(1)` | Year-over-year price changes |

### 2.3 Time-Aware Train/Test Split

For time series: preserve temporal ordering. Train on earlier years, test on later.

```python
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]
```

## 3. Baselines

**Always establish baselines:**
1. **Random guessing:** Expected balanced accuracy = 0.5
2. **Majority class:** Always predict most common outcome

Any model must outperform BOTH. Without baselines, you cannot assess whether a model extracts any signal.

## 4. Support Vector Machines (SVM)

### 4.1 Core Idea

Find the separating hyperplane that **maximises the geometric margin** between classes.

**Separating hyperplane:** $w^T x + b = 0$

**Classification rule:** If $w^T x + b > 0$ → class +1; if < 0 → class −1

**Margin width:** $\frac{2}{\|w\|}$

### 4.2 Support Vectors

Training observations on the margin boundary. **Only these points determine the hyperplane.** Moving non-support points does not change the solution.

### 4.3 Soft-Margin SVM (Parameter C)

Allows misclassifications but penalises them:
- **Large C** → heavily penalise violations → narrow margin → lower bias, higher variance
- **Small C** → tolerate violations → wider margin → higher bias, lower variance

### 4.4 The Kernel Trick (RBF)

$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

Implicitly maps data to high-dimensional space where a linear separator exists. The boundary is **curved in original space**, linear in transformed space.

```python
from sklearn.svm import SVC
svm_linear = SVC(kernel="linear", C=1.0)
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
```

**SVM requires feature scaling** (relies on dot products).

## 5. Decision Trees

### 5.1 Core Idea

Partition feature space into rectangular regions using binary splits. Local threshold rules, not global boundaries.

### 5.2 Algorithm

At each node: select feature $X_j$ and threshold $t$, split into $X_j \leq t$ and $X_j > t$, choose split maximising impurity reduction.

### 5.3 Impurity Measures

**Gini impurity:** $G = 1 - \sum_{k} p_k^2$

**Entropy:** $H = -\sum_{k} p_k \log p_k$

In practice, Gini and entropy produce similar trees.

### 5.4 Key Properties
- **Interpretable:** Explicit if-then rules
- **No scaling required** (unlike SVM, KNN)
- **For regression:** Replace impurity with MSE; terminal nodes output mean
- **Greedy:** Does NOT globally optimise future splits

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
```

## 6. Random Forests

### 6.1 Problem with Single Trees

High variance — small changes in data produce very different trees.

### 6.2 Two Variance-Reduction Mechanisms

1. **Bootstrap aggregation (bagging):** Each tree trained on a bootstrap sample
2. **Random feature selection:** Only $\sqrt{p}$ features considered at each split

$$\hat{f}_{RF}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=5)
```

### 6.3 Balanced Random Forest

Each tree trained on balanced subsample (equal class sizes). Forces model to learn minority structure. Better minority recall but more false positives.

```python
from imblearn.ensemble import BalancedRandomForestClassifier
```

## 7. Gradient Boosting

**Random Forest reduces variance. Boosting reduces bias.**

Trees built sequentially, each correcting residual errors:

$$F_m(x) = F_{m-1}(x) + \eta \cdot T_m(x)$$

**Key hyperparameters:**
- `n_estimators`: number of trees
- `learning_rate` ($\eta$): shrinkage factor (smaller = more stable)
- `max_depth`: tree complexity

**Modern libraries:** XGBoost (regularised), LightGBM (histogram-based, fast), CatBoost (native categorical handling).

```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=3)
```

## 8. Feature Selection

### 8.1 Three Approaches

1. **Filter methods:** Independent of model (mutual information, correlation, F-tests)
2. **Recursive Feature Elimination (RFE):** Fit model, remove weakest feature, repeat
3. **Sequential Feature Selection (SFS):** Forward (start empty, add) or backward (start full, remove)

```python
from sklearn.feature_selection import RFE, SequentialFeatureSelector
rfe = RFE(estimator=tree, n_features_to_select=5)
sfs = SequentialFeatureSelector(tree, n_features_to_select=5, direction="forward")
```

**Critical rule:** Feature selection MUST be performed on training data only, and recomputed inside CV folds.

## 9. Time-Aware Cross-Validation

Standard K-fold shuffles data → breaks temporal ordering.

**TimeSeriesSplit:** Forward expanding window, no shuffling.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

## 10. Feature Importance vs. SHAP

### 10.1 Random Forest Feature Importance (impurity-based)
- Total Gini reduction per feature, averaged across trees
- Shows magnitude only, NOT direction
- Biased toward continuous/high-cardinality variables

### 10.2 SHAP (SHapley Additive exPlanations)

$$f(x) = \phi_0 + \sum_j \phi_j$$

- Decomposes each prediction into feature contributions
- Shows magnitude AND direction
- Observation-specific
- Captures heterogeneous effects

```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test_df)
shap.summary_plot(shap_values[:, :, 1], X_test_df)
```

| Property | RF Importance | SHAP |
|---|---|---|
| Direction | No | Yes |
| Per-observation | No | Yes |
| Heterogeneity | No | Yes |

## 11. Support Vector Regression (SVR)

$$\min \frac{1}{2}\|w\|^2 + C\sum\max(0, |y_i - \hat{y}_i| - \varepsilon)$$

$\varepsilon$-tube: errors inside the tube are ignored. Outside the tube, penalised linearly. OLS penalises ALL errors quadratically; SVR ignores small errors and encourages flat functions.

---

# WEEK 07: Unsupervised Learning & Dimensionality Reduction

---

## 1. Supervised vs. Unsupervised Learning

| Supervised | Unsupervised |
|---|---|
| Labelled data (features + outcome) | No labels (features only) |
| Predict known answer | Discover unknown structure |
| Evaluation against ground truth | Open-ended, subjective evaluation |

### 1.1 Four Families of Unsupervised Learning

1. **Dimensionality Reduction** (this week): Compact representation preserving structure
2. **Clustering** (Week 8): Discover natural groups
3. **Anomaly Detection** (Week 9): Identify unusual observations
4. **Association Rules** (not covered): Co-occurrence patterns

## 2. The Curse of Dimensionality

### Problem 1: Visualisation Impossible
Pairwise plots needed = $\frac{p(p-1)}{2}$. With 50 variables → 1,225 plots, and still miss multi-variable interactions.

### Problem 2: Distances Become Unreliable
In high dimensions, all points become equidistant. Distance-based algorithms (clustering, KNN, anomaly detection) break down.

### Problem 3: Correlated Features Destabilise Models
Multicollinearity → unstable coefficients, large standard errors.

### Problem 4: Exponential Data Requirements
Reliable estimation needs ~10–20 observations per parameter. Data needed grows exponentially with dimensions.

**Solution:** Transform $p$ variables into $k \ll p$ new variables preserving essential structure.

## 3. Data Preprocessing for Dimensionality Reduction

### 3.1 Missingness Treatment

| Mechanism | Treatment |
|---|---|
| Structural zeros (policy rule) | Recode to zero |
| Lifecycle absences (entry/exit) | Average over active years |
| True unknowns | Defer to imputation |

**Never drop observations reflexively** — dropped observations may be the most analytically important.

### 3.2 Log Transform for Multiplicative Data

When quantities span orders of magnitude (e.g., emissions: 5,000 to 20,000,000):
- Use `log(1+x)` to handle genuine zeros
- Substantive justification: multiplicative processes → log-normal distribution
- Fixes within-variable distribution shape (standardisation alone insufficient)

### 3.3 Order: log-transform → impute → standardise

## 4. Principal Component Analysis (PCA)

### 4.1 What PCA Does

Finds new coordinate system aligned with **directions of maximum variance**. Does NOT change data shape — only rotates axes.

### 4.2 The Formula

$$\text{PC}_k = w_{k1}z_1 + w_{k2}z_2 + \cdots + w_{kp}z_p$$

Weights $w_{ki}$ = **loadings**. They tell how strongly each variable contributes to each component.

### 4.3 Key Properties

1. Components are **uncorrelated** (orthogonal)
2. **Ordered by variance explained**: PC1 > PC2 > PC3 > ...
3. As many PCs as input dimensions — **reduction occurs when you retain fewer**

### 4.4 How Many Components?

- **Scree plot / elbow rule:** Look for diminishing returns
- **Cumulative variance threshold:** Retain 80–90% of total variance
- **No universal rule** — depends on objective (visualisation, modelling, interpretation)
- **Variance explained ≠ importance:** A low-variance component may capture policy-relevant structure

### 4.5 Interpreting Loadings

Each loading tells the correlation between the original variable and the component. Large positive/negative loadings define what the component represents.

```python
pca = PCA(n_components=k)
pca.fit(X_scaled)
loadings = pd.DataFrame(pca.components_.T, index=feature_names,
                        columns=[f'PC{i+1}' for i in range(k)])
```

### 4.6 Critical Limitation

**PCA CANNOT handle categorical variables.** One-hot encoding before PCA uses wrong geometry:
1. Wrong distances (all categories equally dissimilar)
2. Cardinality bias (more categories → more weight)
3. Uninterpretable loadings

## 5. Multiple Correspondence Analysis (MCA)

### 5.1 Categorical Analogue of PCA

Uses **chi-square distance** instead of Euclidean:

$$d_{\chi^2}(i, i') = \sum_j \frac{1}{p_j}\left(\frac{x_{ij}}{x_{i\cdot}} - \frac{x_{i'j}}{x_{i'\cdot}}\right)^2$$

Rare categories contribute more to distance (appropriate for identifying structural outliers).

### 5.2 MCA Variance Explained

MCA eigenvalue percentages are low by design (3–4% per dimension). **Do NOT compare to PCA's figures.** What matters is whether dimensions are interpretable.

```python
import prince
mca = prince.MCA(n_components=5)
mca.fit(categorical_data)
```

## 6. Factor Analysis of Mixed Data (FAMD)

### 6.1 Why Needed

Most real datasets are mixed (numeric + categorical). Three naive approaches fail:
1. PCA on everything (wrong geometry for categoricals)
2. Separate PCA + MCA (loses cross-type relationships)
3. Concatenation (incommensurable distance metrics)

### 6.2 How FAMD Works

1. Standardises numeric variables (like PCA)
2. Normalises category indicators by $\sqrt{\text{marginal frequency}}$ (like MCA)
3. Single SVD on the combined matrix

**FAMD standardises internally — do NOT pass pre-standardised data.**

### 6.3 Method Selection Guide

| Data Type | Correct Method |
|---|---|
| All numeric | PCA |
| All categorical | MCA |
| Mixed (the realistic default) | **FAMD** |

```python
famd = prince.FAMD(n_components=5)
famd.fit(mixed_data)
```

## 7. UMAP (Uniform Manifold Approximation and Projection)

### 7.1 When Linear Methods Are Not Enough

PCA finds best *linear* summary. UMAP reveals non-linear structure (hard discontinuities, non-convex clusters, curved manifolds).

### 7.2 How UMAP Works

1. Build neighbourhood graph (K nearest neighbours in high-D)
2. Project to 2D preserving neighbourhood structure

### 7.3 Key Parameter: n_neighbors

- Small (5): Many tight micro-clusters, fragmented, artefacts likely
- Medium (30): Balance local/global structure
- Large (100): Smooth, global structure, may lose fine detail

**Always check stability** by varying `n_neighbors`. Only trust structure that persists across settings.

```python
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1)
embedding = reducer.fit_transform(X_scaled)
```

## 8. Autoencoders

### 8.1 What an Autoencoder Is

Neural network trained to **reconstruct its own input** through a bottleneck.

```
Input (p features) → Encoder → Bottleneck (k neurons, k << p) → Decoder → Output (p features)
```

The bottleneck forces compression. Bottleneck values = **latent representation**.

### 8.2 Architecture

```python
encoder = nn.Sequential(
    nn.Linear(INPUT_DIM, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, LATENT_DIM)
)
decoder = nn.Sequential(
    nn.Linear(LATENT_DIM, 16), nn.ReLU(),
    nn.Linear(16, 32), nn.ReLU(),
    nn.Linear(32, INPUT_DIM)  # No activation
)
```

### 8.3 Training

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    reconstruction = model(X_tensor)
    loss = criterion(reconstruction, X_tensor)
    loss.backward()
    optimizer.step()
```

### 8.4 Key Differences from PCA

| Property | PCA | Autoencoder |
|---|---|---|
| Ordered by variance | Yes | No |
| Orthogonal components | Yes | Not guaranteed |
| Interpretable loadings | Yes | No |
| Transformation | Linear | Non-linear |
| Interpretation | Algebraic (loadings) | Geometric (who is close) |

### 8.5 Reconstruction Error → Anomaly Detection

$$\text{Reconstruction Error}_i = \frac{1}{p}\sum_{j=1}^{p}(x_{ij} - \hat{x}_{ij})^2$$

High reconstruction error → observation doesn't fit learned structure → candidate anomaly. Foundation for Week 9.

## 9. Dimensionality Reduction in Supervised Pipelines

### 9.1 No-Leakage Principle

PCA/scaler/imputer must be **fit on training data only**. Use sklearn `Pipeline` to enforce automatically.

```python
from sklearn.pipeline import Pipeline
pca_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('classifier', LogisticRegression(class_weight='balanced'))
])
pca_pipe.fit(X_train, y_train)
```

For PyTorch autoencoders outside Pipeline, manage separation manually: fit encoder on X_train only, then extract features for both sets.

---

# APPENDIX: Quick Reference — Code Patterns

## A. Complete Regression Pipeline
```python
# 1. Drop missing outcomes
df_model = df.dropna(subset=["target"]).copy()
X = df_model.drop(columns=["target", "id_cols"])
y = df_model["target"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Scale (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Impute (fit on train only)
imputer = KNNImputer(n_neighbors=5)
X_train_imp = imputer.fit_transform(X_train_scaled)
X_test_imp = imputer.transform(X_test_scaled)

# 5. Fit models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
ols = LinearRegression().fit(X_train_imp, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_imp, y_train)
lasso = Lasso(alpha=0.05).fit(X_train_imp, y_train)

# 6. Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
for model, name in [(ols, "OLS"), (ridge, "Ridge"), (lasso, "Lasso")]:
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_imp)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_imp)))
    print(f"{name}: Train RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}")
```

## B. Complete Classification Pipeline
```python
# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import (balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score)

y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.3f}")
```

## C. Cross-Validation with Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 15, 20]}
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    return_train_score=True
)
grid.fit(X_train_scaled, y_train)
print(f"Best K: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
print(f"Test score: {grid.score(X_test_scaled, y_test):.3f}")
```

## D. PCA Pipeline
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Scree plot
plt.bar(range(1, 6), pca.explained_variance_ratio_ * 100)
plt.xlabel('Component')
plt.ylabel('Variance Explained (%)')

# Loadings
loadings = pd.DataFrame(pca.components_.T,
    index=feature_names, columns=[f'PC{i+1}' for i in range(5)])
```

## E. SHAP Analysis
```python
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test_df)
shap.summary_plot(shap_values[:, :, 1], X_test_df)  # For class 1
```

---

# APPENDIX: Key Formulas Summary

| Formula | Name | Context |
|---|---|---|
| $\hat{y} = \beta_0 + \sum \beta_j x_j$ | Linear Regression | Regression |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid | Classification |
| $\log\frac{P}{1-P} = \beta_0 + \sum \beta_j x_j$ | Logit | Logistic Regression |
| $e^{\beta_j}$ | Odds Ratio | Coefficient interpretation |
| $z = \frac{x - \bar{x}}{\sigma}$ | Standardisation | Preprocessing |
| MAE $= \frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Mean Absolute Error | Regression evaluation |
| RMSE $= \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Root Mean Squared Error | Regression evaluation |
| $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ | Coefficient of Determination | Regression evaluation |
| Accuracy $= \frac{TP+TN}{Total}$ | Accuracy | Classification (use cautiously) |
| Balanced Acc $= \frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$ | Balanced Accuracy | Classification (preferred) |
| Precision $= \frac{TP}{TP+FP}$ | Precision | Classification |
| Recall $= \frac{TP}{TP+FN}$ | Recall/Sensitivity | Classification |
| $F_1 = 2\frac{P \cdot R}{P + R}$ | F1 Score | Classification |
| $\lambda\sum\beta_j^2$ | L2 Penalty | Ridge |
| $\lambda\sum\|\beta_j\|$ | L1 Penalty | Lasso |
| $G = 1 - \sum p_k^2$ | Gini Impurity | Decision Trees |
| $H = -\sum p_k \log p_k$ | Entropy | Decision Trees |
| $K(x_i,x_j) = \exp(-\gamma\|x_i-x_j\|^2)$ | RBF Kernel | SVM |
| Margin $= \frac{2}{\|w\|}$ | SVM Margin Width | SVM |
| $\text{PC}_k = \sum w_{kj} z_j$ | Principal Component | PCA |
| Reconstruction Error $= \frac{1}{p}\sum(x_j - \hat{x}_j)^2$ | MSE | Autoencoders |

---

*End of DS202W Complete Handout*
