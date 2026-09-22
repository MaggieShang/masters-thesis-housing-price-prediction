import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm




###Part3 Data and Preprocessing###

# read data
df = pd.read_csv("house_price.csv")

#3.1 Descriptive Statistics
cols=["SalePrice","GrLivArea","TotalBsmtSF","1stFlrSF","LotArea","YearBuilt","OverallQual","OverallCond","GarageArea","TotRmsAbvGrd"]
desc=df[cols].describe().T[['count','mean','std','min','50%','max']].round(2)
desc.columns=['Count','Mean','StdDev','Min','Median','Max']
print(desc)
desc.to_csv("table_3_1.csv")



###part 3.2###


# =================================================================
# 3.0 Missing Value Diagnostics Plot
# =================================================================
# Calculate percentage of missing values
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)

# Plot horizontal bar chart
plt.figure(figsize=(10, 8))
bars = plt.barh(missing_pct.index, missing_pct.values, color='steelblue', alpha=0.8)

# Add percentage labels to the right of the bars
for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1f}%',
             va='center', ha='left', fontsize=9)

plt.xlabel('Percentage of Missing Values (%)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Features with Missing Values in the Ames Dataset', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlim(0, 110) # Leave space for labels
plt.tight_layout()

# plt.savefig("figure_3_missing.png", dpi=300)

plt.show()
# 3.1 outlier-plot of "GrLivArea" and "SalePrice"

# Identify outliers
outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]

# Plot all data points
plt.scatter(df["GrLivArea"], df["SalePrice"], color='blue', alpha=0.3, label='Normal data')

# Highlight outliers
plt.scatter(outliers["GrLivArea"], outliers["SalePrice"], color='red', label='Outliers')

# Get axis limits
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

# Draw partial boundary lines
plt.plot([4000, 4000], [ymin, 300000],
         linestyle='--', color='black', linewidth=1.5, alpha=0.6)

plt.plot([4000, xmax], [300000, 300000],
         linestyle='--', color='black', linewidth=1.5, alpha=0.6)

# Labels and title
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

# Add legend
plt.legend()

plt.show()


# 3.2.1 Show why log transformation for salesprice is needed

plt.figure(figsize=(10, 4))

# Plot original SalePrice distribution
plt.subplot(1, 2, 1)
plt.hist(df["SalePrice"], bins=30)
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.title("(a) Original Distribution")

# Plot log-transformed SalePrice distribution
plt.subplot(1, 2, 2)
plt.hist(np.log1p(df["SalePrice"]), bins=30)
plt.xlabel("log(SalePrice)")
plt.ylabel("Frequency")
plt.title("(b) Log-Transformed Distribution")

plt.tight_layout()
plt.show()


# 3.2.2 show why "LotFrontage" imputed using median
plt.hist(df["LotFrontage"], bins=30)
plt.show()


# =================================================================
# 3.2.2 Correlation Heatmap
# =================================================================
# Select the top 10 numerical features most correlated with SalePrice
corrmat = df.corr(numeric_only=True)
top_corr_features = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

# Create the correlation matrix for these top features
cm = np.corrcoef(df[top_corr_features].values.T)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=top_corr_features.values,
                 xticklabels=top_corr_features.values, cmap='coolwarm', alpha=0.9)
plt.title('Pearson Correlation Heatmap of Top Features', fontsize=14)
plt.tight_layout()


plt.savefig("figure_3_corr.png", dpi=300)
plt.show()

# =================================================================
# 3.2.4 Categorical Impact - Boxplots
# =================================================================
plt.figure(figsize=(12, 10))

# (a) Overall Quality vs SalePrice (Original Scale)
plt.subplot(2, 1, 1)
sns.boxplot(x='OverallQual', y='SalePrice', data=df, palette='viridis')
plt.title('SalePrice Distribution by Overall Quality', fontsize=14)
plt.xlabel('Overall Quality (1-10)', fontsize=12)
plt.ylabel('SalePrice ($)', fontsize=12)

# (b) Neighborhood vs SalePrice (Original Scale)
plt.subplot(2, 1, 2)
neighborhood_order = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index
sns.boxplot(x='Neighborhood', y='SalePrice', data=df, order=neighborhood_order, palette='magma')
plt.xticks(rotation=45)
plt.title('SalePrice Distribution by Neighborhood (Sorted by Median)', fontsize=14)
plt.xlabel('Neighborhood', fontsize=12)
plt.ylabel('SalePrice ($)', fontsize=12)

plt.tight_layout()
plt.savefig("figure_3_boxplot.png", dpi=300)
plt.show()

###part 3.3###

# 1. Remove non-informative identifier
df.drop(columns=["Id"], inplace=True)

# 2. Remove extreme outliers based on domain knowledge
df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))].copy()

# 3. Apply log transformation to target variable to reduce skewness
df["SalePrice"] = np.log1p(df["SalePrice"])

# 4. Fill categorical variables where NA means "feature not present"
none_cols = [
    "Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]

for col in none_cols:
    if col in df.columns:
        df[col] = df[col].fillna("None")

# 5. Impute missing numerical values using median (robust to outliers)
median_cols = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]
for col in median_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 6. Impute categorical variables with few missing values using mode
mode_cols = ["Electrical", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
             "KitchenQual", "Functional", "SaleType"]
for col in mode_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# 7. Replace missing values with zero for structural features (absence of feature)
zero_cols = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea"
]
for col in zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 8. Feature engineering: create more meaningful variables
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# 9. Verify that no missing values remain
missing_after = df.isnull().sum()
print("Remaining missing values:")
print(missing_after[missing_after > 0])

# 10. Split dataset into features (X) and target variable (y)
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# 11. Convert categorical variables to numeric using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# 12. Convert encoded features to float
X_encoded = X_encoded.astype(float)

print("Processed X shape:", X_encoded.shape)
print("Target y shape:", y.shape)



### Part 4&5 Results ###

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# =================================================================
# Hyperparameter Tuning on Training Set Only
# =================================================================

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Grid Search
rf_param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [6, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=3
)

rf_grid.fit(X_train, y_train)


# Neural Network Grid Search with scaling inside pipeline
nn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42
    ))
])

nn_param_grid = {
    "mlp__hidden_layer_sizes": [
        (32,),
        (64,),
        (128,),
        (128, 64),
        (128, 64, 32)
    ],
    "mlp__learning_rate_init": [0.001, 0.0005, 0.0001],
    "mlp__alpha": [0.001, 0.01],
}

nn_grid = GridSearchCV(
    estimator=nn_pipeline,
    param_grid=nn_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=3
)

nn_grid.fit(X_train, y_train)


print("Best Random Forest parameters:")
print(rf_grid.best_params_)
print("Best RF CV RMSE:", -rf_grid.best_score_)

print("Best Neural Network parameters:")
print(nn_grid.best_params_)
print("Best NN CV RMSE:", -nn_grid.best_score_)

#=================================================================
# 5-fold Cross-Validation for Tuned Models
#=================================================================

rf_cv_model = rf_grid.best_estimator_
nn_cv_model = nn_grid.best_estimator_

rf_cv_scores = cross_validate(
    rf_cv_model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "R2": "r2"
    },
    n_jobs=-1
)

nn_cv_scores = cross_validate(
    nn_cv_model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "R2": "r2"
    },
    n_jobs=-1
)

def summarize_cv(scores, model_name):
    return {
        "Model": model_name,
        "CV RMSE Mean": -scores["test_RMSE"].mean(),
        "CV RMSE Std": scores["test_RMSE"].std(),
        "CV MAE Mean": -scores["test_MAE"].mean(),
        "CV MAE Std": scores["test_MAE"].std(),
        "CV R2 Mean": scores["test_R2"].mean(),
        "CV R2 Std": scores["test_R2"].std()
    }

cv_results_df = pd.DataFrame([
    summarize_cv(rf_cv_scores, "Tuned Random Forest"),
    summarize_cv(nn_cv_scores, "Tuned Neural Network")
])

print("\n5-Fold Cross-Validation Results on Training Set:")
print(cv_results_df)

cv_results_df.to_csv("table_5_cv_results.csv", index=False)

#================================================================
# Train Final Models Using Best Parameters + Baseline Models
#================================================================

# -----------------------------
# Tuned Random Forest
# -----------------------------
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# -----------------------------
# Tuned Neural Network
# -----------------------------
nn_model = nn_grid.best_estimator_
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

# Extract scaler for later SHAP analysis
scaler = nn_model.named_steps["scaler"]
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Baseline Model 1: Linear Regression
# -----------------------------
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# -----------------------------
# Baseline Model 2: Ridge Regression
# -----------------------------
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# -----------------------------
# Baseline Model 3: XGBoost
# -----------------------------
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

#================================================================
# Evaluation Function
#================================================================

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Results")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

#================================================================
# Final Model Evaluation on Hold-Out Test Set
#================================================================

results_linear = evaluate_model(y_test, y_pred_linear, "Linear Regression")
results_ridge = evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
results_rf = evaluate_model(y_test, y_pred_rf, "Tuned Random Forest")
results_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")
results_nn = evaluate_model(y_test, y_pred_nn, "Tuned Neural Network")

results_df = pd.DataFrame([
    results_linear,
    results_ridge,
    results_rf,
    results_xgb,
    results_nn
])

print("\nFinal model comparison on hold-out test set:")
print(results_df)

results_df.to_csv("table_5_model_comparison.csv", index=False)



# =================================================================
# Neural Network Performance Diagnosis
# =================================================================
# This part explains why the NN failed (Diagnosis 1 & 2)

# Plotting the Learning Curve to check for overfitting or convergence issues
plt.figure(figsize=(10, 6))

plt.plot(nn_model.named_steps["mlp"].loss_curve_)

plt.xlabel('Iterations')

plt.ylabel('Training Loss (MSE)')

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

# =================================================================
# Proceed to Interpretability (SHAP)
# =================================================================

# Random Forest SHAP
import shap
import matplotlib.pyplot as plt

# Create SHAP explainer for Random Forest
explainer_rf = shap.Explainer(rf_model, X_train)

# Compute SHAP values on test set
shap_values_rf = explainer_rf(X_test)

# Summary plot
shap.summary_plot(shap_values_rf, X_test, show=False)
plt.tight_layout()
plt.show()

# Mean absolute SHAP values
shap_importance_rf = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanAbsSHAP": np.abs(shap_values_rf.values).mean(axis=0)
}).sort_values(by="MeanAbsSHAP", ascending=False)

print(shap_importance_rf.head(10))


#NN SHAP
# Use a small background sample for SHAP
background_size = 100
test_size = 50

background_idx = np.random.choice(X_train_scaled.shape[0], background_size, replace=False)
background = X_train_scaled[background_idx]

X_test_sample = X_test_scaled[:test_size]

# Create SHAP explainer for Neural Network

#explainer_nn = shap.KernelExplainer(nn_model.predict, background)

# Extract trained MLP from pipeline
mlp_model = nn_model.named_steps["mlp"]

# Create SHAP explainer for Neural Network using the internal MLP model
explainer_nn = shap.KernelExplainer(mlp_model.predict, background)

# Compute SHAP values
shap_values_nn = explainer_nn.shap_values(X_test_sample)

# Convert to DataFrame for feature names
X_test_sample_df = pd.DataFrame(X_test_sample, columns=X_test.columns)

# Summary plot
shap.summary_plot(shap_values_nn, X_test_sample_df, show=False)
plt.tight_layout()
plt.show()

# SHAP importance table
shap_importance_nn = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanAbsSHAP": np.abs(shap_values_nn).mean(axis=0)
}).sort_values(by="MeanAbsSHAP", ascending=False)

print(shap_importance_nn.head(10))




#less layer#
# Simplified Neural Network for diagnostic comparison
simple_nn_model = MLPRegressor(
    hidden_layer_sizes=(32,),   # simpler architecture
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate_init=0.0005,
    max_iter=3000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42
)

# Train model
simple_nn_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_simple_nn = simple_nn_model.predict(X_test_scaled)

# Evaluation
results_simple_nn = evaluate_model(
    y_test,
    y_pred_simple_nn,
    "Simple Neural Network"
)

# Updated comparison table
results_df = pd.DataFrame([
    results_rf,
    results_nn,
    results_simple_nn
])

print("\nUpdated model comparison:")
print(results_df)






# --- 1. Calculate residuals and select representative sample indices ---
# Create a DataFrame to analyze prediction errors
analysis_df = pd.DataFrame({
    'True_Price': y_test,
    'Pred_Price': y_pred_rf,
    'Abs_Error': np.abs(y_test - y_pred_rf)
}, index=y_test.index) # Maintain the original index

# Selection: Best Case, Worst Case, and Typical Case
best_idx = analysis_df['Abs_Error'].idxmin()
worst_idx = analysis_df['Abs_Error'].idxmax()

# Typical Case: Find the sample with an error closest to the median error
median_err = analysis_df['Abs_Error'].median()
typical_idx = (analysis_df['Abs_Error'] - median_err).abs().idxmin()

print(f"\n--- SGH Thesis Local Analysis Indices ---")
print(f"Best Prediction Case (Min Error): {best_idx}")
print(f"Worst Prediction Case (Max Error): {worst_idx}")
print(f"Typical Prediction Case (Median Error): {typical_idx}")

# --- 2. Local explanation function ---
def generate_local_explanation(sample_index, case_name):
    # Locate the position of the sample within the test set array
    loc_pos = X_test.index.get_loc(sample_index)
    
    plt.figure(figsize=(12, 8))
    # Plot waterfall chart to visualize how features impact specific property valuation
    shap.plots.waterfall(shap_values_rf[loc_pos], max_display=10, show=False)
    #plt.title(f"SHAP Local Explanation: {case_name} (Index {sample_index})")
    plt.tight_layout()
    # Recommendation: Save the figure for inclusion in the thesis
    # plt.savefig(f"shap_local_{case_name.lower().replace(' ', '_')}.png")
    plt.show()

# --- 3. Execute generation ---
generate_local_explanation(best_idx, "Best Case")
generate_local_explanation(typical_idx, "Typical Case")
generate_local_explanation(worst_idx, "Worst Case")



### 5.6 Stability Test: Remove TotalSF ###

# Remove the most important feature
X_encoded_no_totalsf = X_encoded.drop(columns=["TotalSF"])

# Split data again using the same random state
X_train_no_ts, X_test_no_ts, y_train_no_ts, y_test_no_ts = train_test_split(
    X_encoded_no_totalsf, y, test_size=0.2, random_state=42
)

# Train Random Forest without TotalSF
best_rf_params = rf_grid.best_params_

rf_no_ts_model = RandomForestRegressor(
    **best_rf_params,
    random_state=42,
    n_jobs=-1
)

rf_no_ts_model.fit(X_train_no_ts, y_train_no_ts)
y_pred_rf_no_ts = rf_no_ts_model.predict(X_test_no_ts)

# Evaluate Random Forest without TotalSF
results_rf_no_ts = evaluate_model(
    y_test_no_ts, y_pred_rf_no_ts, "Random Forest without TotalSF"
)

# Compare original RF and RF without TotalSF
stability_results_df = pd.DataFrame([results_rf, results_rf_no_ts])
print("\nStability test results:")
print(stability_results_df)
