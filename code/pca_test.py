import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data (replace this with your actual data)
np.random.seed(42)
n_samples = 1000

# Create 5 boolean independent variables
X1 = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
X2 = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
X3 = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
X4 = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
X5 = np.random.choice([0, 1], n_samples, p=[0.55, 0.45])

# Create dependent variable with some relationship to independent variables
# Making it more likely to be 1 when X1 and X4 are 1, less likely when X3 is 1
y_prob = 0.3 + 0.3*X1 + 0.2*X2 - 0.1*X3 + 0.4*X4 + 0.1*X5
y_prob = np.clip(y_prob, 0, 1)  # Keep probabilities between 0 and 1
y = np.random.binomial(1, y_prob)

# Create DataFrame
df = pd.DataFrame({
    'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'y': y
})

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nVariable means (proportion of 1s):")
print(df.mean())

# Separate independent and dependent variables
X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
y = df['y']

# Method 1: Direct PCA on Independent Variables
print("\n" + "="*50)
print("METHOD 1: PCA ON INDEPENDENT VARIABLES")
print("="*50)

# Apply PCA to independent variables
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Print explained variance
print("Explained variance ratio by each component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"Component {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\nCumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")

# Show component loadings (how much each original variable contributes to each component)
components_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(5)],
    index=['X1', 'X2', 'X3', 'X4', 'X5']
)

print("\nPCA Component Loadings:")
print(components_df.round(4))

# Method 2: Correlation between original variables and dependent variable
print("\n" + "="*50)
print("METHOD 2: CORRELATION WITH DEPENDENT VARIABLE")
print("="*50)

correlations = X.corrwith(y).sort_values(key=abs, ascending=False)
print("Correlation with dependent variable (sorted by absolute value):")
for var, corr in correlations.items():
    print(f"{var}: {corr:.4f}")

# Method 3: Variable importance through PCA + Regression
print("\n" + "="*50)
print("METHOD 3: PCA-BASED VARIABLE IMPORTANCE")
print("="*50)

# Use fewer components (e.g., explaining 80% of variance)
n_components = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.8) + 1
print(f"Using {n_components} components to explain ≥80% of variance")

pca_reduced = PCA(n_components=n_components)
X_pca_reduced = pca_reduced.fit_transform(X)

# Fit logistic regression on PCA components
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_pca_reduced, y)

# Calculate variable importance by combining PCA loadings with regression coefficients
loadings = pca_reduced.components_  # Shape: (n_components, n_features)
coefficients = log_reg.coef_[0]     # Shape: (n_components,)

# Variable importance = sum of (loading * coefficient) for each variable across components
variable_importance = np.abs(loadings.T @ coefficients)
importance_df = pd.DataFrame({
    'Variable': ['X1', 'X2', 'X3', 'X4', 'X5'],
    'Importance': variable_importance
}).sort_values('Importance', ascending=False)

print("\nVariable Importance (PCA + Logistic Regression):")
print(importance_df)

# Method 4: Individual Logistic Regression for comparison
print("\n" + "="*50)
print("METHOD 4: DIRECT LOGISTIC REGRESSION (FOR COMPARISON)")
print("="*50)

log_reg_direct = LogisticRegression(random_state=42)
log_reg_direct.fit(X, y)

direct_importance = pd.DataFrame({
    'Variable': ['X1', 'X2', 'X3', 'X4', 'X5'],
    'Coefficient': log_reg_direct.coef_[0],
    'Abs_Coefficient': np.abs(log_reg_direct.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("Direct Logistic Regression Coefficients:")
print(direct_importance)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: PCA Explained Variance
axes[0,0].bar(range(1, 6), pca.explained_variance_ratio_)
axes[0,0].set_title('PCA Explained Variance by Component')
axes[0,0].set_xlabel('Principal Component')
axes[0,0].set_ylabel('Explained Variance Ratio')
axes[0,0].set_xticks(range(1, 6))

# Plot 2: Component Loadings Heatmap
sns.heatmap(components_df.iloc[:, :3], annot=True, cmap='RdBu_r', center=0, ax=axes[0,1])
axes[0,1].set_title('PCA Component Loadings (First 3 Components)')

# Plot 3: Correlation with Dependent Variable
correlations.plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Correlation with Dependent Variable')
axes[1,0].set_ylabel('Correlation')
axes[1,0].tick_params(axis='x', rotation=45)

# Plot 4: Variable Importance Comparison
x_pos = np.arange(len(importance_df))
axes[1,1].bar(x_pos - 0.2, importance_df['Importance'], 0.4, label='PCA-based')
axes[1,1].bar(x_pos + 0.2, direct_importance.sort_values('Variable')['Abs_Coefficient'], 0.4, label='Direct LogReg')
axes[1,1].set_title('Variable Importance Comparison')
axes[1,1].set_xlabel('Variables')
axes[1,1].set_ylabel('Importance')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(importance_df['Variable'])
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*50)
print("SUMMARY OF MOST IMPACTFUL VARIABLES")
print("="*50)
print("1. By Correlation:", correlations.index[0], f"({correlations.iloc[0]:.4f})")
print("2. By PCA+LogReg:", importance_df.iloc[0]['Variable'], f"({importance_df.iloc[0]['Importance']:.4f})")
print("3. By Direct LogReg:", direct_importance.iloc[0]['Variable'], f"({direct_importance.iloc[0]['Abs_Coefficient']:.4f})")

