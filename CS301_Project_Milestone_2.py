# # CS301-002 Project Milestone 2 - Group 12: Happiness vs. Cost of Living
# ### Group Members:
# * James Cayetano - jtc62@njit.edu
# * Brandon Zhou - bz33@njit.edu
# * John Kim - jyk26@njit.edu
# * Aran Kashani - awk34@njit.edu
# 
# ### Data Sources:
# Dataset 1: [City Happiness Index - 2024 | Kaggle](https://www.kaggle.com/datasets/emirhanai/city-happiness-index-2024)
# - This dataset contains over 520 cities from across the world with features related to quality of life such as noise levels, the level of traffic density, air quality index, and healthcare quality.
# 
# Dataset 2: [Global Cost of Living | Kaggle](https://www.kaggle.com/datasets/mvieira101/global-cost-of-living)
# - This dataset contains around 5000 cities from across the world with features related to said cities’ cost of living such as the cost for different types of meals, groceries, the prices of gas, property value, and salary rate.

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

os.makedirs("output", exist_ok=True)

# 1. Exploratory Data Analysis

happiness = pd.read_csv('content/train.csv')
cost = pd.read_csv('content/cost-of-living_v2.csv')

#
#   HAPPINESS Exploratory Data Analysis
#

happiness = happiness.drop('Cost_of_Living_Index', axis=1) # We will be calculating our own cost of living index using the cost dataset.

# Plot correlation heatmaps of each numerical feature
fig, ax = plt.subplots(1, 2, figsize=(18,8))
sns.heatmap(happiness[['Year', 'Decibel_Level', 'Green_Space_Area', 'Air_Quality_Index', 'Happiness_Score', 'Healthcare_Index']].corr(), cmap="coolwarm", annot=True, ax=ax[0])
sns.heatmap(happiness[['Year', 'Decibel_Level', 'Green_Space_Area', 'Air_Quality_Index', 'Happiness_Score', 'Healthcare_Index']][happiness['Happiness_Score'] >= 1].corr(), cmap="coolwarm", annot=True, ax=ax[1])
ax[0].set_title('Heatmap of all data')
ax[1].set_title('Heatmap of data where Happiness_Score >= 1')
plt.tight_layout()
plt.savefig('output/happiness_corrheatmaps.png', bbox_inches='tight')
plt.close(fig)

happiness = happiness[happiness['Happiness_Score'] >= 1] # Use only data with valid happiness scores
happiness = happiness.drop(['Year', 'Month'], axis=1)

# Plot distributions of each feature and the target
happiness_numeric = {
    'Decibel_Level':np.arange(50, 95, 5),
    'Green_Space_Area':np.arange(0, 450, 50),
    'Air_Quality_Index':np.arange(0, 275, 25),
    'Happiness_Score':np.arange(0, 12, 2),
    'Healthcare_Index':np.arange(30, 110, 10)
}

fig, ax = plt.subplots(3, 2, figsize=(20, 20))
for idx, col in enumerate(happiness_numeric.keys()):
    sns.histplot(happiness[col], ax=ax[idx//2][idx % 2], kde=True, bins=happiness_numeric[col])
sns.countplot(data=happiness, x='Traffic_Density', order=['Low', 'Medium', 'High', 'Very High'], ax=ax[2][1]) 
plt.title('Histograms of each feature in the first dataset')
plt.tight_layout()
plt.savefig('output/happiness_featurehistograms.png')
plt.close()

#
#   COST Exploratory Data Analysis
#

cost = cost.drop(['x54', 'x55'], axis=1)

# count and plot the number of NaN values in each row
cost['missing_count'] = cost.isna().sum(axis=1)

sns.histplot(cost['missing_count'], kde=True, bins=np.arange(0, 60, 2))
plt.title('Distribution of Missing Values per Record')
plt.xlabel('Number of missing values')
plt.tight_layout()
plt.savefig("output/cost_missingvalues.png")
plt.close()

# It is unreasonable to view the distribution of every single feature in this dataset, so we will first group them into the following categories:
# * Food (subgroups: restaurant and grocery)
# * Transportation (subgroups: public transit, taxis, fuel, vehicle purchase)
# * Utilities
# * Lifestyle (subgroups: discretionary goods, clothing, recreation)
# * Housing
# * Rent
# * Childcare
# 
# For each subcategory, we will min-max scale its values then average them. Then, we can average the subcategories to get the value for the category. This is necessary because higher-cost items like meats would have more impact on the average than a lower-cost item such as rice. Additionally, some subcategories have more columns than others, which would cause them to have a higher impact the averages for the category.
# 
# If a category does not have a subcategory, its values will be min-max scaled and then averaged.
# 
# Once the category averages are computed, they will be transformed into a scale from 0 to 100.


# food
restaurant = ['x1', 'x2', 'x3', 'x6', 'x7', 'x8']
grocery = ['x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23']

# transportation
public_transit = ['x28', 'x29']
taxi = ['x30', 'x31', 'x32']
fuel = ['x33']
vehicle = ['x34', 'x35']

# utilities
utilities = ['x36', 'x37', 'x38']

# lifestyle
discretionary = ['x4', 'x5', 'x24', 'x25', 'x26', 'x27']
clothing = ['x44', 'x45', 'x46', 'x47']
recreation = ['x39', 'x40', 'x41']

# housing
housing = ['x52', 'x53']

# rent
rent = ['x48', 'x49', 'x50', 'x51']

# childcare
childcare = ['x42', 'x43']

# scale values in each column
cols = ['x' + str(num) for num in range(1, 54)]
min_max_scaler = MinMaxScaler()
cost_scaled = pd.DataFrame(min_max_scaler.fit_transform(cost[cols]), columns=[col + '_scaled' for col in cols], index=cost.index)

# compute average of each subcategory/category without subcategories
subcategories = {'restaurant':restaurant, 'grocery':grocery, 'public_transit':public_transit, 'taxi':taxi, 'fuel':fuel, 'vehicle':vehicle, 'utilities':utilities, 'discretionary':discretionary, 'clothing':clothing, 'recreation':recreation, 'housing':housing, 'rent':rent, 'childcare':childcare}

for name, sc in subcategories.items():
    cols = [x + '_scaled' for x in sc]
    cost_scaled[name] = cost_scaled[cols].mean(axis=1)

# compute average of each category with subcategories
cost_scaled['food'] = cost_scaled[['restaurant', 'grocery']].mean(axis=1)
cost_scaled['transportation'] = cost_scaled[['public_transit', 'taxi', 'fuel', 'vehicle']].mean(axis=1)
cost_scaled['lifestyle'] = cost_scaled[['discretionary', 'clothing', 'recreation']].mean(axis=1)

# match category values to cities in the original data
cost_categorized = cost[['city', 'country', 'missing_count']].join(cost_scaled[['food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare']])

# calculate cost of living index by averaging categories:
cost_categorized['cost_of_living'] = cost_categorized[['food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare']].mean(axis=1)

# convert everything to a 0-100 scale:
for col in ['food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare', 'cost_of_living']:
    cost_categorized[col] = cost_categorized[col] * 100


plt.figure(figsize=(10, 6))
sns.heatmap(cost_categorized[['food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare', 'cost_of_living', 'missing_count']].corr(), cmap=sns.cubehelix_palette(as_cmap=True), annot=True)
plt.title('Heatmap of correlation between cost of living categories')
plt.tight_layout()
plt.savefig("output/cost_categoryheatmap.png")
plt.close()

# Plot distributions of each category
categories = {
    'food': np.arange(0, 70, 5),
    'transportation': np.arange(0, 105, 5),
    'utilities': np.arange(0, 70, 5),
    'lifestyle': np.arange(0, 70, 5),
    'housing': np.arange(0, 105, 5),
    'rent': np.arange(0, 105, 10),
    'childcare': np.arange(0, 100, 5),
    'cost_of_living': np.arange(0, 60, 5)
}

fig, ax = plt.subplots(4, 2, figsize=(20, 20))
for idx, col in enumerate(categories.keys()):
    sns.histplot(cost_categorized[col], ax=ax[idx//2][idx % 2], kde=True, bins=categories[col])
plt.title('Histograms of Each Cost-of-Living Category')
plt.tight_layout()
plt.savefig("output/cost_categorydistributions.png")
plt.close()

#
#   MERGED DATA Exploratory Data Analysis
#

# merge with happiness data for bivariate exploration with the target
cost_categorized = cost_categorized.rename(columns={'city':'City'})
merged = pd.merge(happiness, cost_categorized, on='City')

# There are some cities in different countries that share a name. Since the country isn't be used in the statistcal analysis/ML, we can arbitrarily drop duplicates.
merged = merged.drop([4, 25, 93, 63, 44, 82, 84]) # 4             London                Canada
                                                  # 25         Barcelona             Venezuela
                                                  # 93         Cambridge                Canada
                                                  # 63           Colombo                Brazil
                                                  # 44          Santiago                Panama
                                                  # 82          Hamilton               Bermuda
                                                  # 84          Hamilton             Australia

# Drop cities with high amount of missing values
merged = merged.drop([95, 79, 92])

plt.figure(figsize=(12, 8))
sns.heatmap(merged[['Decibel_Level', 'Green_Space_Area', 'Air_Quality_Index', 'Happiness_Score', 'Healthcare_Index', 'missing_count', 'food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare', 'cost_of_living']].corr(), cmap="coolwarm", annot=True)
plt.title('Correlation Heatmap of the Merged Datasets')
plt.tight_layout()
plt.savefig("output/merged_corrheatmap.png")
plt.close()

# Bivariate Exploration
sns.pairplot(merged)
plt.title('Pairplot Between all Numerical Features of the Merged Data')
plt.tight_layout()
plt.savefig("output/merged_pairplot.png")
plt.close()


features = ['Decibel_Level', 'Green_Space_Area', 'Air_Quality_Index', 'Healthcare_Index', 'food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare', 'cost_of_living']

fig, ax = plt.subplots(len(features)//2, 2, figsize=(20, 40))
for idx, col in enumerate(features):
    sns.barplot(data=merged, x='Traffic_Density', y=col, order=['Low', 'Medium', 'High', 'Very High'], ax=ax[idx//2][idx % 2])
plt.title('Average Value of each Feature and Traffic Density')
plt.tight_layout()
plt.savefig("output/merged_trafficdensityvsfeatures.png")
plt.close()


features = ['Decibel_Level', 'Green_Space_Area', 'Air_Quality_Index', 'Healthcare_Index', 'food', 'transportation', 'utilities', 'lifestyle', 'housing', 'rent', 'childcare', 'cost_of_living']

fig, ax = plt.subplots(len(features)//2, 2, figsize=(20, 40))
for idx, col in enumerate(features):
    sns.scatterplot(merged, x=col, y='Happiness_Score', ax=ax[idx//2][idx % 2])
plt.title('Scatterplots between Individual Features and the Target')
plt.tight_layout()
plt.savefig("output/merged_featuresvstarget.png")
plt.close()

sns.barplot(data=merged, x='Traffic_Density', y='Happiness_Score', order=['Low', 'Medium', 'High', 'Very High'])
plt.title('Average Happiness Score by Traffic Density')
plt.tight_layout()
plt.savefig("output/merged_trafficvstarget.png")
plt.close()

#
#   STATISTCAL ANALYSIS
#

df = merged.copy()

# Encode traffic density
traffic_map = {'Low' : 1, 'Medium': 2, 'High' : 3, 'Very High' : 4}
df['Traffic_Density_Encoded'] = df['Traffic_Density'].map(traffic_map)

# Define features
test_features = ['Decibel_Level', 'Traffic_Density_Encoded', 'Green_Space_Area', 'Air_Quality_Index', 'Healthcare_Index', 'cost_of_living']

# Hypothesis testing
alpha = 0.05

with open("output/pearson_results.txt", "w") as file:
    file.write("="*60)
    file.write("PEARSON CORRELATION vs HAPPINESS SCORE")
    file.write("="*60)
    file.write(f"{'Feature':<28} {'r':>8} {'P-Value':>10}   Verdict")
    file.write("-"*60)

    for col in test_features:
        clean = df[[col, 'Happiness_Score']].dropna()
        r, p = stats.pearsonr(clean[col], clean['Happiness_Score'])
        verdict = "We reject the null hypothesis." if p < alpha else "We fail to reject the null hypothesis."
        file.write(f"{col:<28} {r:>8.4f} {p:>10.2e}   {verdict}")

    file.write("-"*60)
    file.write(f"\nAlpha: {alpha}")
    file.write("Reject H0 if p-value < 0.05")

#
#   MODEL EVALUATION
#
# Features and Target
X = merged[['Decibel_Level', 'Traffic_Density', 'Green_Space_Area', 'Air_Quality_Index', 'Healthcare_Index', 'cost_of_living']]
y = merged['Happiness_Score']

# Preprocess Categorical Column
transformer = ColumnTransformer(
    transformers=[
        ('traffic', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']]), ['Traffic_Density'])
    ],
    remainder='passthrough'
)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Transform Data
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

with open("output/model_results.txt", "w") as file:
    file.write(f"R^2: {r2}")
    file.write(f"RMSE: {rmse}")


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2)

plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.title("Actual vs Predicted Happiness Scores (Linear Regression)")
plt.tight_layout()
plt.savefig("output/model_regressiontest.png")
plt.close()

# Get feature names after transformation
feature_names = ['Traffic_Density_encoded',
                 'Decibel_Level', 'Green_Space_Area',
                 'Air_Quality_Index', 'Healthcare_Index',
                 'cost_of_living']

coeffs = model.coef_

with open("output/model_coefficients.txt", "w") as file:
    for name, coef in zip(feature_names, coeffs):
        file.write(f"{name}: {coef}")

#
#   KNOWLEDGE DISCOVERY
#

features_to_cluster = ['Happiness_Score', 'cost_of_living', 'Traffic_Density', 'Healthcare_Index']

cluster_data = merged[features_to_cluster].dropna().copy()

for col in cluster_data.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    cluster_data[col] = le.fit_transform(cluster_data[col])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=cluster_data,
    x='cost_of_living',
    y='Happiness_Score',
    hue='Cluster',
    palette='viridis',
    alpha=0.7
)
plt.title('Knowledge Discovery: City Clusters (Cost vs. Happiness)')
plt.xlabel('Cost of Living Index')
plt.ylabel('Happiness Score')
plt.tight_layout()
plt.savefig("output/clustering_result.png")
plt.close()

cluster_means = cluster_data.groupby('Cluster').mean()
cluster_means.to_csv("output/cluster_means.csv")
