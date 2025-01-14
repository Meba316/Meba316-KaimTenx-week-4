import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Load the training and test datasets
train_df = pd.read_csv('train.csv')  # Training data with sales (target variable)
test_df = pd.read_csv('test.csv')    # Test data without sales (used for prediction)

# Quick check of the data to ensure it's loaded correctly
print(train_df.head())

# Check basic information (data types, number of null values, etc.)
print(train_df.info())

# Check descriptive statistics for numerical features (mean, median, etc.)
print(train_df.describe())

# Check for missing values
print(train_df.isnull().sum())

# Fill missing values in 'CompetitionDistance' with the median
train_df['CompetitionDistance'].fillna(train_df['CompetitionDistance'].median(), inplace=True)


# Calculate IQR for 'Sales'
Q1 = train_df['Sales'].quantile(0.25)
Q3 = train_df['Sales'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
train_df = train_df[(train_df['Sales'] >= lower_bound) & (train_df['Sales'] <= upper_bound)]



# Plot a histogram for 'Sales'
plt.hist(train_df['Sales'], bins=50, color='skyblue')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()



# Compute correlation matrix
corr = train_df.corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Check promotion distribution in training and test datasets
train_promo_dist = train_df['Promo'].value_counts(normalize=True)
test_promo_dist = test_df['Promo'].value_counts(normalize=True)

print("Training Promo Distribution:\n", train_promo_dist)
print("Test Promo Distribution:\n", test_promo_dist)


# Compare sales during different holidays
holiday_sales = train_df.groupby('StateHoliday')['Sales'].mean()
print(holiday_sales)


# Compare sales with and without promotions
promo_sales = train_df.groupby('Promo')['Sales'].mean()
print(promo_sales)

# Sales by Assortment Type
assortment_sales = train_df.groupby('Assortment')['Sales'].mean()
print(assortment_sales)




# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Log the start of the data loading process
logging.info("Started loading the dataset")
train_df = pd.read_csv('train.csv')
logging.info("Dataset loaded successfully")

# Log when cleaning and analysis begins
logging.info("Started exploratory data analysis")
