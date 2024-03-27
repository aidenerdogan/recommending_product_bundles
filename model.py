
# # Product Bundles Recommender System on E-commerce Data by Association Rules
# ## Aiden Ahmet Erdogan


# The developed system aims to elevate the online shopping experience through the integration of a recommendation engine, utilizing a public dataset sourced from the UCI Machine Learning Repository. This dataset encompasses a year's worth of transactional data from an online store.
# 
# The primary objective is to optimize product bundle recommendations by leveraging association rules derived from the dataset. The goal is to maximize the likelihood of purchase by suggesting relevant product bundles for each item. In instances where recommendations are lacking or insufficient for a particular product, we address the issue by completing or recommending from the most frequently suggested options to mitigate the cold start problem.
# 
# This system is designed to proactively address customer preferences within the online store environment by providing personalized product recommendations based on individual selections.
# 
# The advantages of such a system are extensive, benefiting both customers and the company:
# 
# - **For Customers:** Personalized recommendations streamline the product discovery process, enhancing user satisfaction and overall shopping experience. By facilitating the search for future purchases, customers are more inclined to return, fostering increased loyalty.
# 
# - **For the Company:** Implementation of a recommendation engine cultivates customer loyalty, thereby reducing churn rates and associated costs linked to acquiring new customers. Additionally, through cross-selling complementary products, the company can diversify revenue streams. In instances where recommendations are insufficient, strategies are employed to address the cold start problem by completing or recommending from the most commonly suggested options.


# # 1. Set up environment: Import some libraries & data transformation


from fpgrowth_py import fpgrowth # Load package *fpgrowth_py* for association rules

import numpy as np # linear algebra
import pandas as pd # data processing, data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph
%matplotlib inline
import time



# # Loading Dataset


data = pd.read_csv('data.csv', encoding='latin1') # or encoding='ISO-8859-1' # import data from a csv file


# # Data Analysis


data.head()


data.info()


# Inferences:
# 
# The dataset consists of 541,909 entries and 8 columns. Here is a brief overview of each column:
# 
# - InvoiceNo: This is an object data type column that contains the invoice number for each transaction. Each invoice number can represent multiple items purchased in a single transaction.
# - StockCode: An object data type column representing the product code for each item.
# - Description: This column, also an object data type, contains descriptions of the products. It has some missing values, with 540,455 non-null entries out of 541,909.
# - Quantity: This is an integer column indicating the quantity of products purchased in each transaction.
# - InvoiceDate: A datetime column that records the date and time of each transaction.
# - UnitPrice: A float column representing the unit price of each product.
# - CustomerID: A float column that contains the customer ID for each transaction. This column has a significant number of missing values, with only 406,829 non-null entries out of 541,909.
# - Country: An object column recording the country where each transaction took place.
# 
# From a preliminary overview, it seems that there are missing values in the Description and CustomerID columns which need to be addressed. The InvoiceDate column is already in datetime format, which will facilitate further time series analysis. We also observe that a single customer can have multiple transactions as inferred from the repeated CustomerID in the initial rows.
# 
# The next steps would include deeper data cleaning and preprocessing to handle missing values, potentially erroneous data, and to create new features that can help in achieving the project goals.


# Summerize statustics

# Summary statistics for numerical variables
data.describe().T


# Summary statistics for categorical variables
data.describe(include='object').T


# Inferences:
# 
# InvoiceDate: 
# - A datetime column that records the date and time of each transaction.
# 
# Quantity:
# - The average quantity of products in a transaction is approximately 9.55.
# - The quantity has a wide range, with a minimum value of -80995 and a maximum value of 80995. The negative values indicate returned or cancelled orders, which need to be handled appropriately.
# - The standard deviation is quite large, indicating a significant spread in the data. The presence of outliers is indicated by a large difference between the maximum and the 75th percentile values.
# 
# UnitPrice:
# - The average unit price of the products is approximately 4.61.
# - The unit price also shows a wide range, from -11062.06 to 38970, which suggests the presence of errors or noise in the data, as negative prices don't make sense.
# - Similar to the Quantity column, the presence of outliers is indicated by a large difference between the maximum and the 75th percentile values.
# 
# CustomerID:
# - There are 406829 non-null entries, indicating missing values in the dataset which need to be addressed.
# - The Customer IDs range from 12346 to 18287, helping in identifying unique customers.
# 
# InvoiceNo:
# - There are 25900 unique invoice numbers, indicating 25900 separate transactions.
# - The most frequent invoice number is 573585, appearing 1114 times, possibly representing a large transaction or an order with multiple items.
# 
# StockCode:
# - There are 4070 unique stock codes representing different products.
# - The most frequent stock code is 85123A, appearing 2313 times in the dataset.
# 
# Description:
# - There are 4223 unique product descriptions.
# - The most frequent product description is "WHITE HANGING HEART T-LIGHT HOLDER", appearing 2369 times.
# - There are some missing values in this column which need to be treated.
# 
# Country:
# - The transactions come from 38 different countries, with a dominant majority of the transactions (approximately 91.4%) originating from the United Kingdom.


# # Cleaning


# Convert InvoiceDate to datetime type
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


# Initially, I will determine the percentage of missing values present in each column, followed by selecting the most effective strategy to address them:


# Calculating the percentage of missing values for each column
missing_data = data.isnull().sum()
missing_percentage = (missing_data[missing_data > 0] / data.shape[0]) * 100

# Prepare values
missing_percentage.sort_values(ascending=True, inplace=True)

# Plot the barh chart
fig, ax = plt.subplots(figsize=(15, 4))
ax.barh(missing_percentage.index, missing_percentage, color='red')

# Annotate the values and indexes
for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
    ax.text(value+0.5, i, f"{value:.2f}%", ha='left', va='center', fontweight='bold', fontsize=18, color='black')

# Set x-axis limit
ax.set_xlim([0, 40])

# Add title and xlabel
plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
plt.xlabel('Percentages (%)', fontsize=16)
plt.show()


# Handling Missing Values Strategy:
# 
# CustomerID (24.93% missing values):
# 
# - The CustomerID column contains nearly a quarter of missing data. This column is essential for creating a recommendation system. Imputing such a large percentage of missing values might introduce significant bias or noise into the analysis.
# 
# - Moreover, since the recomending products is based on customer behavior and preferences, it's crucial to have accurate data on customer identifiers. Therefore, removing the rows with missing CustomerIDs seems to be the most reasonable approach to maintain the integrity of the the analysis.
# 
# Description (0.27% missing values):
# 
# - The Description column has a minor percentage of missing values. However, it has been noticed that there are inconsistencies in the data where the same StockCode does not always have the same Description. This indicates data quality issues and potential errors in the product descriptions.
# 
# - Given these inconsistencies, imputing the missing descriptions based on StockCode might not be reliable. Moreover, since the missing percentage is quite low, it would be prudent to remove the rows with missing Descriptions to avoid propagating errors and inconsistencies into the subsequent analyses.
# 
# By removing rows with missing values in the CustomerID and Description columns, we aim to construct a cleaner and more reliable dataset, which is essential for achieving accurate an effective recommendation system.


# This line filters the DataFrame to include rows where either 'CustomerID' or 'Description' is null
data.loc[data[['CustomerID', 'Description']].isnull().any(axis=1)].head()



# Removing rows with missing values in 'CustomerID' and 'Description' columns
data = data.dropna(subset=['CustomerID', 'Description'])


# Verifying the removal of missing values
data.isnull().sum().sum()


# Finding duplicate rows (keeping all instances)
duplicate_rows = data[data.duplicated(keep=False)]

# Sorting the data by certain columns to see the duplicate rows next to each other
duplicate_rows_sorted = duplicate_rows.sort_values(by=['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Quantity'])

# Displaying the first 10 records
duplicate_rows_sorted.head(10)


# Handling Duplicates Strategy:
# 
# In the context of this project, the presence of completely identical rows, including identical transaction times, suggests that these might be data recording errors rather than genuine repeated transactions. Keeping these duplicate rows can introduce noise and potential inaccuracies in the recommendation system.
# 
# Therefore, I am going to remove these completely identical duplicate rows from the dataset. Removing these rows will help in achieving a cleaner dataset, which in turn would aid in building more accurate product recomendation based on customer's unique purchasing behaviors. Moreover, it would help in creating a more precise recommendation system by correctly identifying the products with the most purchases.


# Displaying the number of duplicate rows
print(f"The dataset contains {data.duplicated().sum()} duplicate rows that need to be removed.")

# Removing duplicate rows
data.drop_duplicates(inplace=True)


# Getting the number of rows in the dataframe
data.shape[0]


# Treating Cancelled Transactions: 
# 
# To refine our understanding of customer behavior and preferences, we need to take into account the transactions that were cancelled. Initially, we will identify these transactions by filtering the rows where the InvoiceNo starts with "C". Subsequently, we will analyze these rows to understand their common characteristics or patterns:


# # Filter out the rows with InvoiceNo starting with "C" and create a new column indicating the transaction status
# data['Transaction_Status'] = np.where(data['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')

# # Analyze the characteristics of these rows (considering the new column)
# cancelled_transactions = data[data['Transaction_Status'] == 'Cancelled']
# cancelled_transactions.describe().drop('CustomerID', axis=1)

invoice_to_del = [el for el in data['InvoiceNo'].unique() if el[0] == 'C'] #find invoices to cancelled
data=data[data['InvoiceNo'].map(lambda x: x not in invoice_to_del)] # delete these products


# Inferences from the Cancelled Transactions Data:
# 
# - All quantities in the cancelled transactions are negative, indicating that these are indeed orders that were cancelled.
# - The UnitPrice column has a considerable spread, showing that a variety of products, from low to high value, were part of the cancelled transactions.


# Strategy for Handling Cancelled Transactions:
# 
# Considering the project's objective to creating a recommendation system based on customer behaviour, it's imperative to understand the cancellation patterns of customers. Therefore, the strategy is to retain these cancelled transactions in the dataset, marking them distinctly to facilitate further analysis. This approach will:
# 
# - Allow the recommendation system to possibly prevent suggesting products that have a high likelihood of being cancelled, thereby improving the quality of recommendations.


# # Finding the percentage of cancelled transactions
# cancelled_percentage = (cancelled_transactions.shape[0] / data.shape[0]) * 100

# # Printing the percentage of cancelled transactions
# print(f"The percentage of cancelled transactions in the dataset is: {cancelled_percentage:.2f}%")


# Correcting StockCode Anomalies¶
# 
# - First of all, lets find the number of unique stock codes and to plot the top 10 most frequent stock codes along with their percentage frequency:


# Finding the number of unique stock codes
unique_stock_codes = data['StockCode'].nunique()

# Printing the number of unique stock codes
print(f"The number of unique stock codes in the dataset is: {unique_stock_codes}")


# Finding the top 10 most frequent stock codes
top_10_stock_codes = data['StockCode'].value_counts(normalize=True).head(10) * 100

# Plotting the top 10 most frequent stock codes
plt.figure(figsize=(12, 5))
top_10_stock_codes.plot(kind='barh', color='red')

# Adding the percentage frequency on the bars
for index, value in enumerate(top_10_stock_codes):
    plt.text(value, index+0.25, f'{value:.2f}%', fontsize=10)

plt.title('Top 10 Most Frequent Stock Codes')
plt.xlabel('Percentage Frequency (%)')
plt.ylabel('Stock Codes')
plt.gca().invert_yaxis()
plt.show()


# Inferences on Stock Codes:
# 
# - **Product Variety:** The dataset contains 3684 unique stock codes, indicating a substantial variety of products available in the online retail store. This diversity can potentially lead to the identification of distinct recomendation system, with preferences for different types of products.
# 
# - **Popular Items:** A closer look at the top 10 most frequent stock codes can offer insights into the popular products or categories that are frequently purchased by customers.
# 
# - **Stock Code Anomalies**: We observe that while most stock codes are composed of 5 or 6 characters, there are some anomalies like the code 'POST'. These anomalies might represent services or non-product transactions (perhaps postage fees) rather than actual products. To maintain the focus of the project, which is creating a recommendation system, these anomalies should be further investigated and possibly treated appropriately to ensure data integrity.


# To delve deeper into identifying these anomalies, let's explore the frequency of the number of numeric characters in the stock codes, which can provide insights into the nature of these unusual entries:


# 


# Finding the number of numeric characters in each unique stock code
unique_stock_codes = data['StockCode'].unique()
numeric_char_counts_in_unique_codes = pd.Series(unique_stock_codes).apply(lambda x: sum(c.isdigit() for c in str(x))).value_counts()

# Printing the value counts for unique stock codes
print("Value counts of numeric character frequencies in unique stock codes:")
print("-"*70)
print(numeric_char_counts_in_unique_codes)


# Inference:
# 
# The output indicates the following:
# 
# - A majority of the unique stock codes (3676 out of 3684) contain exactly 5 numeric characters, which seems to be the standard format for representing product codes in this dataset.
# 
# - There are a few anomalies: 7 stock codes contain no numeric characters and 1 stock code contains only 1 numeric character. These are clearly deviating from the standard format and need further investigation to understand their nature and whether they represent valid product transactions.
# 
# Now, let's identify the stock codes that contain 0 or 1 numeric characters to further understand these anomalies:


# Finding and printing the stock codes with 0 and 1 numeric characters
anomalous_stock_codes = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]

# Printing each stock code on a new line
print("Anomalous stock codes:")
print("-"*22)
for code in anomalous_stock_codes:
    print(code)


# Let's calculate the percentage of records with these anomalous stock codes:


# Calculating the percentage of records with these stock codes
percentage_anomalous = (data['StockCode'].isin(anomalous_stock_codes).sum() / len(data)) * 100

# Printing the percentage
print(f"The percentage of records with anomalous stock codes in the dataset is: {percentage_anomalous:.2f}%")


# Inference:
# 
# - Based on the analysis, we find that a very small proportion of the records, 0.48%, have anomalous stock codes, which deviate from the typical format observed in the majority of the data. Also, these anomalous codes are just a fraction among all unique stock codes (only 8 out of 3684).
# 
# - These codes seem to represent non-product transactions like "BANK CHARGES", "POST" (possibly postage fees), etc. Since they do not represent actual products and are a very small proportion of the dataset, including them in the analysis might introduce noise and distort the recommendation system.
# 
# Strategy:
# 
# - Given the context of the project, where the aim is to develop a product recommendation system, it would be prudent to exclude these records with anomalous stock codes from the dataset. This way, the focus remains strictly on genuine product transactions, which would lead to a more accurate and meaningful analysis.
# 
# Thus, the strategy would be to filter out and remove rows with these anomalous stock codes from the dataset before proceeding with further analysis and model development:


# Removing rows with anomalous stock codes from the dataset
data = data[~data['StockCode'].isin(anomalous_stock_codes)]

# Getting the number of rows in the dataframe
data.shape[0]


# Cleaning Description Column
# 
# First, I will calculate the occurrence count of each unique description in the dataset. Then, I will plot the top 30 descriptions. This visualization will give a clear view of the highest occurring descriptions in the dataset:


# Calculate the occurrence of each unique description and sort them
description_counts = data['Description'].value_counts()

# Get the top 30 descriptions
top_30_descriptions = description_counts[:30]

# Plotting
plt.figure(figsize=(12,8))
plt.barh(top_30_descriptions.index[::-1], top_30_descriptions.values[::-1], color='red')

# Adding labels and title
plt.xlabel('Number of Occurrences')
plt.ylabel('Description')
plt.title('Top 30 Most Frequent Descriptions')

# Show the plot
plt.show()


# Inferences on Descriptions:
# 
# - The most frequent descriptions are generally household items, particularly those associated with kitchenware, lunch bags, and decorative items.
# 
# - Interestingly, all the descriptions are in uppercase, which might be a standardized format for entering product descriptions in the database. However, considering the inconsistencies and anomalies encountered in the dataset so far, it would be prudent to check if there are descriptions entered in lowercase or a mix of case styles.


# Find unique descriptions containing lowercase characters
lowercase_descriptions = data['Description'].unique()
lowercase_descriptions = [desc for desc in lowercase_descriptions if any(char.islower() for char in desc)]

# Print the unique descriptions containing lowercase characters
print("The unique descriptions containing lowercase characters are:")
print("-"*60)
for desc in lowercase_descriptions:
    print(desc)


# Inference:
# 
# - Upon reviewing the descriptions that contain lowercase characters, it is evident that some entries are not product descriptions, such as "Next Day Carriage" and "High Resolution Image". These entries seem to be unrelated to the actual products and might represent other types of information or service details.
# linkcode
# Strategy:
# 
# - Step 1: Remove the rows where the descriptions contain service-related information like "Next Day Carriage" and "High Resolution Image", as these do not represent actual products and would not contribute to the recommendation system we aim to build.
# 
# - Step 2: For the remaining descriptions with mixed case, standardize the text to uppercase to maintain uniformity across the dataset. This will also assist in reducing the chances of having duplicate entries with different case styles.
# By implementing the above strategy, we can enhance the quality of our dataset, making it more suitable for the analysis and modeling phases of our project.


service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]

# Calculate the percentage of records with service-related descriptions
service_related_percentage = data[data['Description'].isin(service_related_descriptions)].shape[0] / data.shape[0] * 100

# Print the percentage of records with service-related descriptions
print(f"The percentage of records with service-related descriptions in the dataset is: {service_related_percentage:.2f}%")

# Remove rows with service-related information in the description
data = data[~data['Description'].isin(service_related_descriptions)]

# Standardize the text to uppercase to maintain uniformity across the dataset
data['Description'] = data['Description'].str.upper()


# Getting the number of rows in the dataframe
data.shape[0]


# Treating Zero Unit Prices
# 
# In this step, first I am going to take a look at the statistical description of the UnitPrice column:


data['UnitPrice'].describe()


# Inference:
# 
# The minimum unit price value is zero. This suggests that there are some transactions where the unit price is zero, potentially indicating a free item or a data entry error. To understand their nature, it is essential to investigate these zero unit price transactions further. A detailed analysis of the product descriptions associated with zero unit prices will be conducted to determine if they adhere to a specific pattern:


data[data['UnitPrice']==0].describe()[['Quantity']]


# Inferences on UnitPrice:
# 
# - The transactions with a unit price of zero are relatively few in number (33 transactions).
# 
# - These transactions have a large variability in the quantity of items involved, ranging from 1 to 12540, with a substantial standard deviation.
# 
# 
# Strategy:
# 
# - Given the small number of these transactions and their potential to introduce noise in the data analysis, the strategy should be to remove these transactions from the dataset. This would help in maintaining a cleaner and more consistent dataset, which is essential for building an accurate and reliable recommendation system.


# Removing records with a unit price of zero to avoid potential data entry errors
data = data[data['UnitPrice'] > 0]


# Resetting the index of the cleaned dataset
data.reset_index(drop=True, inplace=True)

# Getting the number of rows in the dataframe
data.shape[0]


data.info()


# # 4. Association Rules modelling : Fp growth algorithm


# Fp Growth is a Data Mining model based on **association rules**.
# 
# This model allows, from a transaction history, to determine the set of most frequent association rules in the dataset. To do so, it needs as input parameter the set of transactions composed of the product baskets the customers have already purchased. 
# 
# Step-1: Given a dataset of transactions, the first step of FP-growth is to calculate item frequencies and identify frequent items.
# 
# Step-2: The second step of FP-growth uses a suffix tree (FP-tree) structure to encode transactions without generating candidate sets explicitly, which are usually expensive to generate. After the second step, the frequent itemsets can be extracted from the FP-tree and the model returns a set of product association rules like the example below: 
# 
#             {Product A + Product B} --> {Product C} with 60% probability
#             {Product B + Product C} --> {Product A + Product D} with 78% probability
#             {Prodcut C} --> {Product B + Product D} with 67% probability
#             etc.
#             
# To establish this table, the model needs to be provided with 2 hyperparameters :
# * **minSupRatio** : minimum support for an itemset to be identified as frequent. For example, if an item appears 3 out of 5 transactions, it has a support of 3/5=0.6.
# * **minConf** :minimum confidence for generating Association Rule. Confidence is an indication of how often an association rule has been found to be true. For example, if in the transactions itemset X appears 4 times, X and Y co-occur only 2 times, the confidence for the rule X => Y is then 2/4 = 0.5. The parameter will not affect the mining for frequent itemsets, but specify the minimum confidence for generating association rules from frequent itemsets.
# 
# Once the association rules have been calculated, all you have to do is apply them to the customers' product baskets. 


# - FP-Growth(tree based): fatser than Apriori(array based) cause of no need candidate key generation (all probablities of candidates (1, 2, 3 ..., n))
# - FP-Growth memory friendly (less memory intensive) cause no need key. array for all candidates will take more memory, space consumed for the tree in FP
# - exponantialy increase in cost for appriori by increase in items on on other hand fp increase linearly
# 
# FP-Growth (frequent-pattern growth) algorithm is an improved algorithm of the Apriori algorithm. It compresses data sets to a FP-tree, scans the database twice, does not produce the candidate item sets in mining process, and greatly improves the mining efficiency
# 


# Step 3: Recommendation Algorithm - FP-Growth
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 3: Recommendation Algorithm - FP-Growth
# One-hot encode the data for Market Basket Analysis
basket = (data.groupby(['InvoiceNo', 'StockCode'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

print("Basket Shape: ", basket.shape)

basket.head()


# Convert quantity to binary values (0 or 1)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)


a=time.time()
# Generate frequent item sets using FP-Growth
frequent_itemsets_fp = fpgrowth(basket_sets, min_support=0.005, use_colnames=True)

# Generate association rules
rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=0.5)
b=time.time()
print(b-a)
rules_fp.shape


# antecedents is basket
# consequents is nex_product
# confidence is probablity

rules = rules_fp[['antecedents', 'consequents', 'confidence']]

associations=rules.sort_values(by='confidence',ascending=False)
print('Dimensions of the association table are : ', associations.shape)
associations.head(10)


# Rename columns in associations DataFrame
associations.rename(columns={'antecedents': 'basket', 'consequents': 'product_to_recommend', 'confidence': 'probability'}, inplace=True)


# # Convert frozenset to list
# associations['basket'] = associations['basket'].apply(list)
# associations['product_to_recommend'] = associations['product_to_recommend'].apply(list)

# associations


# Sort DataFrame by 'InvoiceDate' in descending order
df = data.sort_values(by='InvoiceDate', ascending=False)

# Get unit prices and descriptions
unit_prices = df.groupby('StockCode')['UnitPrice'].first()
product_descriptions = df.groupby('StockCode')['Description'].first()



def get_price(product_ids):
    return round(sum(unit_prices.get(product_id, 0) for product_id in product_ids), 2)

def get_description(product_ids):
    return ', '.join(product_descriptions.get(product_id, '') for product_id in product_ids)



# Apply functions to create new columns
associations["price"] = associations["product_to_recommend"].apply(get_price)
associations["description"] = associations["product_to_recommend"].apply(get_description)

associations


# Define the product list for which you want recommendations
product_of_interest = '21756'

# Filter association rules for the product of interest
filtered_rules =  associations[associations['basket'] == frozenset({product_of_interest})]

# Sort the filtered rules based on lift (or any other relevant metric)
top_recommendations = filtered_rules.sort_values(by='probability', ascending=False).head(10)

# Display the top recommendations
print("Top 10 Recommendations for", product_of_interest)
top_recommendations



# Define the product for which you want recommendations
product_of_interest = '22920'

# Filter association rules for the product of interest
filtered_rules =  associations[associations['basket'] == frozenset({product_of_interest})]

# Sort the filtered rules based on lift (or any other relevant metric)
top_recommendations = filtered_rules.sort_values(by='probability', ascending=False).head(10)

# Display the top recommendations
print("Top 10 Recommendations for", product_of_interest)
top_recommendations



# # 6. Results


# #### Anticipation of customer needs :


print('On average, the recommendation system can predict in ', associations['probability'].mean() *100,  '% of the cases the next product that the customer will buy.')


# #### Turnover generated :


print('With only 1 single product proposed, the recommendation system can generate a turnover in this case up to : ', round(associations['price'].sum()), ' euros.') 


# Among a product catalog of more than 3000 items, a simple model based on association rules can predict in **~37%** of the cases the next product that the customer will buy and thus generate significant additional revenue. 
# 
# The advantage of this model is that it offers very good accuracy while being both easy to implement and explainable. Indeed, unlike some other artificial intelligence models that can seem like "black boxes" because they are difficult to explain, the results of the Fp Growth model are understandable because you will find all the rules specific to your business. For example, if you know that most of the time your customers buy product A and product B together, you will see it immediately in your association table ! 
# 





def get_bundle_price(product_ids):
    total_price = 0
    for product_id in product_ids:
        if product_id in unit_prices.index:
            total_price += unit_prices[product_id]
    return round(total_price, 2)



associations


frequent_itemsets_fp = frequent_itemsets_fp.sort_values("support", ascending=False)
frequent_itemsets = frequent_itemsets_fp["itemsets"].apply(lambda x: list(x)).explode()
type(list(frequent_itemsets))


# Function to get bundle for a product ID
def get_bundle(product_id, associations=associations, num_products=5):
    if product_id in associations["basket"].apply(list).explode().values:
        # Get association rules related to the provided product ID
        product_rules = associations[associations['basket'] == frozenset({product_id})]
        bundle_products = []
        bundle_price = 0
        
        if not product_rules.empty:
            # Sort by probability to get most relevant associations
            product_rules = product_rules.sort_values(by='probability', ascending=False)

            # Add products from antecedents and consequents of association rules
            for _, row in product_rules.iterrows():
                bundle_products = bundle_products + (list(row['product_to_recommend']))
                
                # Check if the desired number of products is reached
                if len(set(bundle_products)) >= num_products:
                    break
            
            bundle_products = list(set(bundle_products))
            # Get the total price for the bundle
            bundle_price = round(sum(unit_prices.get(product, 0) for product in bundle_products), 2)

        if len(bundle_products) < num_products or product_rules.empty:
            # If no association rules found for the product, pick from top frequent items
            frequent_itemsets = frequent_itemsets_fp.sort_values("support", ascending=False)
            frequent_items = list(frequent_itemsets["itemsets"].apply(lambda x: list(x)).explode())
            uniques_frequent_items = [x for x in frequent_items if x not in bundle_products]
            sample_size = (num_products-len(bundle_products))
            bundle_products = list(bundle_products)+ uniques_frequent_items[:sample_size]
            bundle_products = list(set(bundle_products))

            # Get the total price for the bundle
            bundle_price = round(sum(unit_prices.get(product, 0) for product in bundle_products), 2)
            
            return bundle_products, bundle_price
        else:
            return list(bundle_products), bundle_price

    else:
        return "Product ID not found in the dataset"

# Example usage:
product_id = '22920'  # Replace with the desired product ID
bundle_result = get_bundle(product_id)
if isinstance(bundle_result, tuple):
    bundle_products, bundle_price = bundle_result
    print("Bundle Products:", bundle_products)
    print("Bundle Price:", bundle_price)
else:
    print(bundle_result)



