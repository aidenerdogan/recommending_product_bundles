# conrad_recommending_product_bundles
Conrad Electronics Recommending Product Bundles

### FP-Growth Algorithm Overview:
- **Objective:** Association rule mining based on transaction history to determine the set of most frequent association rules.
- **Step 1: Item Frequency Calculation:** Calculate item frequencies and identify frequent items in the dataset.
- **Step 2: FP-Tree Construction:** Utilize a suffix tree (FP-tree) structure to encode transactions without generating candidate sets explicitly.
- **Result:** Extract frequent itemsets from the FP-tree to generate association rules.

### Hyperparameters:
1. **minSupRatio:** Minimum support for an itemset to be identified as frequent.
2. **minConf:** Minimum confidence for generating association rules from frequent itemsets.

### Association Rule Example:
- Example Association Rules:
  - {Product A + Product B} --> {Product C} with 60% probability
  - {Product B + Product C} --> {Product A + Product D} with 78% probability
  - {Product C} --> {Product B + Product D} with 67% probability

### Comparison with Apriori Algorithm:
- FP-Growth is faster than Apriori as it doesn't require candidate key generation.
- FP-Growth is memory-friendly as it utilizes less memory compared to Apriori, which requires storing candidate sets.
- FP-Growth's cost increases linearly with an increase in items, while Apriori's cost increases exponentially.

### Advantages of FP-Growth:
- Improved efficiency compared to the Apriori algorithm.
- Compression of datasets into an FP-tree.
- Reduced mining process complexity by avoiding the generation of candidate item sets.

Overall, FP-Growth algorithm is a powerful approach for association rule mining, offering speed and memory efficiency advantages over traditional methods like Apriori.

- FP-Growth(tree based): fatser than Apriori(array based) cause of no need candidate key generation (all probablities of candidates (1, 2, 3 ..., n))
- FP-Growth memory friendly (less memory intensive) cause no need key. array for all candidates will take more memory, space consumed for the tree in FP
- exponantialy increase in cost for appriori by increase in items on on other hand fp increase linearly

FP-Growth (frequent-pattern growth) algorithm is an improved algorithm of the Apriori algorithm. It compresses data sets to a FP-tree, scans the database twice, does not produce the candidate item sets in mining process, and greatly improves the mining efficiency


#fastest and most updated way to install requirements uv, for more check [here](https://pypi.org/project/uv/)
pip install uv

#create env to not relfect anly local or exist file.
#I assume taht you have conda installed otherwise chek [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for how to create virtual env without conda.
conda create -n conrad_rpb python=3.12 
conda activate conrad_rpb
uv pip install -r requirements.txt

