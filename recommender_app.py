from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the basket data and association rules
try:
    associations = pd.read_csv("association_rules.csv")
    frequent_itemsets_fp = pd.read_csv("frequent_itemsets_fp.csv")
    unit_prices = pd.read_csv("unit_prices.csv")
except Exception as e:
    print("Error loading data:", e)

# Function to get bundle for a product ID
def get_bundle(product_id, associations=associations,frequent_itemsets_fp=frequent_itemsets_fp, unit_prices=unit_prices, num_products=5):
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


@app.route('/recommend', methods=['GET'])
def recommend_bundle():
    product_id = request.args.get('product_id')
    print("********: ",product_id)
    
    if product_id is None:
        return jsonify({'error': 'Product ID parameter is missing'}), 400
    bundle_products, bundle_price = [], []
    try:
        product_id = '23170' 
        bundle_result = get_bundle(product_id)
        print(bundle_result)
        if isinstance(bundle_result, tuple):
            bundle_products, bundle_price = bundle_result
            print("Bundle Products:", bundle_products)
            print("Bundle Price:", bundle_price)
        else:
            print(bundle_result)
    except Exception as e:
        print("Error retrieving bundle information:", e)
        return jsonify({'error': 'Error retrieving bundle information'}), 500
    
    return jsonify({
        'product_id': product_id,
        'bundle_products': bundle_products,
        'bundle_price': bundle_price
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)