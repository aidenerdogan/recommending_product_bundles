# conrad_recommending_product_bundles
Conrad Electronics Recommending Product Bundles

#fastest and most updated way to install requirements uv, for more check [here](https://pypi.org/project/uv/)
pip install uv

#create env to not relfect anly local or exist file.
#I assume taht you have conda installed otherwise chek [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for how to create virtual env without conda.
conda create -n conrad_rpb python=3.12 
conda activate conrad_rpb
uv pip install -r requirements.txt

