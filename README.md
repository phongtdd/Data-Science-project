# Data Science Project

**Mini project for Data Science courses**

## Project Source Code  

For examination purposes, you can check the source code of the project. The repository is organized as follows:  

### 1. **Model**  
- This folder contains all the code required to train our model, which is used for predicting the price of clothes.  
- Additionally, it includes the pre-trained model path, allowing the model to be run without retraining.  

### 2. **UI**  
- *[Details to be added]*  

### 3. **label_encode**  
- This folder contains two models for label encoding: one for brand names and another for colors.  

### 4. **final_data**  
- This folder stores the raw data for each category collected during the crawling process.  

### 5. **Crawl**  
This folder contains all the scripts for extracting product data from the Amazon website. The included files serve specific purposes in the data crawling pipeline as outlined below:  
1. **`search.py`**: Allows users to define a category of interest (e.g., "t-shirt", "dress") and perform a search query. The script outputs product information into a JSON file located at `data/amazon_unique_<your_query>.json`.  
2. **`extract_asin.py`**: Extracts the Amazon Standard Identification Number (ASIN) codes from the JSON file generated in the previous step.  
3. **`variants.py`**: Retrieves detailed variant information of a product based on the extracted ASIN codes. Variants include attributes such as size, color, and the ASIN codes corresponding to each variant.  
4. **`filter_data.py`**: Applies filters to ensure that the size attributes of the variant products fall within a predefined range: [Small, Medium, Large, X-Large, XX-Large].  
5. **`scrapes.py`**: Extracts comprehensive information about a product identified by its ASIN code.  

### 6. **Data**  
This folder stores all the raw data obtained during the data crawling process.


### 7. **EDA notebook**  
- This notebook contains the code for visualizing and analyzing the data during the exploratory data analysis phase.  

### 8. **Preprocess_initial_data notebook**  
- This notebook focuses on normalizing the raw data and preprocessing it for training and visualization.  

### 9. **Final_data.json and preprocess_data.json**  
- These files represent the processed data:
  - `final_data.json`: Data after preprocessing.
  - `preprocess_data.json`: Data prepared for model training.  
