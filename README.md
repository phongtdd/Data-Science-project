# Data Science Project
Mini-project for course of Intro to Data Science, with the aim of developing a application that predictr the price of clothes.

# Project Source Code  

For examination purposes, you can check the source code of the project. The repository is organized as follows:  

<<<<<<< HEAD
### 1. **Model**  
=======
### 1. model
>>>>>>> f46ce65ff4d31cc4cf71bfc25f76e93df3c845ce
- This folder contains all the code required to train our model, which is used for predicting the price of clothes.  
- Additionally, it includes the pre-trained model path, allowing the model to be run without retraining.  

### 2. UI 
- This folder contains all the code related to the user interface of the Streamlit application, including the main ui.py file for launching the interface, modules for predictions using different models and utility functions located in the UI.utils directory.

### 3. label_encode
- This folder contains two models for label encoding: one for brand names and another for colors.  

### 4. final_data
- This folder stores the raw data for each category collected during the crawling process.  

<<<<<<< HEAD
### 5. **Crawl**  
This folder contains all the scripts for extracting product data from the Amazon website. The included files serve specific purposes in the data crawling pipeline as outlined below:  
1. **`search.py`**: Allows users to define a category of interest (e.g., "t-shirt", "dress") and perform a search query. The script outputs product information into a JSON file located at `data/amazon_unique_<your_query>.json`.  
2. **`extract_asin.py`**: Extracts the Amazon Standard Identification Number (ASIN) codes from the JSON file generated in the previous step.  
3. **`variants.py`**: Retrieves detailed variant information of a product based on the extracted ASIN codes. Variants include attributes such as size, color, and the ASIN codes corresponding to each variant.  
4. **`filter_data.py`**: Applies filters to ensure that the size attributes of the variant products fall within a predefined range: [Small, Medium, Large, X-Large, XX-Large].  
5. **`scrapes.py`**: Extracts comprehensive information about a product identified by its ASIN code.  

### 6. **Data**  
This folder stores all the raw data obtained during the data crawling process.

=======
### 5. crawl  
- *[Details to be added]*  

### 6. data
- *[Details to be added]*  
>>>>>>> f46ce65ff4d31cc4cf71bfc25f76e93df3c845ce

### 7. EDA notebook 
- This notebook contains the code for visualizing and analyzing the data during the exploratory data analysis phase.  

<<<<<<< HEAD
### 8. **Preprocess_initial_data notebook**  
- This notebook focuses on normalizing the raw data and preprocessing it for training and visualization.  

### 9. **Final_data.json and preprocess_data.json**  
=======
### 8. preprocess_initial_data notebook
- This notebook focuses on normalizing the raw data and preprocessing it for training and visualization.  

### 9. final_data.json and preprocess_data.json
>>>>>>> f46ce65ff4d31cc4cf71bfc25f76e93df3c845ce
- These files represent the processed data:
  - `final_data.json`: Data after preprocessing.
  - `preprocess_data.json`: Data prepared for model training.  

# Run the Application
Follow these steps to launch the app to test all models:

1. Navigate to the Project Directory
Open a terminal or command prompt and navigate to the project folder:

```bash
cd Data-Science-Project
```

2. Use the following command to run the app:
```bash
python -m streamlit run UI/ui.py
```
Once the command runs successfully, Streamlit will launch a local server.
Open the provided URL in your web browser. It typically looks like this:
```bash
http://localhost:8501
```
If the app does not open automatically, copy and paste the URL into your browser.

3. Interact with the UI
- First, select the entry mothod: 
  - Choose "Choose a product in database" to test predictions using an existing product from the database.
  - Select "Enter the data manually" to input custom product details for prediction.

- After finish input the data you want, select a Model.
- Finally, click on the Predict button to view the prediction results.
