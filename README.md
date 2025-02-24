# MediMatcher: AI Powered Prescription Assistant

 

## Source
- This app uses **Tesseract OCR** for Optical Character Recognition (OCR) to extract text from prescription images.
- The medicine matching feature leverages fuzzy string matching via `difflib` and a **Trie** data structure for fast medicine name lookup.
- The dataset containing the medicine names and compositions is downloaded from [Kaggle: A-Z Medicine List from India](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india).

## Features
- **Upload Prescription Images**: Upload prescription images in formats like PNG, JPG, JPEG.
- **OCR Text Extraction**: Uses Tesseract OCR to extract text from images.
- **Medicine Matching**: Matches extracted text with a preloaded dataset of medicines using fuzzy matching and Trie-based search.
- **Pharmacist Cart**: Allows pharmacists to add identified medicines to a cart and proceed with checkout.

## Usage

- Clone my repository.
  
  ```
  git clone https://github.com/hitha-cse513/MediMatcher
  cd MediMatcher
  ```

- Open CMD in working directory.
- Run the following command to install the required dependencies:

  ```
  pip install -r requirements.txt
  ```

- Ensure Tesseract is installed on your machine. If not, follow the installation instructions on the [Tesseract GitHub page](https://github.com/tesseract-ocr/tesseract).
- Place the dataset CSV file (`Medicine.csv`) in the project folder.
- `app.py` is the main Python file for the Streamlit web application.
- To run the app, use the following command in your terminal or IDE:

  ```
  streamlit run app.py
  ```


## Screenshots

<img src="https://github.com/hitha-cse513/MediMatcher1.png">
