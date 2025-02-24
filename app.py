import os
import re
import cv2
import pytesseract
import pandas as pd
import difflib
import streamlit as st
from PIL import Image as PILImage

# Set the path to Tesseract executable (adjust this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# Load the dataset (CSV file containing medicines)
def load_dataset():
    return pd.read_csv('Medicine.csv')  # Replace with the path to your CSV file

df = load_dataset()

# Preprocess image to improve OCR accuracy
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    processed_img = PILImage.fromarray(thresholded)
    return processed_img

# Extract medicine names from the OCR text
def extract_medicine_names(ocr_text):
    lines = ocr_text.split('\n')
    medicine_keywords = [
        'tablet', 'capsule', 'capsules', 'syrup', 'injection', 'vaccine', 'cream', 'solution', 
        'ointment', 'suspension', 'drops', 'pills', 'powder', 'tab', 'tabs', 'mg', 'g','ml','mg'
    ]
    medicines = []

    for line in lines:
        if any(keyword in line.lower() for keyword in medicine_keywords):
            # Remove dosages and unnecessary characters
            cleaned_line = re.sub(r'\d+\s*(mg|g|ml|tabs|tab|mls|dose|mg/ml|gm|capsule|pill|mls)?', '', line)
            cleaned_line = re.sub(r'[^\w\s]', '', cleaned_line)
            words = cleaned_line.split()

            for word in words:
                if len(word) > 4:
                    medicines.append(word.strip())
                    break  # Adding first matching medicine name
    return medicines

# Clean the OCR and CSV text to ignore numbers and extra information
def clean_text(text):
    cleaned_text = re.sub(r'\s*\(.*\)', '', text)  # Remove anything in parentheses
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # Remove digits
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation
    cleaned_text = cleaned_text.lower().strip()  # Normalize to lowercase and strip spaces
    return cleaned_text

# Function to calculate character overlap between two strings
def character_overlap(str1, str2):
    return len(set(str1.lower()) & set(str2.lower()))

# Efficient fuzzy matching using `difflib`
def get_closest_matches(medicine, all_fields, n=3, cutoff=0.5):  # Removed @lru_cache and its use of df
    matches = difflib.get_close_matches(medicine, all_fields, n=n, cutoff=cutoff)

    ranked_matches = []
    for match in matches:
        match_len = len(match)
        overlap_count = character_overlap(medicine, match)
        ranked_matches.append((match, match_len, overlap_count))

    ranked_matches.sort(key=lambda x: (-abs(len(medicine) - x[1]), -x[2]))

    return [match[0] for match in ranked_matches[:n]]

# Build a list of all relevant fields (for use in get_closest_matches)
def build_all_fields(df):
    return pd.concat([df['name'].apply(lambda x: str(x).lower()),
                      df['short_composition1'].apply(lambda x: str(x).lower()), 
                      df['short_composition2'].apply(lambda x: str(x).lower())], ignore_index=True).tolist()

# Trie Data Structure for storing and searching medicine names and compositions
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.suggestions = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, full_name):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.suggestions.append(full_name)

    def search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No match found
            node = node.children[char]
        return self._get_all_suggestions(node)

    def _get_all_suggestions(self, node):
        suggestions = []
        if node.is_end_of_word:
            suggestions.extend(node.suggestions)
        for child in node.children.values():
            suggestions.extend(self._get_all_suggestions(child))

        return suggestions

# Build Trie from the dataset
def build_medicine_trie(df):
    trie = Trie()
    for _, row in df.iterrows():
        name = str(row.get('name', '')).lower()
        composition = str(row.get('short_composition1', '')).lower()
        composition2 = str(row.get('short_composition2', '')).lower()

        # Insert both name and composition into the Trie
        trie.insert(name, name)
        trie.insert(composition, name)
        trie.insert(composition2, name)

    return trie

# Match medicine names using Trie (for close and partial matches)
def match_medicines_with_trie(medicine_list, trie, all_fields):
    matched_medicines = []

    for medicine in medicine_list:
        cleaned_medicine = clean_text(medicine.strip())
        suggestions = trie.search(cleaned_medicine)

        if suggestions:
            matched_medicines.append(f"Suggested Matches: {', '.join(suggestions)}")
        else:
            closest_matches = get_closest_matches(cleaned_medicine, all_fields, n=3, cutoff=0.5)
            if closest_matches:
                matched_medicines.append(f"Closest Matches: {', '.join(closest_matches)}")
            else:
                matched_medicines.append(f"No matches found for {medicine}")
    
    return matched_medicines

# Display medicine matches for pharmacist selection
def display_medicine_selection(matched_medicines, ocr_medicines):
    for medicine in matched_medicines:
        if "Suggested Matches" in medicine:
            options = medicine.replace('Suggested Matches: ', '').split(", ")
            options = sorted(options, key=lambda x: -len(set(x.lower()) & set(ocr_medicines.lower())))
            selected_option = st.radio("Select a Match", options[:3], key=medicine)
            if st.button(f"Add {selected_option} to Cart"):
                if 'cart' not in st.session_state:
                    st.session_state.cart = []
                st.session_state.cart.append(selected_option)
        elif "Closest Matches" in medicine:
            options = medicine.replace('Closest Matches: ', '').split(", ")
            selected_option = st.radio("Select a Closest Match", options[:3], key=medicine)
            if st.button(f"Add {selected_option} to Cart"):
                if 'cart' not in st.session_state:
                    st.session_state.cart = []
                st.session_state.cart.append(selected_option)
        else:
            st.write(medicine)
            if st.button(f"Add {medicine} to Cart"):
                if 'cart' not in st.session_state:
                    st.session_state.cart = []
                st.session_state.cart.append(medicine)

# Main OCR processing function (updated with selection step for pharmacist)
def ocr_processing(image, df, trie, all_fields):
    processed_img = preprocess_image(image)

    ocr_text = pytesseract.image_to_string(processed_img, config='--psm 6')
    medicines = extract_medicine_names(ocr_text)
    matched_medicines = match_medicines_with_trie(medicines, trie, all_fields)

    st.subheader("Matched Medicines:")
    display_medicine_selection(matched_medicines, ' '.join(medicines))

# Run the Streamlit app (updated to use Trie)
def run():
    st.title("MediMatcher : AI Powered Prescription Assistant")

    df = load_dataset()
    trie = build_medicine_trie(df)
    all_fields = build_all_fields(df)  # Preprocessed fields for caching

    img_file = st.file_uploader("Choose a Prescription Image", type=['png', 'jpg', 'jpeg'])
    
    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        os.makedirs('./Uploaded_Images', exist_ok=True)
        
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        img = PILImage.open(img_file)
        col1, col2 = st.columns([1, 2])  # Create two columns
        
        with col1:
            st.image(img, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            with st.spinner('Processing image...'):
                ocr_processing(save_image_path, df, trie, all_fields)
        
        if 'cart' in st.session_state and st.session_state.cart:
            st.header("🛒 **Cart**")
            for item in st.session_state.cart:
                st.write(item)

            if st.button("Checkout"):
                st.success("Successfully Checked Out!")
                st.session_state.cart.clear()  # Clear cart after checkout
        else:
            st.warning("No medicines found in the Cart.")

if __name__ == "__main__":
    run()
