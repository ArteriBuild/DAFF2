import streamlit as st
import pdfplumber
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_strategy_elements(text):
    elements = {
        'Goals': re.findall(r'(?:goal|objective)[:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE),
        'Priorities': re.findall(r'(?:priority|key focus area)[:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE),
        'Actions': re.findall(r'(?:action|initiative|measure)[:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE),
        'Principles': re.findall(r'(?:principle|core value)[:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE),
        'Approaches': re.findall(r'(?:approach|strategy|method)[:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    }
    return elements

def compare_policy_to_elements(policy_text, elements):
    vectorizer = TfidfVectorizer()
    all_elements = [item for sublist in elements.values() for item in sublist]
    all_text = [policy_text] + all_elements
    tfidf_matrix = vectorizer.fit_transform(all_text)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

def generate_impact_report(policy_text, elements, similarities):
    report = "Policy Impact Analysis Report\n\n"
    report += f"Policy Text: {policy_text[:200]}...\n\n"

    similarity_index = 0
    for category, items in elements.items():
        report += f"{category}:\n"
        for item in items:
            impact = "High" if similarities[similarity_index] > 0.5 else "Medium" if similarities[similarity_index] > 0.3 else "Low"
            report += f"- {item}\n  Impact: {impact} (Similarity: {similarities[similarity_index]:.2f})\n"
            similarity_index += 1
        report += "\n"

    return report

def main():
    logo = Image.open("logo.png")
    st.image(logo, width=200)

    st.title("DAFF Policy Impact Analyser")

    st.write("""
    Welcome to the DAFF Policy Impact Analyser!

    This app analyses the potential impact of a proposed policy on key elements 
    (goals, priorities, actions, principles, and approaches) extracted from important 
    strategy documents in the agriculture, fisheries, and forestry sectors.

    Here's how it works:
    1. Select the strategy documents you want to include in the analysis.
    2. Click the 'Extract Strategy Elements' button to process the selected documents.
    3. Enter your proposed policy text.
    4. The app compares your policy to the extracted elements using natural language processing.
    5. An impact report is generated, showing how your policy might affect each strategy element.

    Let's get started!
    """)

    pdf_files = [
        "Digital Foundations Agriculture Strategy.pdf",
        "National Biosecurity Strategy.pdf",
        "Pacific Biosecurity Strategy.pdf",
        "Threatened Species Strategy.pdf"
    ]

    st.subheader("Select Reference Strategy Documents")
    st.write("Choose the documents you want to include in the analysis:")

    selected_files = []
    for pdf_file in pdf_files:
        if st.checkbox(pdf_file, value=True):
            selected_files.append(pdf_file)

    if not selected_files:
        st.warning("Please select at least one document for analysis.")
        return

    if st.button("Extract Strategy Elements"):
        with st.spinner("Extracting strategy elements from selected documents... This may take a moment."):
            all_elements = {
                'Goals': [], 'Priorities': [], 'Actions': [], 'Principles': [], 'Approaches': []
            }
            for pdf_file in selected_files:
                if os.path.exists(pdf_file):
                    text = extract_text_from_pdf(pdf_file)
                    elements = extract_strategy_elements(text)
                    for key in all_elements:
                        all_elements[key].extend(elements[key])
                else:
                    st.warning(f"File not found: {pdf_file}")

        total_elements = sum(len(v) for v in all_elements.values())
        if total_elements > 0:
            st.success(f"Successfully processed {len(selected_files)} document(s) and extracted {total_elements} strategy elements.")

            st.session_state.all_elements = all_elements

            st.subheader("Policy Analysis")
            policy_text = st.text_area("Enter your proposed policy text here:")

            if st.button("Analyze Policy"):
                if policy_text and st.session_state.all_elements:
                    with st.spinner("Analyzing policy..."):
                        similarities = compare_policy_to_elements(policy_text, st.session_state.all_elements)
                        report = generate_impact_report(policy_text, st.session_state.all_elements, similarities)
                        st.text_area("Impact Report", report, height=400)
                else:
                    st.warning("Please enter policy text to analyze.")
        else:
            st.error("No strategy elements were extracted from the selected documents. Please try selecting different documents or refine the extraction method.")

    st.write("""
    Note: This analysis uses natural language processing to compare your policy text with 
    strategy elements extracted from the selected documents. The results should be interpreted 
    as potential impacts and used as a starting point for further, more detailed analysis by domain experts.
    """)

if __name__ == "__main__":
    main()