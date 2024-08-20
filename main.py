import streamlit as st
import pdfplumber
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np

# Streamlit configuration
st.set_page_config(page_title="Policy Impact Analyzer", layout="wide")

# Sample policy text
SAMPLE_POLICY = """
Proposed Policy: Sustainable Agricultural Development and Digital Innovation Initiative

Our policy aims to modernize and sustain Australia's agricultural sector through the following key initiatives:

1. Digital Transformation in Agriculture:
   - Implement a nationwide program to provide high-speed internet access to all rural and remote farming communities.
   - Develop and promote the use of AI-driven farm management systems to optimize crop yields and resource usage.
   - Establish a national agricultural data platform to facilitate information sharing and decision-making among farmers, researchers, and policymakers.

2. Biosecurity Measures:
   - Enhance border control measures to prevent the introduction of invasive species and diseases.
   - Allocate funds for research into new detection technologies for potential biosecurity threats.
   - Conduct regular biosecurity awareness campaigns for farmers and the general public.

3. Sustainable Farming Practices:
   - Introduce incentives for farmers adopting regenerative agriculture techniques.
   - Promote crop diversification and rotation to improve soil health and reduce pest pressures.
   - Encourage the use of precision agriculture technologies to minimize water usage and chemical inputs.

4. Climate Resilience in Agriculture:
   - Develop climate-resistant crop varieties through increased funding for agricultural research.
   - Implement a carbon credit system for farmers who adopt practices that sequester carbon in soil.
   - Establish regional climate adaptation plans for different agricultural zones across Australia.

5. Pacific Region Collaboration:
   - Share agricultural best practices and technologies with Pacific Island nations to enhance their food security.
   - Collaborate on regional biosecurity initiatives to protect shared ecosystems.
   - Provide training and resources to Pacific farmers on sustainable and climate-smart agriculture techniques.

6. Biodiversity and Wildlife Conservation:
   - Create wildlife corridors in agricultural landscapes to support native species movement.
   - Implement a grant program for farmers who protect and restore native habitats on their properties.
   - Develop guidelines for wildlife-friendly farming practices, particularly in areas adjacent to protected habitats.

This policy seeks to balance technological advancement, environmental sustainability, and regional cooperation to ensure a resilient and productive agricultural sector for Australia's future.
"""

# Existing functions remain the same
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

def compare_policy_to_document(policy_text, document_text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([policy_text, document_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    feature_names = vectorizer.get_feature_names_out()
    policy_tfidf = tfidf_matrix[0].toarray()[0]
    doc_tfidf = tfidf_matrix[1].toarray()[0]

    top_policy_terms = [feature_names[i] for i in policy_tfidf.argsort()[-20:][::-1]]
    top_doc_terms = [feature_names[i] for i in doc_tfidf.argsort()[-20:][::-1]]

    common_terms = set(top_policy_terms) & set(top_doc_terms)

    return similarity, list(common_terms), top_policy_terms, top_doc_terms

def generate_impact_description(similarity, common_terms):
    if similarity > 0.3:
        impact_level = "High Alignment"
        description = "The proposed policy shows strong alignment with this strategy document."
    elif similarity > 0.2:
        impact_level = "Moderate Alignment"
        description = "The proposed policy shows notable alignment with this strategy document."
    elif similarity > 0.1:
        impact_level = "Low Alignment"
        description = "The proposed policy shows some alignment with this strategy document, but there are likely significant differences."
    else:
        impact_level = "Minimal Alignment"
        description = "The proposed policy shows little alignment with this strategy document."

    if common_terms:
        description += f" Key areas of potential alignment include: {', '.join(common_terms)}."
    else:
        description += " No specific areas of alignment were identified."

    description += " Consider how this alignment (or lack thereof) impacts the policy's effectiveness and comprehensiveness."

    return impact_level, description

def generate_impact_report(policy_text, document_similarities):
    report = "Policy Impact Analysis Report\n\n"
    report += f"Policy Text: {policy_text[:200]}...\n\n"
    report += "Impact on Strategy Documents:\n"
    for doc, (similarity, common_terms, top_policy_terms, top_doc_terms) in document_similarities.items():
        impact_level, impact_description = generate_impact_description(similarity, common_terms)
        report += f"- {doc}:\n"
        report += f"  Impact Level: {impact_level}\n"
        report += f"  Similarity Score: {similarity:.2f}\n"
        report += f"  Description: {impact_description}\n"
        report += f"  Key Aligned Terms: {', '.join(common_terms)}\n"
        report += f"  Top Policy Terms: {', '.join(top_policy_terms[:10])}\n"
        report += f"  Top Document Terms: {', '.join(top_doc_terms[:10])}\n\n"
    return report

def main():
    logo = Image.open("logo.png")
    st.image(logo, width=200)

    st.title("Agriculture, Fisheries and Forestry Policy Impact Analyzer")

    st.write("""
    Welcome to the Agriculture, Fisheries and Forestry Policy Impact Analyzer!

    This app analyzes the potential impact of a proposed policy on key strategy documents 
    in the agriculture, fisheries, and forestry sectors.

    Here's how it works:
    1. Select the strategy documents you want to include in the analysis.
    2. Enter your proposed policy text.
    3. The app compares your policy to the entire content of each selected document using natural language processing.
    4. An impact report is generated, showing how your policy might relate to each strategy document, including a detailed description of the potential impact and specific areas of alignment.

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

    st.subheader("Policy Analysis")
    st.write("You can use our sample policy or enter your own:")

    if st.button("Use Sample Policy"):
        st.session_state.policy_text = SAMPLE_POLICY

    policy_text = st.text_area("Enter your proposed policy text here:", value=st.session_state.get('policy_text', ''), height=300)

    if st.button("Analyze Policy"):
        if policy_text and selected_files:
            with st.spinner("Analyzing policy impact..."):
                document_similarities = {}
                for pdf_file in selected_files:
                    if os.path.exists(pdf_file):
                        document_text = extract_text_from_pdf(pdf_file)
                        similarity, common_terms, top_policy_terms, top_doc_terms = compare_policy_to_document(policy_text, document_text)
                        document_similarities[pdf_file] = (similarity, common_terms, top_policy_terms, top_doc_terms)
                    else:
                        st.warning(f"File not found: {pdf_file}")

                if document_similarities:
                    report = generate_impact_report(policy_text, document_similarities)
                    st.text_area("Impact Report", report, height=400)

                    # Visualization of impact
                    st.subheader("Impact Visualization")
                    for doc, (similarity, common_terms, top_policy_terms, top_doc_terms) in document_similarities.items():
                        st.write(f"{doc}:")
                        st.progress(similarity)
                        impact_level, _ = generate_impact_description(similarity, common_terms)
                        st.write(f"Impact Level: {impact_level}")
                        st.write(f"Key Aligned Terms: {', '.join(common_terms)}")
                        st.write(f"Top Policy Terms: {', '.join(top_policy_terms[:10])}")
                        st.write(f"Top Document Terms: {', '.join(top_doc_terms[:10])}")
                        st.write("---")
                else:
                    st.error("No documents could be processed. Please check the selected files and try again.")
        else:
            st.warning("Please enter policy text and select at least one document to analyze.")

    st.write("""
    Note: This analysis uses natural language processing to compare your policy text with 
    the content of the selected strategy documents. The results should be interpreted 
    as potential impacts and used as a starting point for further, more detailed analysis by domain experts.
    """)

if __name__ == "__main__":
    main()