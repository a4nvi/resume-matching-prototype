import streamlit as st
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

# Initialize models
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
summarizer = pipeline('summarization')

# Helper Functions
def keyword_overlap(resume, description):
    resume_words = set(word_tokenize(resume.lower()))
    desc_words = set(word_tokenize(description.lower()))
    overlap = resume_words.intersection(desc_words)
    return len(overlap) / len(desc_words) * 100 if len(desc_words) > 0 else 0

def synonym_expansion(resume, description):
    resume_words = word_tokenize(resume.lower())
    expanded_resume = set(resume_words)
    for word in resume_words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_resume.add(lemma.name().lower())
    description_words = set(word_tokenize(description.lower()))
    overlap = expanded_resume.intersection(description_words)
    return len(overlap) / len(description_words) * 100 if len(description_words) > 0 else 0

def semantic_similarity(resume, description):
    embeddings = sbert_model.encode([resume, description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100

def summarization_similarity(resume, description):
    try:
        # Limit input size to prevent token limit errors
        if len(resume.split()) > 500:
            resume = ' '.join(resume.split()[:500])

        summary = summarizer(resume, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    except Exception as e:
        print(f"Summarizer error: {e}")
        summary = resume  # Fallback

    embeddings = sbert_model.encode([summary, description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100

def hybrid_scoring(k1, k2, k3, k4):
    return (0.2 * k1) + (0.3 * k2) + (0.3 * k3) + (0.2 * k4)

# Streamlit UI
st.title("📄 Resume vs Internship Matching System")

resume = st.text_area("Paste Resume Text", "John is a software engineering student skilled in Python, JavaScript, React, and Django.", height=150)
description = st.text_area("Paste Internship Description", "Looking for a Software Development Intern with experience in Python, JavaScript, and Django frameworks.", height=150)

if st.button("Match"):
    if not resume.strip() or not description.strip():
        st.error("Both fields are required!")
    else:
        st.info("Matching in progress...")

        level1 = keyword_overlap(resume, description)
        level2 = synonym_expansion(resume, description)
        level3 = semantic_similarity(resume, description)
        level4 = summarization_similarity(resume, description)
        final_score = hybrid_scoring(level1, level2, level3, level4)

        st.success("✅ Matching Complete!")
        st.write(f"**Level 1 - Keyword Overlap:** {level1:.2f}%")
        st.write(f"**Level 2 - Synonym Expansion:** {level2:.2f}%")
        st.write(f"**Level 3 - Semantic Similarity:** {level3:.2f}%")
        st.write(f"**Level 4 - Summarization Match:** {level4:.2f}%")
        st.write(f"### 🎯 Final Weighted Match Score: {final_score:.2f}%")
