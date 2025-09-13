import streamlit as st
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

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
    return len(overlap) / len(desc_words) * 100

def synonym_expansion(resume, description):
    resume_words = word_tokenize(resume.lower())
    expanded_resume = set(resume_words)
    
    for word in resume_words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_resume.add(lemma.name().lower())
                
    description_words = set(word_tokenize(description.lower()))
    overlap = expanded_resume.intersection(description_words)
    return len(overlap) / len(description_words) * 100

def semantic_similarity(resume, description):
    embeddings = sbert_model.encode([resume, description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100  # As percentage

def summarization_similarity(resume, description):
    try:
        summary = summarizer(resume, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    except Exception:
        summary = resume  # Fallback to original if summarizer fails
    
    embeddings = sbert_model.encode([summary, description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100  # As percentage

def hybrid_scoring(k1, k2, k3, k4):
    return (0.2 * k1) + (0.3 * k2) + (0.3 * k3) + (0.2 * k4)

# Streamlit UI
st.title("📄 Resume vs Internship Matching System")

resume = st.text_area("Paste Resume Text", height=200)
description = st.text_area("Paste Internship Description", height=200)

if st.button("Match"):
    if not resume or not description:
        st.error("Please provide both resume and description.")
    else:
        st.info("Processing...")

        level1 = keyword_overlap(resume, description)
        level2 = synonym_expansion(resume, description)
        level3 = semantic_similarity(resume, description)
        level4 = summarization_similarity(resume, description)
        final_score = hybrid_scoring(level1, level2, level3, level4)

        st.success("✅ Matching Complete!")
        st.write(f"**Level 1 - Keyword Overlap Score:** {level1:.2f}%")
        st.write(f"**Level 2 - Synonym Expansion Score:** {level2:.2f}%")
        st.write(f"**Level 3 - Semantic Similarity Score:** {level3:.2f}%")
        st.write(f"**Level 4 - Summarization Semantic Match:** {level4:.2f}%")
        st.write(f"### 🎯 Final Weighted Match Score: {final_score:.2f}%")
