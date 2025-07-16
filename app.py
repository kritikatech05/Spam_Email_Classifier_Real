import streamlit as st
import joblib
from PIL import Image

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📬 Spam Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Paste an email or message below to check if it's spam 🚫</p>", unsafe_allow_html=True)
st.markdown("---")

# Text input area
email_text = st.text_area("📨 Enter email content here:", height=200, placeholder="e.g. Congratulations! You've won a free ticket to Maldives...")

# Predict button
if st.button("🔍 Predict"):
    if not email_text.strip():
        st.warning("⚠️ Please enter some email content.")
    else:
        X_input = vectorizer.transform([email_text])
        result = model.predict(X_input)[0]
        if result == 1:
            st.error("🚫 This is SPAM!")
        else:
            st.success("✅ This is NOT spam.")

# Sample examples
with st.expander("📄 Try Sample Emails"):
    st.code("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!", language="text")
    st.code("Hey John, just wanted to confirm our meeting at 3PM tomorrow.", language="text")

# Footer
st.markdown("---")
st.markdown("<small style='color: gray;'>Built with ❤️ by Kritika Sharma | Internship Project @ Celebal Technologies</small>", unsafe_allow_html=True)
