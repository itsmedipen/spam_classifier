import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
snow = SnowballStemmer('english')

def transform_text(text):
    text = text.lower()#lower
    text = nltk.word_tokenize(text) #word tokenisation

    y = []

    for i in text:## removng special characters only selecting alpha numeric
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(snow.stem(i))
        
    return ' '.join(y)

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl','rb') as f:
    vector = pickle.load(f)

st.title("Safe Inbox")

# User input
user_input = st.text_area('Enter your message:',placeholder = 'Check messages')

if st.button("Predict",type= 'primary') and user_input.strip() != '':
    with st.spinner('Predicting...'):
        #pass to the transform text

        transform_sms = transform_text(user_input)
        # Transform the user input
        vector_input = vector.transform([transform_sms])

        
        # Make prediction
        prediction = model.predict(vector_input)
        
        # Show result
        if prediction[0] == 1:
            st.error("⚠️ Alert: This message is SPAM")
        else:
            st.success("✅ This message is SAFE")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 0.9em;
        color: gray;
    }
    </style>
    <div class="footer">
        🧑Developed by <strong>Dipen Sherpa</strong>&nbsp;|&nbsp; © 2025
    </div>
    """,
    unsafe_allow_html=True
)
