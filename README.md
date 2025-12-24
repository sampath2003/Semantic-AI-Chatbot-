# Semantic Intent-Based Customer Support Chatbot with BERT Sentiment Analysis

This project implements an intelligent **e-commerce customer support chatbot** using a hybrid NLP and LLM-based architecture. The system identifies user intent through **semantic similarity using Sentence-BERT**, detects customer sentiment using a **BERT-based deep learning model**, and generates human-like responses using **GPT-4o-mini**.  
The chatbot is deployed using a **Streamlit-based chat interface** designed to resemble an iMessage-style conversation.

---

## Features

### Zero-shot Intent Classification (Sentence-BERT)
- No predefined intent labels required
- Intent inferred via semantic similarity
- Uses customer support utterances from the Bitext dataset

### Deep Learning Sentiment Analysis (BERT)
- Model used: `nlptown/bert-base-multilingual-uncased-sentiment`
- Classifies user sentiment into:
  - Negative
  - Neutral
  - Positive

### Natural Language Response Generation (GPT-4o-mini)
- Used only for response generation
- Considers:
  - User query
  - Top semantic matches
  - Inferred intent
  - Detected sentiment

### Streamlit Chat Interface
- User and bot chat bubbles
- Message timestamps
- Display of inferred intent and sentiment
- Expandable section showing top SBERT similarity matches

---

## 2. Install Dependencies
pip install -r requirements.txt

## 3. Configure OpenAI API Key

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here

## 4. Running the Chatbot type the below command in the terminal

streamlit run app.py


## 5. Open your browser and navigate to:

http://localhost:8501

