import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


# CONFIG: Load API Key

client = OpenAI(api_key="YOUR_API_KEY")


# LOAD EVERYTHING (SBERT + SENTIMENT + DATA)

@st.cache_resource
def load_all():
    # Load knowledge base
    df = pd.read_csv("utterances_clean.csv")       # merged dataset created earlier
    embeddings = np.load("utterance_embeddings.npy")

    # Load SBERT model
    sbert = SentenceTransformer("all-mpnet-base-v2")

    # Load SAFE sentiment model (safetensors)
    sent_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    return df, embeddings, sbert, sent_tokenizer, sent_model


df, embeddings, sbert_model, sent_tokenizer, sent_model = load_all()


# SENTIMENT ANALYSIS

def detect_sentiment(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = sent_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)[0]
    labels = ["negative", "positive"]
    sentiment = labels[torch.argmax(scores)]
    return sentiment.capitalize()


# SEMANTIC SEARCH (SBERT)

def semantic_search(query, top_k=3):
    query_emb = sbert_model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_emb, embeddings)[0]

    best_idx = np.argsort(scores)[::-1][:top_k]
    results = []

    for idx in best_idx:
        results.append({
            "utterance": df.loc[idx, "utterance"],
            "category": df.loc[idx, "category"],       # intent comes from category
            "score": float(scores[idx])
        })

    return results


# GPT RESPONSE REWRITER

def rewrite_with_gpt(user_query, samples, intent, sentiment):
    examples = "\n".join([f"- {x['utterance']}" for x in samples])

    prompt = f"""
You are a helpful and friendly customer support AI.

User Message:
{user_query}

Detected Intent (semantic category): {intent}
User Sentiment: {sentiment}

Use these semantically similar examples to understand the meaning:
{examples}

Now generate a helpful, concise, natural response.
If sentiment is negative, respond with extra empathy.
Avoid mentioning that you used examples or categories.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# MAIN CHATBOT PIPELINE

def chatbot_response(user_message):
    # 1. Detect sentiment using DistilBERT
    sentiment = detect_sentiment(user_message)

    # 2. Find top SBERT semantic matches
    similar = semantic_search(user_message)
    
    # 3. Derive semantic intent from the closest example
    intent = similar[0]["category"] if similar else "Unknown"

    # 4. Generate natural response using GPT (rewriter)
    reply = rewrite_with_gpt(
        user_message,
        similar,       # SBERT context
        intent,
        sentiment
    )

    # Return all values (4 total)
    return reply, intent, sentiment, similar



# STREAMLIT UI

st.set_page_config(page_title="Semantic Support Chatbot", layout="centered")

st.markdown("<h1>🤖 Smart Semantic Customer Support Bot</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def add_message(role, content, intent=None, sentiment=None, context=None):
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "intent": intent,
        "sentiment": sentiment,
        "context": context,   # <–– NEW
        "time": datetime.now().strftime("%I:%M %p")
    })


# Display chat history
# ------------------------------
# DISPLAY CHAT HISTORY (iMessage BLUE & GRAY)
# ------------------------------
for msg in st.session_state["messages"]:
    # USER MESSAGE (Blue bubble on right)
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; margin:6px 0;'>
                <div style='background:#0A84FF; color:white; padding:12px 16px;
                            border-radius:18px; max-width:70%; 
                            font-size:15px; line-height:1.45;
                            box-shadow:0px 1px 3px rgba(0,0,0,0.20);'>
                    {msg['content']}
                </div>
            </div>

            <div style='text-align:right; font-size:10px; color:#bbb; margin-right:6px;'>
                {msg['time']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # BOT MESSAGE (Blue & Grey iMessage style + Context Panel)
    else:
        # Bot bubble
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-start; margin:6px 0;'>
                <div style='background:#E5E5EA; color:black; padding:12px 16px;
                            border-radius:18px; max-width:70%;
                            font-size:15px; line-height:1.45;
                            box-shadow:0px 1px 3px rgba(0,0,0,0.20);'>
                    🤖 {msg['content']}
                    <div style='font-size:11px; color:#555; margin-top:8px;'>
                        Intent: <b>{msg['intent']}</b> |
                        Sentiment: <b>{msg['sentiment']}</b>
                    </div>
                </div>
            </div>

            <div style='text-align:left; font-size:10px; color:#bbb; margin-left:6px;'>
                {msg['time']}
            </div>
            """,
            unsafe_allow_html=True
        )

        # CONTEXT PANEL (SBERT matches + scores)
        with st.expander("🔍 Show Context Used"):
            st.write("### SBERT Top Matches")
            for i, ctx in enumerate(msg.get("context", [])):
                st.write(
                    f"**Match {i+1}**\n"
                    f"- Utterance: *{ctx['utterance']}*\n"
                    f"- Category (Intent Source): `{ctx['category']}`\n"
                    f"- Similarity Score: **{ctx['score']:.4f}**\n"
                )

            st.write("### Interpretation")
            st.write(f"- **Semantic Intent:** `{msg['intent']}`")
            st.write(f"- **Sentiment:** `{msg['sentiment']}`")

            st.caption(
                "These results come from SBERT embeddings + DistilBERT sentiment classifier. "
                "GPT only rewrites the final answer using this context."
            )



# User input
user_query = st.text_input("Type your message:")

if st.button("Send"):
    add_message("user", user_query)

    with st.spinner("🤖 Processing..."):
        reply, intent, sentiment, context = chatbot_response(user_query)

    add_message("bot", reply, intent=intent, sentiment=sentiment, context=context)

    st.rerun()
