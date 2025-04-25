from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_session import Session
from llama_index import VectorStoreIndex, ServiceContext, download_loader
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.schema import Document
from dotenv import load_dotenv
import os
import fitz
from openai import ChatCompletion
import re
import uuid
import difflib
import unidecode
from datetime import datetime
import pytz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")  # ✅ corrected __name__
CORS(app, resources={r"/chat": {"origins": "*"}}, supports_credentials=True)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid.uuid4()))
Session(app)

# Timezone
nl_timezone = pytz.timezone("Europe/Amsterdam")

# ChatGPT / Embedding setup
llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

doc_dir = "./documents"
GENERAL_REPLIES = {
    "hi": "Hello! I'm your assistant from the Civil Engineering Department of Twente University 😊",
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a bot, but I'm ready to help you!",
    "thank you": "You're welcome!",
    "thanks": "Always happy to help!",
    "bye": "Goodbye! If you have more questions later, just ask 😊"
}

FEEDBACK_TRIGGERS = ["feedback", "suggestion", "report", "comment"]

user_context_memory = {}

def match_general_reply(cleaned_input):
    for key in GENERAL_REPLIES:
        if cleaned_input == key:
            return GENERAL_REPLIES[key]
        if difflib.SequenceMatcher(None, cleaned_input, key).ratio() > 0.87:
            return GENERAL_REPLIES[key]
    return None

def resolve_pronouns(user_input, history):
    context = "\n".join(history[-6:])
    try:
        result = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Rewrite only the user's follow-up question using the conversation history to clarify vague references."},
                {"role": "user", "content": f"History:\n{context}\nFollow-up: {user_input}\nRewritten:"}
            ],
            api_key=os.getenv("OPENAI_API_KEY")
        )
        return result["choices"][0]["message"]["content"].strip()
    except:
        return user_input

def load_sentences():
    sentences = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(doc_dir, filename)
            doc = fitz.open(path)
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    if isinstance(block, tuple) and len(block) > 4:
                        text = block[4].strip()
                        for sentence in re.split(r'(?<=[.!?])\s+', text):
                            clean = sentence.strip()
                            if len(clean) > 40 or re.search(r"[0-9]{1,2}\s*EC", clean, re.IGNORECASE):
                                sentences.append(clean)
            doc.close()
    return sentences

def load_excel_schedule():
    for filename in os.listdir(doc_dir):
        if filename.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(doc_dir, filename), skiprows=5, engine="openpyxl")
                return df
            except:
                pass
    return pd.DataFrame()

def schedule_for_today():
    df = load_excel_schedule()
    if df.empty:
        return []
    today = datetime.now(nl_timezone).strftime("%d-%m-%Y")
    df_today = df[df['Begin date'] == today]
    events = []
    for _, row in df_today.iterrows():
        parts = [
            f"{row['Begin time']}–{row['End time']}" if pd.notnull(row['Begin time']) and pd.notnull(row['End time']) else "",
            row.get('Course/Description', ""),
            row.get('Activity Type', ""),
            row.get('Hall', ""),
            f"by {row['Lecturer']}" if pd.notnull(row.get('Lecturer')) else ""
        ]
        event = " | ".join([p for p in parts if p])
        events.append(event)
    return events

def find_relevant_sentences(query: str, max_hits=30):
    if not SENTENCES:
        return ""
    vectorizer = TfidfVectorizer().fit([query] + SENTENCES)
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(SENTENCES)
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    top = np.argsort(sims)[::-1][:max_hits]
    return "\n".join([SENTENCES[i] for i in top if sims[i] > 0.05])

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({"response": "No input provided"}), 400

        cleaned_input = unidecode.unidecode(user_input).strip("?!.: ").lower()
        session_id = request.remote_addr or str(uuid.uuid4())
        user_context_memory.setdefault(session_id, []).append(user_input)

        if any(trigger in cleaned_input for trigger in FEEDBACK_TRIGGERS):
            return jsonify({"response": "We’d love your feedback! Please email it to faima.jalali@utwente.nl 📩"})

        reply = match_general_reply(cleaned_input)
        if reply:
            return jsonify({"response": reply})

        if "scheduled for today" in cleaned_input or "today’s schedule" in cleaned_input:
            events = schedule_for_today()
            return jsonify({"response": "\n".join(events) if events else "Nothing found"})

        if "time" in cleaned_input or "date" in cleaned_input:
            current_time = datetime.now(nl_timezone).strftime("%H:%M on %A, %d %B %Y")
            return jsonify({"response": f"The current time is {current_time}."})

        clarified = resolve_pronouns(user_input, user_context_memory[session_id])
        context_nodes = retriever.retrieve(clarified)
        node_texts = list(set([n.get_text() for n in context_nodes if n.get_text()]))

        if not node_texts or len(" ".join(node_texts)) < 50:
            node_texts += [find_relevant_sentences(clarified)]

        all_context = "\n".join(node_texts[:20]).strip()

        if not all_context:
            return jsonify({"response": "Nothing found"})

        result = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a document-based assistant for Civil Engineering students at Twente University.
Use ONLY the context below. Do not guess. Do not use general knowledge.
Always return all relevant matches found in the context, even if the user uses singular phrasing.
If no relevant answer is found, reply: 'Nothing found'."""},
                {"role": "user", "content": f"Context:\n{all_context}\n\nQuestion: {clarified}"}
            ],
            api_key=os.getenv("OPENAI_API_KEY")
        )
        return jsonify({"response": result["choices"][0]["message"]["content"].strip()})

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"response": "An error occurred processing your message."}), 500

@app.route('/')
def home():
    return render_template("index.html")

# === Load documents at startup ===
SENTENCES = load_sentences() + load_excel_schedule().astype(str).apply(" | ".join, axis=1).tolist()

PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()
documents = []

for filename in os.listdir(doc_dir):
    if filename.endswith(".pdf"):
        documents.extend(loader.load(file_path=os.path.join(doc_dir, filename)))

for filename in os.listdir(doc_dir):
    if filename.endswith(".xlsx"):
        try:
            df = pd.read_excel(os.path.join(doc_dir, filename), skiprows=5, engine="openpyxl")
            for _, row in df.iterrows():
                content = "\n".join([f"{df.columns[i]}: {cell}" for i, cell in enumerate(row) if pd.notnull(cell)])
                if content.strip():
                    documents.append(Document(text=content.strip(), metadata={"source": filename, "type": "excel"}))
        except Exception as e:
            print("Excel doc load error:", e)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
retriever = index.as_retriever(similarity_top_k=50)

if __name__ == "__main__":   # ✅ corrected __main__
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
