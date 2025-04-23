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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/chat": {"origins": "*"}}, supports_credentials=True)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid.uuid4()))
Session(app)

doc_dir = "./documents"

# OpenAI + LlamaIndex setup
llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# General polite replies
GENERAL_REPLIES = {
    "hi": "Hello! I'm your assistant from the Civil Engineering Department of Twente University 😊",
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a bot, but I'm ready to help you!",
    "goodbye": "Goodbye! If you have more questions later, just ask 😊",
    "bye": "Bye! See you next time.",
    "thank you": "You're welcome!",
    "thanks": "Always happy to help!",
    "ok": "Got it! Let me know if you have more questions.",
    "yes": "Alright! What would you like to know more about?",
    "no": "Okay. Let me know if anything comes up."
}

def match_general_reply(cleaned_input):
    for key in GENERAL_REPLIES:
        if cleaned_input == key:
            return GENERAL_REPLIES[key]
        if difflib.SequenceMatcher(None, cleaned_input, key).ratio() > 0.87:
            return GENERAL_REPLIES[key]
    return None

def load_sentences():
    sentences = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(os.path.join(doc_dir, filename))
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
            except Exception as e:
                print("PDF sentence load error:", e)
    return sentences

def load_excel_schedule():
    for filename in os.listdir(doc_dir):
        if filename.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(doc_dir, filename), skiprows=5, engine="openpyxl")
                return df
            except Exception as e:
                print("Excel load error:", e)
    return pd.DataFrame()

def schedule_for_today():
    df = load_excel_schedule()
    if df.empty:
        return []

    today = datetime.now().strftime("%d-%m-%Y")
    df_today = df[df['Begin date'] == today]
    events = []
    for _, row in df_today.iterrows():
        start = row['Begin time']
        end = row['End time']
        course = row['Course/Description'] if pd.notnull(row['Course/Description']) else "Unknown Course"
        room = row['Hall'] if pd.notnull(row['Hall']) else "Unknown Room"
        lecturer = row['Lecturer'] if pd.notnull(row['Lecturer']) else "Unknown Lecturer"
        activity = row['Activity Type'] if pd.notnull(row['Activity Type']) else "Unknown Activity"
        event = f"{start}–{end} | {course} | {activity} | {room} | by {lecturer}"
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

        cleaned_input = unidecode.unidecode(user_input).strip("?!.").lower()
        now = datetime.now().strftime("%A, %d %B %Y %H:%M")

        # General replies
        reply = match_general_reply(cleaned_input)
        if reply:
            return jsonify({"response": reply})

        # Excel-based "today" query
        if "scheduled for today" in cleaned_input or "what is today’s schedule" in cleaned_input:
            events = schedule_for_today()
            return jsonify({"response": "\n".join(events) if events else "Nothing found"})

        # General ChatGPT fallback
        general_keywords = ["time", "date", "your name", "hello", "hi", "bye", "joke", "weather", "how are you"]
        if any(kw in cleaned_input for kw in general_keywords):
            result = ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a general assistant. The current date and time is: {now}."},
                    {"role": "user", "content": user_input}
                ],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            return jsonify({"response": result["choices"][0]["message"]["content"].strip()})

        # Fallback to documents
        clarified = user_input
        context_nodes = retriever.retrieve(clarified)
        node_texts = list(set([n.get_text() for n in context_nodes if n.get_text()]))

        if not node_texts or len(" ".join(node_texts)) < 50:
            node_texts += [find_relevant_sentences(clarified)]

        all_context = "\n".join(node_texts[:20]).strip()

        if not all_context or len(all_context) < 10:
            return jsonify({"response": "Nothing found"})

        result = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a document-based assistant for Civil Engineering students at Twente University.
Use ONLY the context below. Do not guess. Do not use general knowledge.
If multiple matches are found (e.g., events, lecturers), return them all.
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

# Initialize PDF + Excel documents
SENTENCES = load_sentences() + load_excel_schedule().astype(str).apply(" | ".join, axis=1).tolist()

# Index creation
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

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
