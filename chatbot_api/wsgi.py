from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
import logging
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.DEBUG)

CHROMA_DB_PATH = "chroma_db_1"
app = Flask(__name__)
app.secret_key = "your-secret-key" 
# Flask-Session configuration
app.config["SESSION_TYPE"] = "filesystem"  # Use filesystem-based sessions
app.config["SESSION_FILE_DIR"] = "./flask_sessions"  # Directory for session files
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)
CORS(app)
embeddings = OllamaEmbeddings(
    model="qwen2.5:3b",
)

try:
    retriever = Chroma(persist_directory=CHROMA_DB_PATH,embedding=embeddings,).as_retriever()
    logging.info("ChromaDB loaded successfully.")
except Exception as e:
    logging.error(f"Error loading ChromaDB: {e}")
    retriever = None 
    
memory = ConversationBufferMemory(memory_key="chat_history")
llm = OllamaLLM(model="qwen2.5:3b", temperature=0.4)

# Pull RAG Prompt Template from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")
print(prompt)
# Define RAG Chain
def format_docs(docs):
    """Format documents into a string with context."""
    return "\n\n".join(doc.page_content for doc in docs)

if retriever:
    print(retriever)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
else:
    logging.warning("RAG chain cannot be initialized because retriever is unavailable.")
    rag_chain = None

# Define standard Chatbot Chain with memory
template = """You are an intelligent assistant named Yassir, designed to assist a human user with any questions or tasks. 
Your primary goal is to provide clear, accurate, and helpful responses. 
While you can answer general questions on any topic, 
you are especially skilled at supporting the user's project related to COVID website analysis. 
Provide guidance, insights, or technical assistance as needed, tailoring your answers to the user's specific requirements.
 Keep your responses concise, 
relevant, and easy to understand.

{chat_history}
Human: {human_input}
Chatbot:"""
chat_prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)
llm_chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True, memory=memory)

# Route de base pour vérifier si l'API est en ligne
@app.route('/')
def home():
    return "Chatbot API is running."


# Helpers for memory serialization
def serialize_memory(memory_buffer):
    """Convert memory buffer into a JSON-compatible format."""
    return [
        {"content": msg.content, "type": "human" if isinstance(msg, HumanMessage) else "ai"}
        for msg in memory_buffer if isinstance(msg, (HumanMessage, AIMessage))
    ]


def deserialize_memory(serialized_buffer):
    """Convert serialized buffer back into a memory buffer."""
    return [
        HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"])
        for msg in serialized_buffer
    ]


# Route pour gérer les messages du chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("Memory state before processing user message:", memory)
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"error": "Message is required"}), 400


        # Restore memory from session
        if "memory" in session:
            try:
                memory.buffer = deserialize_memory(session["memory"])
                logging.debug("Restored memory.buffer from session.")
            except Exception as e:
                logging.error(f"Error restoring memory: {e}")
                memory.buffer = []

        app.logger.debug(f"User message received: {user_message}")
        if rag_chain and retriever:
            # Use RAG for query response if retriever is available
            response = "".join(chunk for chunk in rag_chain.stream(user_message))
        else:
            # Default to standard chatbot chain if no retriever
            response = llm_chain.predict(human_input=user_message)
        print("Memory buffer contents:", memory)


        # Save updated memory buffer to session
        try:
            session["memory"] = serialize_memory(memory.buffer)
            logging.debug(f"Serialized session memory after update: {session['memory']}")
        except Exception as e:
            logging.error(f"Error serializing memory to session: {e}")
            session["memory"] = []
            
        print("Memory buffer contents after processing:", memory)

        app.logger.debug(f"Response from model: {response}")
        return jsonify({"response": response})


    except Exception as e:
        app.logger.exception("An error occurred while processing the message.")
        return jsonify({"error": str(e)}), 500
    
@app.route('/history', methods=['GET'])
def history():
    try:
        print("Session memory content:", session.get("memory", []))

        if "memory" in session:
            memory_buffer = session["memory"]
            formatted_history = [
                {"message": msg, "type": "user" if msg["type"] == "human" else "bot"}
                for msg in memory_buffer
            ]
            return jsonify({"history": formatted_history})
        else:
            return jsonify({"history": []})
    except Exception as e:
        app.logger.exception("An error occurred while fetching history.")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
import logging
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.DEBUG)

CHROMA_DB_PATH = "chroma_db_1"
app = Flask(__name__)
app.secret_key = "your-secret-key" 
# Flask-Session configuration
app.config["SESSION_TYPE"] = "filesystem"  # Use filesystem-based sessions
app.config["SESSION_FILE_DIR"] = "./flask_sessions"  # Directory for session files
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)
CORS(app)
embeddings = OllamaEmbeddings(
    model="qwen2.5:3b",
)

try:
    retriever = Chroma(persist_directory=CHROMA_DB_PATH,embedding=embeddings,).as_retriever()
    logging.info("ChromaDB loaded successfully.")
except Exception as e:
    logging.error(f"Error loading ChromaDB: {e}")
    retriever = None 
    
memory = ConversationBufferMemory(memory_key="chat_history")
llm = OllamaLLM(
    model="qwen2.5:3b",
    base_url="https://51f6-2a04-cec0-1063-e8a3-1fdf-750-8134-ee1.ngrok.io"
)

# Pull RAG Prompt Template from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")
print(prompt)
# Define RAG Chain
def format_docs(docs):
    """Format documents into a string with context."""
    return "\n\n".join(doc.page_content for doc in docs)

if retriever:
    print(retriever)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
else:
    logging.warning("RAG chain cannot be initialized because retriever is unavailable.")
    rag_chain = None

# Define standard Chatbot Chain with memory
template = """You are an intelligent assistant named Yassir, designed to assist a human user with any questions or tasks. 
Your primary goal is to provide clear, accurate, and helpful responses. 
While you can answer general questions on any topic, 
you are especially skilled at supporting the user's project related to COVID website analysis. 
Provide guidance, insights, or technical assistance as needed, tailoring your answers to the user's specific requirements.
 Keep your responses concise, 
relevant, and easy to understand.

{chat_history}
Human: {human_input}
Chatbot:"""
chat_prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)
llm_chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True, memory=memory)

# Route de base pour vérifier si l'API est en ligne
@app.route('/')
def home():
    return "Chatbot API is running."


# Helpers for memory serialization
def serialize_memory(memory_buffer):
    """Convert memory buffer into a JSON-compatible format."""
    return [
        {"content": msg.content, "type": "human" if isinstance(msg, HumanMessage) else "ai"}
        for msg in memory_buffer if isinstance(msg, (HumanMessage, AIMessage))
    ]


def deserialize_memory(serialized_buffer):
    """Convert serialized buffer back into a memory buffer."""
    return [
        HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"])
        for msg in serialized_buffer
    ]


# Route pour gérer les messages du chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("Memory state before processing user message:", memory)
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"error": "Message is required"}), 400


        # Restore memory from session
        if "memory" in session:
            try:
                memory.buffer = deserialize_memory(session["memory"])
                logging.debug("Restored memory.buffer from session.")
            except Exception as e:
                logging.error(f"Error restoring memory: {e}")
                memory.buffer = []

        app.logger.debug(f"User message received: {user_message}")
        if rag_chain and retriever:
            # Use RAG for query response if retriever is available
            response = "".join(chunk for chunk in rag_chain.stream(user_message))
        else:
            # Default to standard chatbot chain if no retriever
            response = llm_chain.predict(human_input=user_message)
        print("Memory buffer contents:", memory)


        # Save updated memory buffer to session
        try:
            session["memory"] = serialize_memory(memory.buffer)
            logging.debug(f"Serialized session memory after update: {session['memory']}")
        except Exception as e:
            logging.error(f"Error serializing memory to session: {e}")
            session["memory"] = []
            
        print("Memory buffer contents after processing:", memory)

        app.logger.debug(f"Response from model: {response}")
        return jsonify({"response": response})


    except Exception as e:
        app.logger.exception("An error occurred while processing the message.")
        return jsonify({"error": str(e)}), 500
    
@app.route('/history', methods=['GET'])
def history():
    try:
        print("Session memory content:", session.get("memory", []))

        if "memory" in session:
            memory_buffer = session["memory"]
            formatted_history = [
                {"message": msg, "type": "user" if msg["type"] == "human" else "bot"}
                for msg in memory_buffer
            ]
            return jsonify({"history": formatted_history})
        else:
            return jsonify({"history": []})
    except Exception as e:
        app.logger.exception("An error occurred while fetching history.")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)

