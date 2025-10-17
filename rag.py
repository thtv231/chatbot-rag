import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

# LangChain RAG/Embedding Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain Chat Model/Core Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

# Tải biến môi trường
load_dotenv()

# --- 1. THIẾT LẬP CẤU HÌNH VÀ HẰNG SỐ ---

SESSION_ID = "multi_llm_session"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Tên model Ollama được chọn
OLLAMA_CHAT_MODEL = "gemma:2b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Khởi tạo session state
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# --- 2. HÀM QUẢN LÝ BỘ NHỚ VÀ RAG ---

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Tạo hoặc trả về đối tượng ChatMessageHistory từ Streamlit session state."""
    if session_id not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_history_store[session_id]


def process_uploaded_file(uploaded_file):
    """Tải, tách và tạo FAISS Vector Store từ file PDF bằng Ollama Embeddings."""

    # Ghi file tạm thời
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        # 1. Load và Split tài liệu
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        # Sử dụng RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 2. Tạo Embeddings bằng OLLAMA
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success(f"Tải lên và xử lý file PDF thành công! Đã dùng {OLLAMA_EMBEDDING_MODEL} cho Embeddings.")
        return vector_store

    except Exception as e:
        st.error(
            f"Lỗi khi xử lý file PDF bằng Ollama: {e}. Đảm bảo Ollama đang chạy và model '{OLLAMA_EMBEDDING_MODEL}' đã được pull.")
        return None
    finally:
        os.remove(temp_file_path)


# --- 3. HÀM KHỞI TẠO MODEL ĐỘNG ---

def initialize_llm(llm_choice, temperature, max_tokens):
    """Khởi tạo và trả về đối tượng LLM dựa trên lựa chọn của người dùng."""

    config_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if llm_choice == "Groq (Llama 3.1 8B)":
        if not GROQ_API_KEY: return None
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, **config_params)

    elif llm_choice == "Gemini (2.5 Flash)":
        if not GEMINI_API_KEY: return None
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, **config_params)

    elif llm_choice == f"Ollama ({OLLAMA_CHAT_MODEL})":
        # Sử dụng model gemma:2b cho Chat
        return ChatOllama(model=OLLAMA_CHAT_MODEL, **config_params)

    return None


# --- 4. CẤU HÌNH GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Khung Chat RAG Đa LLM", layout="wide")
st.title("📚 Chatbot RAG Đa Mô Hình (PDF Q&A)")
st.caption("Chọn model, tải tài liệu PDF (sử dụng Ollama Embeddings), và bắt đầu trò chuyện.")

# Sidebar để lựa chọn Model, Tham số và Tải file
with st.sidebar:
    st.header("Cấu hình Model")

    llm_choice = st.selectbox(
        "Chọn Model LLM:",
        ("Groq (Llama 3.1 8B)", "Gemini (2.5 Flash)", f"Ollama ({OLLAMA_CHAT_MODEL})")
    )

    st.subheader("Điều chỉnh Tham số")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=4096, value=1024, step=50)

    st.markdown("---")
    st.header("Tài liệu RAG (Ollama Embeddings)")
    st.caption(f"Embeddings: **{OLLAMA_EMBEDDING_MODEL}**")

    uploaded_file = st.file_uploader("Tải lên PDF của bạn:", type="pdf")

    if uploaded_file:
        if st.button("Xử lý File PDF"):
            st.session_state.vector_store = process_uploaded_file(uploaded_file)

    # Hiển thị trạng thái RAG
    if st.session_state.vector_store:
        st.success("✅ RAG Active: Sẵn sàng hỏi về tài liệu!")
        st.caption(f"Vector Store đã lưu trữ {len(st.session_state.vector_store.docstore._dict)} chunks.")
    else:
        st.info("RAG Inactive: Chatbot hoạt động ở chế độ thường.")

# --- 5. TẠO CHAIN TÙY THUỘC VÀO TRẠNG THÁI RAG ---

llm = initialize_llm(llm_choice, temperature, max_tokens)
history_chain = None

if llm:
    # 1. Tạo Prompt cơ bản
    if st.session_state.vector_store:
        # Prompt cho RAG
        # CHỈ SỬ DỤNG {context} VÀ {input} để giải quyết lỗi truyền biến
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer the user's question only based on the retrieved context below. The question has been rephrased based on chat history. If you can't find the answer, state that clearly.\n\nContext: {context}"),
            ("user", "Question: {input}")
        ])

        # Tạo History-Aware Retriever (Sử dụng {history} và {input} ở đây)
        history_aware_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("user", "Question: {input}"),
            ("user",
             "Given the above conversation, generate a search query to look up in the documents to answer the latest user question.")
        ])

        retriever = st.session_state.vector_store.as_retriever()
        history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_prompt)
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        history_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    else:
        # Prompt cho chế độ Chat thường (Giữ nguyên)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a helpful assistant running on {llm_choice}. Keep your answers concise and use the chat history to maintain context."),
            MessagesPlaceholder(variable_name="history"),
            ("user", "Question: {input}")
        ])

        core_chain = prompt_template | llm
        history_chain = RunnableWithMessageHistory(
            core_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

# --- 6. GIAO DIỆN CHÍNH (INVOKE) ---

# if llm:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     if prompt_input := st.chat_input("Nhập câu hỏi của bạn..."):
#
#         st.session_state.messages.append({"role": "user", "content": prompt_input})
#         with st.chat_message("user"):
#             st.markdown(prompt_input)
#
#         with st.chat_message("assistant"):
#             with st.spinner(f"Đang hỏi {llm_choice}..."):
#                 try:
#                     config = {"configurable": {"session_id": SESSION_ID}}
#
#                     if st.session_state.vector_store:
#                         # RAG chain trả về dictionary (gồm answer và context)
#                         # Truyền input dưới dạng {"input": prompt_input}
#                         result = history_chain.invoke({"input": prompt_input}, config=config)
#                         response = result["answer"]
#                     else:
#                         # Simple chat chain trả về AIMessage
#                         # Truyền input dưới dạng {"input": prompt_input}
#                         response = history_chain.invoke({"input": prompt_input}, config=config).content
#
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#
#                 except Exception as e:
#                     error_msg = f"Đã xảy ra lỗi: {e}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})
#
# else:
#     st.warning("Vui lòng thiết lập khóa API và chọn model hợp lệ để sử dụng.")
if llm:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Nhập câu hỏi của bạn..."):

        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner(f"Đang hỏi {llm_choice}..."):
                try:
                    config = {"configurable": {"session_id": SESSION_ID}}

                    if st.session_state.vector_store:
                        # RAG chain trả về dictionary (gồm answer và context)
                        result = history_chain.invoke({"input": prompt_input}, config=config)
                        response = result["answer"]
                        context = result.get("context", [])  # Lấy Context Documents

                        # --- HIỂN THỊ RAG NÂNG CAO ---

                        # 1. Hiển thị câu trả lời chính
                        st.info("💡 **PHẢN HỒI RAG DỰA TRÊN TÀI LIỆU:**")
                        st.markdown(response)

                        # 2. Hiển thị nguồn trích dẫn trong Expander
                        if context:
                            with st.expander("📚 Nguồn Tài liệu (Contexts) - Bấm để xem"):
                                for i, doc in enumerate(context):
                                    source_page = doc.metadata.get('page', 'N/A')
                                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown File'))

                                    st.markdown(
                                        f"**Chunk {i + 1}** từ **File:** `{source_name}` (Trang: {source_page})")
                                    # Hiển thị nội dung đoạn trích
                                    st.code(doc.page_content[:300] + "...")
                                    st.markdown("---")
                        else:
                            st.warning("Không tìm thấy thông tin liên quan trong tài liệu được tải lên.")

                    else:
                        # Simple chat chain trả về AIMessage
                        response = history_chain.invoke({"input": prompt_input}, config=config).content
                        st.markdown(response)  # Chat thường dùng markdown đơn giản

                    # Lưu phản hồi vào lịch sử chat hiển thị (session state)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Đã xảy ra lỗi: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.warning("Vui lòng thiết lập khóa API và chọn model hợp lệ để sử dụng.")