import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Tải biến môi trường
load_dotenv()

# --- 1. THIẾT LẬP CẤU HÌNH VÀ MODEL (Khởi tạo một lần) ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not found.")
    st.stop()

# Session ID cố định cho ví dụ đơn giản này
SESSION_ID = "stream_session_001"

# --- 2. THIẾT LẬP BỘ NHỚ VÀ CHAIN ---

# Khởi tạo session state cho lịch sử trò chuyện (dạng dictionary)
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = {}

# Khởi tạo session state cho hiển thị lịch sử trò chuyện (dạng list)
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Tạo hoặc trả về đối tượng ChatMessageHistory từ Streamlit session state."""
    if session_id not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_history_store[session_id]


# Định nghĩa Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Keep your answers concise and professional. Use the chat history to maintain context."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "Question: {question}")
])

# Khởi tạo Model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

# Tạo Core Chain
core_chain = prompt | llm

# Tạo History-Aware Chain
history_chain = RunnableWithMessageHistory(
    core_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# --- 3. HÀM XỬ LÝ CHÍNH ---

def generate_response(question, session_id):
    """Gọi chain có lịch sử trò chuyện."""
    config = {"configurable": {"session_id": session_id}}

    # LangChain sẽ tự động thêm và lấy lịch sử
    answer = history_chain.invoke({"question": question}, config=config)

    return answer.content


# --- 4. GIAO DIỆN STREAMLIT CHÍNH ---

st.set_page_config(page_title="Khung Chat 2 Phía", layout="wide")
st.title("💬 Chatbot with Groq")


# 1. Hiển thị lịch sử trò chuyện đã lưu trong st.session_state.messages
for message in st.session_state.messages:
    # Lấy vai trò (user hoặc assistant) để xác định vị trí chat
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Xử lý Input của người dùng
if prompt_input := st.chat_input("Nhập câu hỏi của bạn..."):

    # 2a. Thêm tin nhắn của User vào lịch sử hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # 2b. Hiển thị tin nhắn User ở khung bên phải
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 2c. Gọi LLM và hiển thị phản hồi của AI
    with st.chat_message("assistant"):
        with st.spinner("AI đang suy nghĩ..."):
            try:
                # Gọi hàm sinh phản hồi (hàm này có cả logic LangChain/Memory)
                response = generate_response(prompt_input, SESSION_ID)

                # In phản hồi của AI ra màn hình
                st.markdown(response)

                # 2d. Thêm tin nhắn của AI vào lịch sử hiển thị
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"Đã xảy ra lỗi: {e}. Vui lòng kiểm tra API key."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})