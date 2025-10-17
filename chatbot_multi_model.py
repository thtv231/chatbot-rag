import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
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

# Khởi tạo session state cho lịch sử trò chuyện (dạng dictionary)
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = {}

# Khởi tạo session state cho hiển thị lịch sử trò chuyện (dạng list)
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 2. HÀM QUẢN LÝ BỘ NHỚ ---

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Tạo hoặc trả về đối tượng ChatMessageHistory từ Streamlit session state."""
    if session_id not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_history_store[session_id]


# --- 3. HÀM KHỞI TẠO MODEL ĐỘNG ---

def initialize_llm(llm_choice, temperature, max_tokens):
    """Khởi tạo và trả về đối tượng LLM dựa trên lựa chọn của người dùng."""

    # Chuẩn hóa các tham số để truyền vào model
    config_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if llm_choice == "Groq (Llama 3.1 8B)":
        if not GROQ_API_KEY:
            st.warning("Thiếu GROQ_API_KEY. Vui lòng nhập khóa API.")
            return None
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, **config_params)

    elif llm_choice == "Gemini (2.5 Flash)":
        if not GEMINI_API_KEY:
            st.warning("Thiếu GEMINI_API_KEY. Vui lòng nhập khóa API.")
            return None
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, **config_params)

    elif llm_choice == "Ollama (gemma3)":
        # Ollama chạy cục bộ, không cần API key
        return ChatOllama(model="gemma3:1b", **config_params)

    return None


# --- 4. CẤU HÌNH GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Khung Chat Đa LLM", layout="wide")
st.title("🤖 Striker1.0")
st.caption("Chọn model, điều chỉnh tham số, và bắt đầu trò chuyện.")

# Sidebar để lựa chọn Model và Tham số
with st.sidebar:
    st.header("Cấu hình Model")

    llm_choice = st.selectbox(
        "Chọn Model LLM:",
        ("Groq (Llama 3.1 8B)", "Gemini (2.5 Flash)", "Ollama (gemma3)")
    )

    st.subheader("Điều chỉnh Tham số")

    # Các tham số cho tất cả các model
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=2048, value=512, step=50)

    st.markdown("---")
    st.info("Lưu ý: Điều chỉnh Temperature, Max tokens để tăng sáng tạo, độ dài câu trả lời .")

# --- 5. HÀM CHẠY CHAIN VÀ GIAO DIỆN CHÍNH ---

# 1. Khởi tạo LLM và Chain mỗi khi tham số thay đổi
llm = initialize_llm(llm_choice, temperature, max_tokens)

if llm:
    # Định nghĩa Prompt Template (chỉ cần làm một lần)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a helpful assistant running on {llm_choice}. Keep your answers concise and use the chat history to maintain context."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Question: {question}")
    ])

    # Tạo Core Chain và History Chain
    core_chain = prompt | llm
    history_chain = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 2. Hiển thị lịch sử trò chuyện đã lưu
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Xử lý Input của người dùng
    if prompt_input := st.chat_input("Nhập câu hỏi của bạn..."):

        # Thêm tin nhắn User vào lịch sử hiển thị
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Gọi LLM và hiển thị phản hồi của AI
        with st.chat_message("assistant"):
            with st.spinner(f"Đang hỏi {llm_choice}..."):
                try:
                    # Gọi hàm generate_response trực tiếp bằng chain đã tạo
                    config = {"configurable": {"session_id": SESSION_ID}}
                    response = history_chain.invoke({"question": prompt_input}, config=config).content

                    st.markdown(response)

                    # Thêm tin nhắn của AI vào lịch sử hiển thị
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Đã xảy ra lỗi khi gọi {llm_choice}. Chi tiết: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.warning("Vui lòng thiết lập khóa API hoặc đảm bảo Ollama đang chạy để sử dụng chatbot.")