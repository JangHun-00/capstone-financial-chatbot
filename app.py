import os
import streamlit as st
import urllib.parse
import pickle
import glob
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

HISTORY_DIR = "./.chat_history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

st.set_page_config(
    page_title="Financial Helper Chatbot",
    page_icon="💰",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }
    
    [data-testid="stChatMessage"] {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stChatInput"] {
        max-width: 900px;
        margin: 0 auto;
    }
    
    [data-testid="stSidebarHeader"] {
        margin-top: 1rem;
        margin-bottom: 0rem;
        height: auto;
    }
    
    [data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="[1/2] LLM 모델을 준비하는 중...")
def get_llm(api_key: str, model: str):
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.2)

@st.cache_resource(show_spinner="[2/2] 저장된 금융 문서를 로드하는 중...")
def load_vectorstore(api_key: str):
    VECTORSTORE_PATH = "faiss_index"
    embeddings = OpenAIEmbeddings(api_key=api_key)
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"벡터 스토어 로드 실패: {e}")
        return None

def get_rag_chain(api_key):
    llm = get_llm(api_key, "gpt-4o-mini")
    vectorstore = load_vectorstore(api_key)
    
    if not vectorstore:
        return None

    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "너는 금융 정보를 설명하는 전문 도우미야. 너의 답변은 다음 규칙을 따라야 해:\n\n"
        "--- 답변 규칙 ---\n"
        "1. **(정의 검색)**: 먼저 '참고 자료'({context})에서 질문받은 용어의 **사전적 정의**가 있는지(주로 'a_*.txt' 파일) 확인해.\n"
        "2. **(Case 1: 정의 있음)**: 만약 '참고 자료'에 사전적 정의가 있다면:\n"
        "   가. 해당 정의를 바탕으로 **핵심 답변**을 해. [cite]를 꼭 달아줘.\n"
        "   나. '참고 자료'에 추가적인 **유용한 정보**(꿀팁, 예시 등, 주로 'c_*.pdf' 파일)가 있다면 '유용한 추가 정보:' 섹션을 만들어 요약해줘.\n"
        "3. **(Case 2: 정의 없음)**: 만약 '참고 자료'에 사전적 정의가 **없다면** (예: '적금'):\n"
        "   가. 너의 **일반 지식**을 사용해 해당 용어의 정의를 설명해.\n"
        "   나. **하지만**, '참고 자료'에 그 용어와 관련된 **유용한 정보**(꿀팁, 예시 등)가 검색되었다면, '참고 자료에서 찾은 유용한 정보:' 섹션을 만들어 반드시 요약해줘.\n"
        "4. **(무관한 자료)**: 만약 '참고 자료'가 질문과 **전혀 무관**하다면, 무시하고 너의 일반 지식으로만 답해.\n"
        "5. **(필수 안내)**: 모든 답변 마지막에는 항상 아래의 안내 문구를 추가해.\n"
        "   '※ 본 정보는 참고용 일반 설명입니다. 실제 투자/세무/법률 판단은 공신력 있는 최신 자료와 전문가 상담을 권장합니다.'\n"
        "--- 참고 자료 ---\n"
        "{context}"
        "---"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def get_session_id():
    """현재 세션의 고유 ID를 반환 (없으면 생성)"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return st.session_state.session_id

def save_chat_history():
    """현재 채팅 기록을 로컬 파일(pickle)로 저장 (제목 포함)"""
    session_id = get_session_id()
    file_path = os.path.join(HISTORY_DIR, f"chat_{session_id}.pkl")
    
    title = "새로운 대화"
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if len(msg["content"]) > 15:
                title = msg["content"][:12] + "..."
            else:
                title = msg["content"][:15]
            break

    data = {
        "title": title,
        "messages": st.session_state.messages
    }
    
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        st.error(f"대화 저장 중 오류 발생: {e}")

def load_chat_history(filename):
    file_path = os.path.join(HISTORY_DIR, filename)
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        if isinstance(data, list):
            st.session_state.messages = data
        else:
            st.session_state.messages = data["messages"]
            
        session_id = filename.replace("chat_", "").replace(".pkl", "")
        st.session_state.session_id = session_id
        
        return True
    except Exception as e:
        st.error(f"대화 불러오기 실패: {e}")
        return False

def get_history_list():
    """저장된 채팅 파일 목록 반환 (최신순, 제목 포함)"""
    files = glob.glob(os.path.join(HISTORY_DIR, "chat_*.pkl"))
    files.sort(reverse=True)
    
    history_data = []
    for f in files:
        filename = os.path.basename(f)
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)
                if isinstance(data, list):
                    title = "저장된 대화 (구버전)"
                else:
                    title = data.get("title", "제목 없음")
                
                history_data.append({"filename": filename, "title": title})
        except:
            continue
            
    return history_data

def start_new_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요 👋 금융 궁금증을 해결해 드릴게요! 어떤 걸 도와드릴까요?"}
    ]
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요 👋 금융 궁금증을 해결해 드릴게요! 어떤 걸 도와드릴까요?"}
        ]
    if "session_id" not in st.session_state:
        get_session_id()


@st.dialog("⚠️ 대화 삭제 확인")
def delete_dialog():
    st.write("현재 대화 기록을 영구적으로 삭제하시겠습니까?")
    st.caption("삭제된 데이터는 복구할 수 없습니다.")
    
    col1, col2 = st.columns(2)
    
    if col1.button("취소", use_container_width=True):
        st.rerun()
        
    if col2.button("삭제", type="primary", use_container_width=True):
        session_id = st.session_state.get("session_id")
        if session_id:
            file_path = os.path.join(HISTORY_DIR, f"chat_{session_id}.pkl")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.messages = [
                        {"role": "assistant", "content": "안녕하세요 👋 금융 궁금증을 해결해 드릴게요! 어떤 걸 도와드릴까요?"}
                    ]
                    st.rerun()
                except Exception as e:
                    st.error(f"오류 발생: {e}")

@st.dialog("📢 이용안내 및 면책조항")
def show_disclaimer():
    st.info(
        """
        **1. 정보의 출처**

        본 챗봇은 금융감독원 금융소비자보호 포털(FINE) 및 금융꿀팁 200선 게시판의 공개 자료를 기반으로 답변합니다.
        
        **2. 법적 효력 부재**

        생성형 AI의 특성상 답변에 부정확한 내용이 포함될 수 있으며, 이는 **법적 효력이 있는 금융 상담이나 투자 조언이 아닙니다.**
        
        **3. 면책 조항**

        본 서비스의 정보를 활용하여 발생한 투자 결과 및 법적 분쟁에 대해 서비스 제공자는 책임을 지지 않습니다. 중요 의사결정 시 반드시 해당 금융사나 전문가와 교차 확인하시기 바랍니다.
        """
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        default_key = ""
        try:
            default_key = st.secrets.get("openai_api_key", "")
        except Exception:
            pass
        if not default_key:
            default_key = os.getenv("OPENAI_API_KEY", "")
        if "api_key" not in st.session_state:
            st.session_state.api_key = default_key

        is_expanded = False
        with st.expander("🔐 API Key 설정", expanded=is_expanded):
            with st.form("api-key-form", clear_on_submit=False):
                api_key_input = st.text_input("API Key 입력", type="password")
                if st.form_submit_button("저장"):
                    st.session_state.api_key = api_key_input.strip()
                    st.rerun()

        st.markdown("### 🗂️ 대화 기록")
        
        if st.button("➕ 새 대화 시작", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()
        
        history_list = get_history_list()
        
        if not history_list:
            st.caption("아직 대화 기록이 없습니다.")
        else:
            for item in history_list:
                filename = item["filename"]
                title = item["title"]
                is_current = (st.session_state.get("session_id") in filename)
                
                if st.button(title, key=filename, use_container_width=True):
                    if not is_current:
                        if load_chat_history(filename):
                            st.rerun()

        st.markdown("---")
        
        if st.button("🗑️ 현재 대화 삭제", use_container_width=True):
            session_id = st.session_state.get("session_id")
            
            if session_id:
                file_path = os.path.join(HISTORY_DIR, f"chat_{session_id}.pkl")

                if os.path.exists(file_path):
                    delete_dialog()

        if st.button("ℹ️ 이용안내 및 면책조항", use_container_width=True):
            show_disclaimer()

def display_source_item(source, search_query=""):
    clean_source = source.replace("\\", "/")
    file_name = clean_source.split("/")[-1]

    if "FINE금융용어사전" in source or file_name.endswith(".txt"):
        try:
            name_body = file_name.replace(".txt", "")
            parts = name_body.split("_") 
            
            if len(parts) >= 3:
                term = "_".join(parts[2:]) 
            else:
                term = name_body
                
            display_term = term.replace("_", " ")
            encoded_term = urllib.parse.quote(display_term)
            
        except:
            display_term = file_name
            encoded_term = urllib.parse.quote(file_name)
            

        base_url = "https://fine.fss.or.kr/fine/fnctip/fncDicary/list.do?menuNo=900021"
        search_url = f"{base_url}&searchCnd=2&searchStr={encoded_term}"
        st.markdown(f"- 📘 **[FINE 금융용어사전: '{display_term}']({search_url})**")

    elif "금융꿀팁" in source or ("c_" in file_name and ".pdf" in file_name):
        try:
            file_num_str = file_name.replace("c_", "").replace(".pdf", "")
            file_num = int(file_num_str)
            
            tip_number = 157 - file_num
            
            search_query_encoded = urllib.parse.quote(f"- ({tip_number})")
            board_url = f"https://www.fss.or.kr/fss/bbs/B0000173/list.do?menuNo=200498&searchCnd=1&searchWrd={search_query_encoded}"
            
            st.markdown(f"- 💡 **[금융꿀팁 {tip_number}호 (금융감독원)]({board_url})**")
            
        except Exception:
            st.write(f"- {source}")

    else:
        st.write(f"- {source}")

def render_chat_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "sources" in msg and msg["sources"]:
                with st.expander("참고한 자료 (출처 & 링크)"):
                    for source in msg["sources"]:
                        query = msg.get("query", "")
                        display_source_item(source, search_query=query)

def process_response(user_input):
    if not st.session_state.api_key:
        st.warning("왼쪽 사이드바에서 API 키를 먼저 설정해 주세요.")
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner("자료를 검색하고 답변을 생성 중입니다..."):
            try:
                chain = get_rag_chain(st.session_state.api_key)
                if chain:
                    response = chain.invoke({"input": user_input})
                    answer = response["answer"]
                    
                    sources = set(
                        doc.metadata.get("source", "출처 불명") 
                        for doc in response.get("context", [])
                    )
                    sorted_sources = sorted(list(sources)) if sources else []

                    placeholder.markdown(answer)
                    
                    if sorted_sources:
                        with st.expander("참고한 자료 (출처 & 링크)"):
                            for source in sorted_sources:
                                display_source_item(source, search_query=user_input)

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sorted_sources,
                        "query": user_input 
                    })
                else:
                    st.error("RAG 체인을 초기화할 수 없습니다. API 키나 벡터 스토어를 확인하세요.")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    save_chat_history()

def handle_user_input():
    if len(st.session_state.messages) == 1:
        st.markdown("### 💡 이런 질문은 어때요?")

        recommendations = [
            "휴면예금 조회 방법 알려줘",
            "퇴직연금 실물이전이 뭐야?",
            "연금저축 중도인출 시 세금은?",
            "금리인하요구권 신청 자격은?",
            "ISA 계좌의 장점이 뭐야?",
            "신용점수 올리는 방법 알려줘",
            "보이스피싱 대처 요령은?",
            "예금자보호제도 한도는 얼마야?",
            "내 계좌 한눈에 서비스가 뭐야?",
            "착오송금 반환지원제도란?"
        ]
        
        cols = st.columns(2)
        
        for i, question in enumerate(recommendations):
            if cols[i % 2].button(question, use_container_width=True):
                process_response(question)
                st.rerun()

    user_input = st.chat_input("금융 관련 질문을 입력하세요 (예: ISA 계좌 장점이 뭐야?)")
    
    if user_input:
        process_response(user_input)
        st.rerun()

def main():
    init_chat_state()
    render_sidebar()

    st.markdown("## 💰 Financial Helper Chatbot")

    render_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main()