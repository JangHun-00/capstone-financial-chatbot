import os
import streamlit as st

# --- LangChain & RAG 관련 임포트 (app.py에서 가져옴) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 페이지 설정 (프론트엔드 스타일 적용) ---
st.set_page_config(
    page_title="Financial Helper Chatbot",
    page_icon="💬",
    layout="wide",
)

# 사용자 정의 CSS (프론트엔드 파일에서 가져옴)
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    [data-testid="stChatMessage"] {
        margin-bottom: 0.5rem;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- RAG 로직 (app.py에서 가져옴) ---

@st.cache_resource(show_spinner="[1/2] LLM 모델을 준비하는 중...")
def get_llm(api_key: str, model: str):
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.2)

@st.cache_resource(show_spinner="[2/2] 저장된 금융 문서를 로드하는 중...")
def load_vectorstore(api_key: str):
    VECTORSTORE_PATH = "faiss_index"
    embeddings = OpenAIEmbeddings(api_key=api_key)
    try:
        # FAISS 인덱스 로드
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
    """RAG 체인을 생성하여 반환하는 헬퍼 함수"""
    llm = get_llm(api_key, "gpt-4o-mini")
    vectorstore = load_vectorstore(api_key)
    
    if not vectorstore:
        return None

    retriever = vectorstore.as_retriever()
    
    # 하이브리드 프롬프트 (app.py에서 가져옴)
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

# --- UI 로직 (streamlit_app.py 기반 수정) ---

def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요 👋 금융 궁금증을 해결해 드릴게요! 어떤 걸 도와드릴까요?"}
        ]

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # API 키 관리 로직 통합
        # 1. secrets 확인
        default_key = st.secrets.get("openai_api_key", "")
        # 2. 환경변수 확인
        if not default_key:
            default_key = os.getenv("OPENAI_API_KEY", "")

        if "api_key" not in st.session_state:
            st.session_state.api_key = default_key

        st.markdown("### 🔐 API Key")
        
        with st.form("api-key-form", clear_on_submit=False):
            api_key_input = st.text_input(
                "API Key 입력",
                type="password",
                placeholder="sk-...",
                value=st.session_state.api_key,
                help="키가 없으면 작동하지 않습니다."
            )
            submitted = st.form_submit_button("저장")
            if submitted:
                st.session_state.api_key = api_key_input.strip()
                if st.session_state.api_key:
                    st.success("API 키가 저장되었습니다!")
                else:
                    st.error("유효한 API 키를 입력해 주세요.")

def render_chat_messages():
    """저장된 대화 기록을 화면에 표시"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 저장된 출처가 있다면 표시 (history 구조에 'sources' 키를 추가해서 관리할 수도 있음)
            if "sources" in msg:
                with st.expander("참고한 자료 (출처)"):
                    for source in msg["sources"]:
                        st.write(f"- {source}")

def handle_user_input():
    user_input = st.chat_input("금융 관련 질문을 입력하세요 (예: 휴면예금이 뭐야?)")
    
    if not user_input:
        return

    # API 키 확인
    if not st.session_state.api_key:
        st.warning("왼쪽 사이드바에서 API 키를 먼저 설정해 주세요.")
        return

    # 1. 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. RAG 답변 생성
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner("자료를 검색하고 답변을 생성 중입니다..."):
            try:
                # RAG 체인 가져오기
                chain = get_rag_chain(st.session_state.api_key)
                if chain:
                    # 체인 실행
                    response = chain.invoke({"input": user_input})
                    answer = response["answer"]
                    
                    # 출처 추출
                    sources = set(
                        doc.metadata.get("source", "출처 불명") 
                        for doc in response.get("context", [])
                    )
                    sorted_sources = sorted(list(sources)) if sources else []

                    # 화면 표시
                    placeholder.markdown(answer)
                    if sorted_sources:
                        with st.expander("참고한 자료 (출처)"):
                            for source in sorted_sources:
                                st.write(f"- {source}")

                    # 3. 대화 기록에 저장 (답변 + 출처)
                    # 나중에 다시 렌더링할 때를 위해 sources도 같이 저장
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sorted_sources
                    })
                else:
                    st.error("RAG 체인을 초기화할 수 없습니다. API 키나 벡터 스토어를 확인하세요.")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

def main():
    init_chat_state()
    render_sidebar()

    st.markdown("## 💬 Financial Helper Chatbot (RAG Ver.)")
    st.caption("LangChain + Streamlit RAG 통합 버전")

    render_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main()