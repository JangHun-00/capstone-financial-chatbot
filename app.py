# app.py
# (RAG 챗봇 실행 파일)
# 'faiss_index'에 저장된 벡터 스토어를 '로드'하여 사용합니다.

import os
import streamlit as st

# --- 최신 LangChain 모듈 임포트 ---

# 1. LLM 및 임베딩 (OpenAI)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 2. 프롬프트, 파서, 스키마 (Core)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 3. 벡터 스토어 (Community) - ★ 로더/스플리터는 여기서 제거됨
from langchain_community.vectorstores import FAISS

# 4. RAG 체인 (Main)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Financial Helper Chatbot", page_icon="💬", layout="centered")

st.title("Financial Helper Chatbot (RAG Ver.)")
st.caption("LangChain + Streamlit RAG 연결 확인용 (로컬 인덱스 로드)")

# --- API Key 관리 ---
# (이 부분은 기존과 동일)
api_key = st.secrets.get("openai_api_key", None)
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")
with st.sidebar:
    st.subheader("🔑 API Key")
    sidebar_key = st.text_input(
        "OpenAI API Key", type="password", placeholder="sk-..."
    )
    model_name = st.selectbox(
        "model select",
        options=["gpt-4o-mini"],
        index=0
    )
    st.markdown("---")
    st.caption("키 우선순위: secrets → 환경변수 → 여기 입력")

if sidebar_key:
    api_key = sidebar_key

if not api_key:
    st.warning("Warning: OpenAI API Key가 필요합니다. 사이드바에 입력해주세요.")
    st.stop()


@st.cache_resource(show_spinner="[1/2] LLM 모델을 준비하는 중...")
def get_llm(_api_key: str, _model: str):
    """LLM 모델을 캐시하여 반환"""
    return ChatOpenAI(api_key=_api_key, model=_model, temperature=0.2)


# ★★★ 핵심 변경점: get_vectorstore -> load_vectorstore ★★★

# (기존 get_vectorstore 함수는 삭제)

@st.cache_resource(show_spinner="[2/2] 저장된 금융 문서를 로드하는 중...")
def load_vectorstore(_api_key: str):
    """
    'faiss_index' 폴더에서 저장된 벡터 스토어를 로드합니다.
    """
    VECTORSTORE_PATH = "faiss_index"
    
    # 1. 벡터 스토어를 로드할 때도 임베딩 모델은 필요합니다.
    embeddings = OpenAIEmbeddings(api_key=_api_key)

    # 2. 로컬 파일에서 벡터 스토어 로드
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings,
            # (FAISS v1.8.0 이상) 로컬 인덱스 로드 시 이 플래그가 필요할 수 있습니다.
            allow_dangerous_deserialization=True 
        )
    except Exception as e:
        st.error(f"벡터 스토어 로드 실패: {e}")
        st.error(f"'{VECTORSTORE_PATH}' 폴더가 존재하지 않거나 손상되었습니다. 'build_vectorstore.py'를 먼저 실행해주세요.")
        st.stop()
        
    return vectorstore

# --- 메인 로직 ---
# LLM 및 벡터 스토어 준비
try:
    llm = get_llm(api_key, model_name)
    # ★ 수정: 저장된 인덱스를 '로드'
    vectorstore = load_vectorstore(api_key)
except Exception as e:
    st.error(f"모델 또는 벡터 스토어 로딩 중 오류가 발생했습니다: {e}")
    st.stop()


# --- RAG 체인 구성 ---
# (이 부분은 기존과 동일)

# 1. Retriever 생성 (벡터 스토어에서 문서를 검색하는 역할)
retriever = vectorstore.as_retriever()

# 2. RAG용 프롬프트 정의
# 2. RAG용 프롬프트 정의
system_prompt = (
    "너는 금융 정보를 설명하는 전문 도우미야. 너의 답변은 다음 규칙을 따라야 해:\n\n"
    "--- 답변 규칙 ---\n"
    "1. **(정의 검색)**: 먼저 '참고 자료'({context})에서 질문받은 용어의 **사전적 정의**가 있는지(주로 'a_*.txt' 파일) 확인해.\n"
    
    "2. **(Case 1: 정의 있음)**: 만약 '참고 자료'에 사전적 정의가 있다면:\n"
    "   가. 해당 정의를 바탕으로 **핵심 답변**을 해. [cite]를 꼭 달아줘.\n"
    "   나. '참고 자료'에 추가적인 **유용한 정보**(꿀팁, 예시 등, 주로 'c_*.pdf' 파일)가 있다면 '유용한 추가 정보:' 섹션을 만들어 요약해줘.\n"

    "3. **(Case 2: 정의 없음)**: 만약 '참고 자료'에 사전적 정의가 **없다면** (예: '적금'):\n"
    "   가. 너의 **일반 지식**을 사용해 해당 용어의 정의를 설명해. (이때 '제공된 자료에 정의가 없어...' 같은 말은 *하지 않아도 돼*.)\n"
    "   나. **하지만**, '참고 자료'에 그 용어와 관련된 **유용한 정보**(꿀팁, 예시 등)가 검색되었다면, '참고 자료에서 찾은 유용한 정보:' 섹션을 만들어 반드시 요약해줘.\n"

    "4. **(무관한 자료)**: 만약 '참고 자료'가 질문과 **전혀 무관**하다면, 무시하고 너의 일반 지식으로만 답해.\n"
    
    "5. **(필수 안내)**: 모든 답변 마지막에는 항상 아래의 안내 문구를 추가해.\n"
    "   '※ 본 정보는 참고용 일반 설명입니다. 실제 투자/세무/법률 판단은 공신력 있는 최신 자료와 전문가 상담을 권장합니다.'\n"
    "--- 참고 자료 ---\n"
    "{context}"
    "---"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 3. LLM + Prompt 체인 생성
document_chain = create_stuff_documents_chain(llm, prompt)

# 4. Retrieval 체인 생성
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- Streamlit UI ---
# (이 부분은 기존과 동일)
user_input = st.text_area("질문을 입력하세요 (예: 휴면예금이 뭐야?)", height=120)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("질문하기", type="primary")

if run:
    if not user_input.strip():
        st.error("질문을 입력해주세요.")
    else:
        with st.spinner("자료를 검색하고 답변을 생성 중..."):
            try:
                response = retrieval_chain.invoke({"input": user_input.strip()})
                
            except Exception as e:
                st.error(f"오류가 발생했어요: {e}")
            else:
                st.markdown("### 답변")
                st.write(response["answer"])
                
                with st.expander("참고한 자료 (출처)"):
                    sources = set(
                        doc.metadata.get("source", "출처 불명") 
                        for doc in response.get("context", [])
                    )
                    if sources:
                        for source in sorted(list(sources)):
                            st.write(f"- {source}")
                    else:
                        st.write("참고한 자료를 찾지 못했습니다.")

                st.info("※ 본 정보는 참고용 일반 설명입니다. 실제 투자/세무/법률 판단은 공신력 있는 최신 자료와 전문가 상담을 권장합니다.")