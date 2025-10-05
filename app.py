# app.py
# LangChain + Streamlit 테스트용 챗봇
# - 입력창 1개, 버튼 1개
# - OpenAI 키는 st.secrets or 환경변수(OPENAI_API_KEY)로 읽음
# - LangChain 체인: Prompt -> LLM -> Text 출력

import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI Chat 모델 (langchain_openai 패키지)
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Financial Helper Chatbot", page_icon="💬", layout="centered")

st.title("Financial Helper Chatbot (test)")
st.caption("LangChain + Streamlit 기본 연결 확인용 — 입력창 하나만!")

# --- API Key 관리 ---
# 1) Streamlit Secrets: openai_api_key 
api_key = st.secrets.get("openai_api_key", None)
# 2) if not exist, find OPENAI_API_KEY
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")
# 3) if not exist both, enter the key
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
    st.warning("Warning: OpenAI API Key is needed.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llm(_api_key: str, _model: str):
    # temperature=0.2: 답변 안정성/일관성 중시
    return ChatOpenAI(api_key=_api_key, model=_model, temperature=0.2)

llm = get_llm(api_key, model_name)

system_prompt = (
    "너는 금융 정보를 설명하는 도우미야. 한국어로 간결하고 정확하게 답해.\n"
    "법률·세무·투자 자문이 아닌 일반 정보라는 점을 분명히 하고, 필요하면 최신 자료 확인을 권고해."
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{user_input}"),
    ]
)

chain = prompt | llm | StrOutputParser()


user_input = st.text_area("질문을 입력하세요 (예: ETF랑 펀드 차이가 뭐야?)", height=120)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("질문하기", type="primary")

if run:
    if not user_input.strip():
        st.error("질문을 입력해주세요.")
    else:
        with st.spinner("생각 중..."):
            try:
                answer = chain.invoke({"user_input": user_input.strip()})
            except Exception as e:
                st.error(f"오류가 발생했어요: {e}")
            else:
                st.markdown("### 답변")
                st.write(answer)
                st.info("※ 본 정보는 참고용 일반 설명입니다. 실제 투자/세무/법률 판단은 공신력 있는 최신 자료와 전문가 상담을 권장합니다.")
