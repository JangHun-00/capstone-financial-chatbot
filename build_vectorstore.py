# build_vectorstore.py
# (최초 1회 또는 문서 변경 시 실행)
# data/ 폴더의 문서를 로드, 분할, 임베딩하여 'faiss_index' 폴더에 저장합니다.

import os
import time
from dotenv import load_dotenv

# 1. 로더, 스플리터, 임베딩, 벡터스토어 import
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# .env 파일에서 환경 변수(API 키) 로드
load_dotenv()

# 저장될 로컬 경로
VECTORSTORE_PATH = "faiss_index"

def build_vectorstore():
    import streamlit as st
    print("벡터 스토어 빌드를 시작합니다...")

    # 1. API 키 확인
    api_key = st.secrets.get("openai_api_key", None)
    # 2) if not exist, find OPENAI_API_KEY
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print(".env 파일을 생성하거나 터미널에서 직접 키를 설정해주세요.")
        return

    # 2. 문서 경로 정의
    pdf_path = "./data/금융감독원금융꿀팁200선"
    txt_path = "./data/FINE금융용어사전"

    # 3. 로더 생성 및 문서 로드
    print(f"'{pdf_path}'에서 PDF 문서를 로드합니다...")
    pdf_loader = PyPDFDirectoryLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    print(f"'{txt_path}'에서 TXT 문서를 로드합니다...")
    txt_loader = DirectoryLoader(
        txt_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    txt_docs = txt_loader.load()
    all_docs = pdf_docs + txt_docs
    print(f"총 {len(all_docs)}개의 문서를 로드했습니다.")

    # 4. 문서 분할 (Chunking)
    print("문서를 분할합니다 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    print(f"총 {len(splits)}개의 문서 조각으로 분할되었습니다.")

    # 5. 임베딩 모델 생성
    print("OpenAI 임베딩 모델을 사용해 벡터화를 시작합니다... (시간이 걸립니다)")
    embeddings = OpenAIEmbeddings(api_key=api_key, chunk_size=200)

    # 6. 벡터 스토어 생성 (FAISS) 및 저장
    start_time = time.time()
    vectorstore = FAISS.from_documents(splits, embeddings)
    end_time = time.time()
    
    print(f"벡터화 완료. (소요 시간: {end_time - start_time:.2f}초)")

    # 7. 로컬에 저장
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"성공: 벡터 스토어를 '{VECTORSTORE_PATH}' 폴더에 저장했습니다.")


if __name__ == "__main__":
    build_vectorstore()