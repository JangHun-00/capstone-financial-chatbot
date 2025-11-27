# rename_files.py
import os
import re

def rename_dictionary_files_keep_id():
    base_path = "./data/FINE금융용어사전"
    
    if not os.path.exists(base_path):
        print(f"경로를 찾을 수 없습니다: {base_path}")
        return

    count = 0
    for filename in os.listdir(base_path):
        # a_로 시작하고 .txt로 끝나는 파일만 대상
        if filename.startswith("a_") and filename.endswith(".txt"):
            old_path = os.path.join(base_path, filename)
            
            try:
                with open(old_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 용어 추출 로직 (이전과 동일)
                clean_content = re.sub(r'\[source.*?\]', '', content).strip()
                first_line = clean_content.split('\n')[0]
                term = first_line.split('[')[0].strip()
                safe_term = re.sub(r'[\\/*?:"<>|]', "", term)
                
                if safe_term:
                    # ★ 변경된 부분: 원래 파일명(확장자 제외) + "_" + 용어 + ".txt"
                    # 예: a_000.txt -> a_000_휴면예금.txt
                    original_name_no_ext = filename.replace(".txt", "")
                    new_filename = f"{original_name_no_ext}_{safe_term}.txt"
                    
                    new_path = os.path.join(base_path, new_filename)
                    
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"[변경] {filename} -> {new_filename}")
                        count += 1
                    else:
                        print(f"[스킵] {new_filename} 이미 존재함")
                        
            except Exception as e:
                print(f"[에러] {filename} 처리 중 오류: {e}")

    print(f"\n총 {count}개의 파일명이 변경되었습니다.")

if __name__ == "__main__":
    rename_dictionary_files_keep_id()