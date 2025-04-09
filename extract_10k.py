import os
import json
import ijson

SOURCE_PATH = os.path.join("data", "cv.json")
DEST_PATH = os.path.join("data", "cv_10k.json")
LIMIT = 10000

os.makedirs("data", exist_ok=True)

def extract_10k():
    print("[Извлечение] Чтение из большого JSON...")
    count = 0
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f, \
         open(DEST_PATH, 'w', encoding='utf-8') as out:
        for item in ijson.items(f, "cvs.item"):
            if count >= LIMIT:
                break
            json.dump(item, out, ensure_ascii=False)
            out.write('\n')
            count += 1
    print(f"Сохранено {count} резюме в {DEST_PATH}")

if __name__ == "__main__":
    extract_10k()
