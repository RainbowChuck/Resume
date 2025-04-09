import json
import os

SOURCE_PATH = os.path.join("data", "cv.json")
DEST_PATH = os.path.join("data", "cv_10k.json")
LIMIT = 10000

def extract_10k_resumes():
    selected = []
    with open(SOURCE_PATH, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i >= LIMIT:
                break
            try:
                record = json.loads(line.strip())
                selected.append(record)
            except json.JSONDecodeError:
                continue

    with open(DEST_PATH, 'w', encoding='utf-8') as outfile:
        for item in selected:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Сохранено {len(selected)} резюме в файл: {DEST_PATH}")

if __name__ == "__main__":
    extract_10k_resumes()
