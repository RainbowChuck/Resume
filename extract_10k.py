import ijson
import json
import os

SOURCE_PATH = os.path.join("data", "cv.json")
DEST_PATH = os.path.join("data", "cv_10k.json")
LIMIT = 10000

def extract_from_nested_json():
    # Проверка существования исходного файла
    if not os.path.exists(SOURCE_PATH):
        print(f"Ошибка: файл {SOURCE_PATH} не найден.")
        return

    try:
        # Открытие исходного файла и целевого файла
        with open(SOURCE_PATH, 'r', encoding='utf-8') as f, \
             open(DEST_PATH, 'w', encoding='utf-8') as out:
            count = 0

            # Использование ijson для потокового чтения массива "cvs"
            for item in ijson.items(f, "cvs.item"):
                if count >= LIMIT:
                    break
                json.dump(item, out, ensure_ascii=False)
                out.write('\n')
                count += 1

        print(f"Сохранено {count} резюме в файл: {DEST_PATH}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    extract_from_nested_json()