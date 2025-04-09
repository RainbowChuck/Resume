with open("data/cv.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i > 4:
            break
        print(line)