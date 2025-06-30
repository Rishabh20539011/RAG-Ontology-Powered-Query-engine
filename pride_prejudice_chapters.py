# Existing helper used in notebooks / demos – no secret leakage, so we keep it as‑is
import re, json, pathlib, requests
from typing import List, Dict

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
raw_text = requests.get(URL, timeout=30).text
# -------------------------------------------------------------------
# 2. Strip Gutenberg header & footer
# -------------------------------------------------------------------
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END   = "*** END OF THE PROJECT GUTENBERG EBOOK"

start_i = raw_text.find(START)
end_i   = raw_text.find(END)

if start_i == -1 or end_i == -1:
    raise RuntimeError("Could not locate Gutenberg markers!")

novel_text = raw_text[start_i + len(START) : end_i].strip()
# normalise newlines
novel_text = novel_text.replace("\r\n", "\n")

print(novel_text)

# -------------------------------------------------------------------
# 3. Split into chapters
# -------------------------------------------------------------------
chap_regex = re.compile(r"^\s*\[?CHAPTER\s+[IVXLCDM]+\.?\]?\s*$", re.IGNORECASE | re.MULTILINE)
matches = list(chap_regex.finditer(novel_text))

chapters: List[Dict[str, str]] = []
for idx, match in enumerate(matches):
    start = match.start()
    end = matches[idx+1].start() if idx+1 < len(matches) else len(novel_text)
    title = match.group(0).strip()
    body  = novel_text[start:end].strip()
    chapters.append({"id": idx+1, "title": title, "text": body})


# -------------------------------------------------------------------
# 4. Persist chapter corpus
# -------------------------------------------------------------------
out_dir = pathlib.Path("pride_prejudice_chapters")
out_dir.mkdir(exist_ok=True)

# JSON dump
(pathlib.Path("chapters.json")
    .write_text(json.dumps(chapters, indent=2, ensure_ascii=False), encoding="utf‑8"))

# one .txt per chapter (convenient for SimpleDirectoryReader)
for ch in chapters:
    (out_dir / f"chapter_{ch['id']:02}.txt").write_text(ch["text"], encoding="utf‑8")

print(f"Saved {len(chapters)} chapters to {out_dir.resolve()}")