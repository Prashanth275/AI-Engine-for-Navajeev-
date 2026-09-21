import os

repo_path = r"D:\apps\Preg-chatbot-ollama"
keywords = ["openai", "OpenAIEmbeddings", "langchain_openai", "OPENAI_API_KEY", "gpt-4o-mini"]

findings = []

for root, dirs, files in os.walk(repo_path):
    # Exclude virtual environment and git
    if ".git" in root or "__pycache__" in root or "venv" in root or "node_modules" in root:
        continue
    for file in files:
        if file.endswith((".py", ".env", ".env.example", ".json", ".txt", ".html", ".md")):
            filepath = os.path.join(root, file)
            relpath = os.path.relpath(filepath, repo_path)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        for kw in keywords:
                            if kw.lower() in line.lower():
                                findings.append({
                                    "file": relpath,
                                    "line": line_num,
                                    "kw": kw,
                                    "content": line.strip()
                                })
                                break
            except Exception as e:
                pass

print("=== PROJECT SOURCE AUDIT (Excluding venv/.git) ===")
if not findings:
    print("ZERO OpenAI references found in source files!")
else:
    for f in findings:
        clean_content = f['content'].encode('ascii', errors='replace').decode('ascii')
        print(f"{f['file']}:{f['line']} [{f['kw']}] -> {clean_content}")
