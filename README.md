# 🛠️ Argo — یک Agent واقعی برای کدنویسی

یه جعبه‌ابزار کامل برای ساخت ایجنت کدنویسی که با هر بک‌اند سازگار با OpenAI کار می‌کنه (OpenRouter, Ollama, vLLM, llama.cpp server, …).

## چه چیزی متفاوته؟

پروژه قبلی (llm-uncensored-toolkit) با این مشکلات دست و پنجه نرم می‌کرد:

- UI دو خطی، اسپاگتی، بدون تاریخچه چندگانه
- parser مبتنی بر regex که با هر مدل جدیدی می‌شکست
- حلقه ایجنت ابتدایی، بدون مدیریت context
- 5 ابزار بدون تست، با رفتار غیرقابل پیش‌بینی
- system prompt که سعی می‌کرد همه چیز رو با دور زدن درست کنه

Argo همه اینا رو از اول می‌نویسه:

- **Frontend مدرن** با Vanilla JS (بدون build step، بدون React، بدون webpack) — چت، تاریخچه، تنظیمات، استریم زنده
- **API تمیز** با FastAPI + Pydantic v2
- **Agent حلقه‌ای قوی** با parser هوشمند (پشتیبانی از `<tool_call>`, ```json blocks, ReAct `Action:`/`Action Input:`، و حالت آزاد)
- **6 ابزار** sandbox شده: bash, read_file, write_file, edit_file, list_dir, search_files
- **مدیریت context** با sliding window + خلاصه‌سازی اختیاری
- **تشخیص حلقه** (loop detection) و محدودیت زمانی
- **استریم SSE** برای چت و agent
- **CLI** هم داره (همون coding_agent قبلی، ولی درست نوشته شده)
- **بدون وابستگی به Colab**: روی هر سرور، VPS، یا حتی لپ‌تاپ اجرا میشه
- **بدون کامپایل**: فقط pip install

## نصب

```bash
git clone <repo> argo
cd argo
pip install -r requirements.txt

# Backend
export LLM_API_KEY="sk-or-..."
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="qwen/qwen3-coder:free"
python -m app.main
# → http://localhost:8000
```

## استفاده

### چت وب
مرورگر رو باز کن روی `http://localhost:8000`. تنظیمات (مدل، API key، base URL) از توی UI قابل تغییره.

### CLI (ایجنت کدنویسی در ترمینال)
```bash
python -m cli.agent "این پروژه رو تحلیل کن و README بنویس"
python -m cli.agent --workdir /path/to/project --auto
python -m cli.agent  # حالت تعاملی
```

### استفاده از مدل‌های رایگان OpenRouter
```bash
export LLM_MODEL="qwen/qwen3-coder:free"
export LLM_API_KEY="sk-or-..."
```

### استفاده از Ollama (لوکال)
```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5-coder:14b"
```

### استفاده از llama.cpp server
```bash
# اول سرور رو بالا بیار
./server -m model.gguf -c 8192 --host 0.0.0.0 --port 8080

# بعد
export LLM_BASE_URL="http://localhost:8080/v1"
export LLM_API_KEY="sk-no-key-required"
export LLM_MODEL="local-model"
```

## ساختار

```
argo/
├── app/                    # FastAPI backend
│   ├── main.py            # entrypoint
│   ├── api/               # REST endpoints
│   ├── agent/             # agent loop + parser
│   ├── tools/             # 6 ابزار sandbox شده
│   ├── core/              # config, llm client
│   ├── static/            # CSS, JS, images
│   └── templates/         # HTML
├── cli/                    # ترمینال ایجنت
├── docs/                   # مستندات
├── tests/                  # تست
└── requirements.txt
```

## لایسنس

MIT
