# 📑 RAG with Cross-Encoders Re-ranking

This is an experiment of using LLM to have a personal specialized assistant using local PDF as knowledge base.

🚨 NOTE: **Requires `Python > 3.10` with  `SQLite > 3.35`**

---

## 🤖 Prerequisites

- [Ollama](https://ollama.dev/download)

## 🔨 Setting up locally

Create virtualenv and install dependencies.

```sh
make setup
```

## ⚡️ Running the application

```sh
make run
```

## ✨ Linters and Formatters

Check for linting rule violations:

```sh
make check
```

Auto-fix linting violations:

```sh
make fix
```

## 🤸‍♀️ Getting Help

```sh
make

# OR

make help
```

## 🔧 Common Issues and Fixes

- If you run into any errors with incompatible version of ChromaDB/Sqlite3, refer to [this solution](https://docs.trychroma.com/troubleshooting#sqlite).
