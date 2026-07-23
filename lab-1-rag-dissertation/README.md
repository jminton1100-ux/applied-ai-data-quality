# RAG Lab — Dissertation Edition

A hands-on lab that builds a working Retrieval-Augmented Generation system over a dissertation, end-to-end, in a single Jupyter notebook.

## About the source document

This lab runs against the author's doctoral dissertation:

> Minton, J. A. (2025). *Data Quality and Enterprise Architecture: An Analysis of the Awareness and Knowledge of Data Quality Artifacts, Principles, Practices, and Application in Enterprise Architecture* (Doctoral dissertation). University of Arkansas at Little Rock. ProQuest No. 31935662.

The dissertation PDF is **not included in this repository**. The author holds full copyright; the PDF is available from the author on request, or via ProQuest. The lab works with any substantial PDF, so you can also point it at a document of your own.

## What's in this folder

- `RAG-Dissertation-Lab.ipynb` — the lab itself, designed to run in Google Colab or local Jupyter

## How to start

1. **Open the notebook in Google Colab.** Go to [colab.research.google.com](https://colab.research.google.com), click *File → Upload notebook*, and select `RAG-Dissertation-Lab.ipynb`. Colab is free and requires no local setup.

2. **Upload the dissertation PDF.** In the Colab file browser (folder icon on the left), drag your PDF into `/content/`. Name it `dissertation.pdf` or change the variable `PDF_PATH` in the first code cell of Part 1.

3. **Add your Anthropic API key.** Sign up at [console.anthropic.com](https://console.anthropic.com), generate an API key, and store it in Colab using the key icon in the left sidebar. Name the secret `ANTHROPIC_API_KEY`. The lab will use under $1 of API credit even with heavy experimentation; new accounts typically receive free starter credit covering this many times over.

4. **Work the cells in order.** Total focused time is roughly 90 minutes. Reflection prompts between code cells are the most important part — do not skip them.

## What you will have at the end

A working RAG system over a dissertation, with citation tracking, that you can demonstrate. The notebook also serves as a portfolio artifact demonstrating an end-to-end RAG implementation with citation tracking.

## If you run into trouble

The most common issues are PDF parsing artifacts (handled in Part 1 with a discussion of when to switch parsers), embedding model downloads taking longer than expected on first run (normal — `all-MiniLM-L6-v2` is about 90MB), and Colab session timeouts (Colab disconnects free sessions after roughly 90 minutes of inactivity; reconnect and re-run from the top).

If you want to switch from Anthropic Claude to a different LLM provider (OpenAI, Mistral, Hugging Face Hub), only the Part 5 code cell that sets up `claude = anthropic.Anthropic()` and the `claude.messages.create(...)` call need to change. The retrieval pipeline is provider-agnostic.
