# Applied AI & Data Quality Labs

Hands-on AI engineering labs by Jason A. Minton, Ph.D. — connecting doctoral research on data quality in Enterprise Architecture to working LLM implementations.

**Research context:** My dissertation ([ProQuest No. 31935662](https://www.proquest.com/)) found that Enterprise Architects and Technology Architects often lack awareness of whether their organizations measure data quality, and that gaps exist in integrating data quality frameworks into EA practice. These labs operationalize that research: each one makes data quality and source grounding a first-class engineering concern in an LLM system.

| Lab | What it builds | Key patterns |
|---|---|---|
| [Lab 1 — RAG over a Dissertation](lab-1-rag-dissertation/) | A retrieval-augmented generation system over a PDF document, with citation tracking | Chunking, embeddings (sentence-transformers), vector store (ChromaDB), grounded prompting, source provenance |
| [Lab 2 — GenAI Visual Analytics](lab-2-genai-visual-analytics/) | A Streamlit app that describes datasets, flags data-quality concerns, and turns plain-English questions into charts | pandas profiling, LLM tool use for structured output, Plotly rendering, AI-narrated captions, caching |

## Quick start

Each lab folder has its own README with setup instructions. Both require:

- Python 3.10+
- An Anthropic API key set as the `ANTHROPIC_API_KEY` environment variable (never hardcoded)

```bash
# Lab 1 dependencies
pip install pypdf sentence-transformers chromadb anthropic

# Lab 2 dependencies
pip install -r lab-2-genai-visual-analytics/requirements.txt
```

## Design principles

Two principles run through both labs, and they come directly from the dissertation:

**Grounding over fluency.** LLMs produce fluent, confident prose whether or not it is anchored to anything verifiable. Both labs engineer against this: Lab 1 restricts answers to retrieved passages and requires citations; Lab 2 forbids invented numbers in captions and surfaces the exact tool call behind every chart.

**Measurement over assumption.** Data quality is conformance to data specifications (Talburt, 2015). Lab 2 measures conformance automatically at ingestion — missing values, impossible values, type inconsistencies, duplicates — and shows the user the results before any analysis begins, rather than leaving quality as an unexamined assumption.

## Author

Jason A. Minton, Ph.D. — Enterprise Data Architect
ORCID: [0000-0002-6077-4601](https://orcid.org/0000-0002-6077-4601)
