# GenAI Visual Analytics — Lab 2

AA working Streamlit application that demonstrates LLM-augmented data exploration with deliberate data-quality awareness, grounded in doctoral research on data quality in Enterprise Architecture (Minton, 2025).

## What this lab demonstrates

Four current AI engineering patterns, end-to-end, in roughly 350 lines of Python:

1. **LLM-generated dataset descriptions** — given a pandas DataFrame, the LLM produces a plain-English summary of what the dataset appears to contain.

2. **LLM-driven data quality flagging** — grounded in Talburt's (2015) definition of data quality as *conformance to data specifications*, the LLM surfaces missing values, impossible values, type inconsistencies, duplicates, and anomalous distributions before any analysis takes place.

3. **Natural-language-to-chart via tool use** — the user asks a plain-English question; the Anthropic API's tool-use feature lets the LLM pick exactly one chart type from a curated palette with structured arguments; Python then renders the chart deterministically. This is the modern alternative to having the LLM emit raw Python (unsafe) or raw JSON specs (brittle).

4. **AI-narrated captions** — after a chart is rendered, the LLM writes a short explanation of what the chart is structured to reveal.

## What's in this folder

- `app.py` — the full Streamlit application
- `sample_data.py` — generates a synthetic Washington water-monitoring dataset with seeded data-quality issues, so the lab runs self-contained
- `GenAI-Visual-Analytics-Lab.ipynb` — a teaching notebook that builds the concepts piece by piece before assembling the app
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Running locally

The simplest path. Requires Python 3.10 or later.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Provide your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# 3. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. The sidebar lets you toggle between the synthetic sample dataset (recommended for first-run exploration) and your own CSV upload.

## Deploying to Streamlit Community Cloud

This is the version you can share as a live URL — ideal for portfolio links or live demonstrations.

1. **Push this folder to a public GitHub repository.** Streamlit Community Cloud reads from public repos. (For a private demo, you can also use a private repo if you connect Streamlit to your GitHub account with the right permissions.)

2. **Sign in to [share.streamlit.io](https://share.streamlit.io)** with your GitHub account.

3. **Click *New app*** and select your repository. Set the main file path to `app.py`.

4. **Add your Anthropic API key as a secret.** In the deployment configuration, go to *Advanced settings → Secrets* and add:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```

5. **Deploy.** First deployment takes 2-3 minutes. After that you have a permanent public URL (`https://your-app-name.streamlit.app`) that you can share or include on your GitHub profile.

Cost note: Streamlit Community Cloud is free for public apps. Anthropic API calls are charged to your account at the standard per-token rate; this app uses well under a dollar of credit even with extensive use, since each interaction is at most three short LLM calls.

## How to use the app

The flow is linear:

1. **Choose a data source** — sample dataset (synthetic WA water monitoring) or upload your own CSV.

2. **Review the overview** — first ten rows plus row/column/missing-value counts.

3. **Read the AI-generated description** — a short paragraph describing what the dataset appears to contain.

4. **Review the data-quality concerns** — high/medium/low-severity issues the LLM flagged. This is the section most directly connected to the dissertation: it surfaces awareness gaps that organizations often leave implicit.

5. **Ask a question in plain English** — for example, *"How does temperature vary across the basins?"* or *"Show me the distribution of pH."* The LLM picks a chart, renders it, and writes a caption.

## Conversation framing

When the moment arises in conversation (most likely something like *"can you show me something you've built recently"*), the things worth highlighting:

- **Tool use, not code execution.** The LLM doesn't write Python that we run blindly — it picks from a curated chart palette with structured arguments, which is debuggable and safe.

- **Data quality as a first-class citizen.** The app surfaces concerns before analysis, not after. This is the practical application of the dissertation's recommendation that organizations need to deliberately measure data quality.

- **Caching at the right boundary.** Dataset descriptions and DQ concerns are cached per dataset signature, so they're computed once and reused across user interactions. This matters for cost and latency in production.

- **Honest about limits.** The LLM can pick the wrong chart, misread a column type, or produce a caption that doesn't quite match what the chart shows. None of these is hidden; the app shows the tool call so the user can audit what happened.

## Extension ideas

Ideas to extend this lab:

- **Verification layer** — after the LLM picks a chart, programmatically verify that the referenced columns exist and have the expected types before rendering. Currently the rendering step will raise an error; a better version surfaces it more usefully.

- **Multi-chart responses** — let the LLM call multiple chart tools for richer answers (e.g., a histogram plus a box plot).

- **Conversational memory** — keep a session history so follow-up questions ("now color by site") work as expected.

- **Data quality severity scoring** — currently the LLM tags concerns as high/medium/low; a more rigorous version would compute deterministic severity from quantitative thresholds and ask the LLM only to explain them.

Each of these is a small extension; together they have the possibility to convert the lab from a demonstration into a production-ready tool.
