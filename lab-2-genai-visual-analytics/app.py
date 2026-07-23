"""
GenAI Visual Analytics — a Streamlit demonstration of LLM-augmented
data exploration with deliberate data-quality awareness.

Demonstrates four engineering patterns for people to see:

    1. LLM-generated dataset summaries from a pandas DataFrame
    2. LLM-driven data-quality flagging grounded in Talburt's definition
       of data quality as conformance to data specifications
    3. Natural-language-to-chart via the Anthropic tool use API (the
       current best-practice pattern for structured LLM output)
    4. AI-narrated captions that explain what each chart shows

To run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py

To deploy to Streamlit Community Cloud, see README.md.
"""

import json
import os

import anthropic
import pandas as pd
import plotly.express as px
import streamlit as st

from sample_data import generate_sample_dataset


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="GenAI Visual Analytics",
    page_icon="📊",
    layout="wide",
)

MODEL = "claude-sonnet-4-5"


# ----------------------------------------------------------------------
# Tool definitions for natural-language-to-chart
# ----------------------------------------------------------------------
# These tool schemas tell the LLM what charts are available and how to
# request one. The LLM examines the user question + dataset schema, then
# calls exactly one tool with structured arguments. This is the modern
# alternative to having the LLM emit raw Python or raw JSON specs.

CHART_TOOLS = [
    {
        "name": "create_bar_chart",
        "description": "Create a bar chart. Use for comparing values across categories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_column": {"type": "string", "description": "Categorical column for the x-axis"},
                "y_column": {"type": "string", "description": "Numeric column for the y-axis"},
                "aggregation": {
                    "type": "string",
                    "enum": ["mean", "sum", "count", "median"],
                    "description": "How to aggregate y_column when x_column has duplicate categories",
                },
                "title": {"type": "string", "description": "A descriptive chart title"},
            },
            "required": ["x_column", "y_column", "aggregation", "title"],
        },
    },
    {
        "name": "create_line_chart",
        "description": "Create a line chart. Use for trends over time or continuous progressions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_column": {"type": "string", "description": "Column for the x-axis, typically a date or ordered numeric"},
                "y_column": {"type": "string", "description": "Numeric column for the y-axis"},
                "color_column": {"type": "string", "description": "Optional grouping column; pass empty string if no grouping"},
                "title": {"type": "string", "description": "A descriptive chart title"},
            },
            "required": ["x_column", "y_column", "title"],
        },
    },
    {
        "name": "create_scatter_chart",
        "description": "Create a scatter plot. Use for relationships between two numeric variables.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_column": {"type": "string", "description": "Numeric column for the x-axis"},
                "y_column": {"type": "string", "description": "Numeric column for the y-axis"},
                "color_column": {"type": "string", "description": "Optional categorical column to color points by; pass empty string for none"},
                "title": {"type": "string", "description": "A descriptive chart title"},
            },
            "required": ["x_column", "y_column", "title"],
        },
    },
    {
        "name": "create_histogram",
        "description": "Create a histogram. Use for understanding the distribution of a single numeric variable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Numeric column to show the distribution of"},
                "title": {"type": "string", "description": "A descriptive chart title"},
            },
            "required": ["column", "title"],
        },
    },
    {
        "name": "create_box_plot",
        "description": "Create a box plot. Use for comparing distributions across categories or surfacing outliers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value_column": {"type": "string", "description": "Numeric column to show distributions of"},
                "group_column": {"type": "string", "description": "Optional categorical column to split distributions by; pass empty string for none"},
                "title": {"type": "string", "description": "A descriptive chart title"},
            },
            "required": ["value_column", "title"],
        },
    },
]


# ----------------------------------------------------------------------
# Chart rendering — deterministic Python that turns a tool call into a chart
# ----------------------------------------------------------------------

def render_chart_from_tool_call(tool_name: str, args: dict, df: pd.DataFrame):
    """Take a tool-use request from the LLM and return a Plotly figure."""
    if tool_name == "create_bar_chart":
        # Aggregate before plotting so the bar represents the requested statistic.
        agg = args.get("aggregation", "mean")
        grouped = df.groupby(args["x_column"])[args["y_column"]].agg(agg).reset_index()
        return px.bar(grouped, x=args["x_column"], y=args["y_column"], title=args["title"])

    if tool_name == "create_line_chart":
        color = args.get("color_column") or None
        return px.line(df, x=args["x_column"], y=args["y_column"], color=color, title=args["title"])

    if tool_name == "create_scatter_chart":
        color = args.get("color_column") or None
        return px.scatter(df, x=args["x_column"], y=args["y_column"], color=color, title=args["title"])

    if tool_name == "create_histogram":
        return px.histogram(df, x=args["column"], title=args["title"])

    if tool_name == "create_box_plot":
        group = args.get("group_column") or None
        return px.box(df, x=group, y=args["value_column"], title=args["title"])

    raise ValueError(f"Unknown chart tool: {tool_name}")


# ----------------------------------------------------------------------
# LLM interaction helpers
# ----------------------------------------------------------------------

def get_client() -> anthropic.Anthropic:
    """Construct an Anthropic client.

    Checks the OS environment first (local development), then falls back
    to Streamlit's secrets store (Streamlit Community Cloud deployment).
    st.secrets raises StreamlitSecretNotFoundError when no secrets.toml
    exists at all, so the lookup is wrapped in try/except.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            key = None
    if not key:
        st.error(
            "ANTHROPIC_API_KEY is not set. For local runs, set it as an "
            "environment variable. For Streamlit Cloud, add it under "
            "App settings → Secrets."
        )
        st.stop()
    return anthropic.Anthropic(api_key=key)


def schema_summary(df: pd.DataFrame) -> str:
    """A compact schema description we can include in prompts."""
    lines = [f"Dataset has {len(df)} rows and {len(df.columns)} columns."]
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({df[col].dtype}): example values {sample}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def get_dataset_description(df_signature: str, schema_text: str, sample_csv: str) -> str:
    """One-paragraph plain-English description of the dataset, cached per dataset."""
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                "Here is a dataset schema and a small sample.\n\n"
                f"{schema_text}\n\n"
                f"Sample (CSV):\n{sample_csv}\n\n"
                "Write one short paragraph (3-4 sentences) describing what this dataset "
                "appears to contain. Be specific about what is being measured and any "
                "structure you can infer. Do not speculate beyond what the schema and "
                "sample support."
            ),
        }],
    )
    return response.content[0].text


@st.cache_data(show_spinner=False)
def get_data_quality_concerns(df_signature: str, schema_text: str, profile_text: str) -> list:
    """LLM-flagged data-quality concerns. Cached per dataset signature."""
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=(
            "You are a data quality reviewer. Talburt (2015) defines data quality as "
            "conformance to data specifications. Your job is to surface concerns that a "
            "downstream analyst or AI system should know about before relying on this data. "
            "Focus on: missing values, impossible or out-of-range values, type "
            "inconsistencies, duplicates, and anomalous distributions. Output a JSON list "
            "of concerns. Each concern is an object with keys 'severity' (one of: high, "
            "medium, low) and 'description' (one sentence). Output ONLY the JSON list, "
            "nothing else."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Schema:\n{schema_text}\n\n"
                f"Profile (descriptive statistics and counts):\n{profile_text}\n\n"
                "List the data quality concerns."
            ),
        }],
    )
    raw = response.content[0].text.strip()

    # Strip code fences if the LLM added them.
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [{"severity": "low", "description": f"Could not parse data quality response: {raw[:200]}"}]


def analyze_question(question: str, df: pd.DataFrame, schema_text: str):
    """Use tool calling to pick a chart, render it, then ask for a caption."""
    client = get_client()

    # Step 1: ask the model which chart to make.
    chart_response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        tools=CHART_TOOLS,
        messages=[{
            "role": "user",
            "content": (
                f"Dataset schema:\n{schema_text}\n\n"
                f"User question: {question}\n\n"
                "Choose exactly one chart tool that best answers this question. "
                "Reference only columns that appear in the schema. "
                "Pick column names exactly as shown in the schema (case and underscores matter)."
            ),
        }],
    )

    # Find the tool-use block in the response.
    tool_call = next((b for b in chart_response.content if b.type == "tool_use"), None)
    if tool_call is None:
        text_block = next((b for b in chart_response.content if b.type == "text"), None)
        message = text_block.text if text_block else "No chart could be generated for this question."
        return {"figure": None, "caption": message}

    # Step 2: render the chart deterministically in Python.
    try:
        figure = render_chart_from_tool_call(tool_call.name, dict(tool_call.input), df)
    except Exception as exc:
        return {"figure": None, "caption": f"Tried to build a {tool_call.name} but hit an error: {exc}"}

    # Step 3: ask for a one-paragraph narrative caption that explains what the chart shows.
    caption_response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"A user asked: '{question}'\n\n"
                f"In response, we produced a {tool_call.name} with these parameters:\n"
                f"{json.dumps(dict(tool_call.input), indent=2)}\n\n"
                f"Dataset summary:\n{schema_text}\n\n"
                "Write a short, plain-English caption (2-3 sentences) explaining what this "
                "chart shows and what an analyst might notice. Do not invent numbers you "
                "cannot see; describe what the chart is structured to reveal."
            ),
        }],
    )

    return {
        "figure": figure,
        "caption": caption_response.content[0].text,
        "tool_name": tool_call.name,
        "tool_args": dict(tool_call.input),
    }


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.title("📊 GenAI Visual Analytics")
st.markdown(
    "Upload a CSV, and an LLM helps you understand it — describing what it contains, "
    "flagging data-quality concerns, and turning your plain-English questions into charts "
    "with AI-narrated captions."
)

with st.sidebar:
    st.header("Data source")
    use_sample = st.checkbox("Use sample dataset (WA water monitoring)", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown(
        "Demonstrates LLM tool-use for chart selection, LLM-driven data-quality "
        "flagging, and AI-narrated visual analytics."
    )

# Load the dataframe.
if use_sample:
    df = generate_sample_dataset()
    st.caption("Showing synthetic Washington water-monitoring data with deliberate data-quality issues.")
elif uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV in the sidebar, or check the sample-dataset box to begin.")
    st.stop()

# ---- Section 1: Dataset overview ----
st.header("1. Dataset overview")
col_a, col_b = st.columns([3, 1])
with col_a:
    st.dataframe(df.head(10), use_container_width=True)
with col_b:
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", len(df.columns))
    st.metric("Missing values", int(df.isna().sum().sum()))

# Build the artifacts we'll reuse: schema text, sample CSV, profile text, signature.
schema_text = schema_summary(df)
sample_csv = df.head(5).to_csv(index=False)
profile_text = (
    f"Missing counts per column:\n{df.isna().sum().to_string()}\n\n"
    f"Numeric column statistics:\n{df.describe(include='all').to_string()}\n\n"
    f"Duplicate row count: {int(df.duplicated().sum())}"
)
df_signature = f"{list(df.columns)}|{len(df)}|{int(df.isna().sum().sum())}"

# ---- Section 2: AI-generated description ----
st.header("2. What does this data appear to be?")
with st.spinner("Asking the LLM to describe the dataset..."):
    description = get_dataset_description(df_signature, schema_text, sample_csv)
st.markdown(description)

# ---- Section 3: Data quality concerns ----
st.header("3. Data quality concerns")
st.caption(
    "Grounded in Talburt's definition of data quality as conformance to data specifications. "
    "Surfacing these before any analysis is the upstream awareness gap that organizations often miss."
)
with st.spinner("Asking the LLM to flag data quality concerns..."):
    concerns = get_data_quality_concerns(df_signature, schema_text, profile_text)
if not concerns:
    st.success("No significant data quality concerns identified.")
else:
    for concern in concerns:
        sev = (concern.get("severity") or "low").lower()
        message = concern.get("description", str(concern))
        if sev == "high":
            st.error(f"**High severity:** {message}")
        elif sev == "medium":
            st.warning(f"**Medium severity:** {message}")
        else:
            st.info(f"**Low severity:** {message}")

# ---- Section 4: Interactive analysis ----
st.header("4. Ask a question")
st.caption("Type a plain-English question about the data. The LLM picks a chart, renders it, and writes a caption.")

example_questions = {
    "WA water monitoring": [
        "How does temperature vary across the basins?",
        "Show the distribution of dissolved oxygen.",
        "Plot pH against turbidity, colored by basin.",
        "What is the trend of conductivity over time at each site?",
    ]
}

if use_sample:
    with st.expander("Example questions for the sample data"):
        for q in example_questions["WA water monitoring"]:
            st.markdown(f"- *{q}*")

user_question = st.text_input("Your question:", placeholder="e.g., How does temperature vary across the basins?")

if user_question:
    with st.spinner("Thinking..."):
        result = analyze_question(user_question, df, schema_text)

    if result.get("figure") is not None:
        st.plotly_chart(result["figure"], use_container_width=True)
        with st.expander("How was this chart selected? (LLM tool call detail)"):
            st.code(f"Tool: {result['tool_name']}\nArguments: {json.dumps(result['tool_args'], indent=2)}")

    st.markdown(f"**Caption:** {result['caption']}")
