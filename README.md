# BI Assistant (RAG)

This CLI loads the provided sales dataset, builds a lightweight knowledge base, and answers questions using a RAG flow.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Create a .env file in the project root:

```
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

If you prefer system environment variables instead, you can run:

```
setx OPENAI_API_KEY "your-key"
setx OPENAI_MODEL "gpt-4o-mini"
```

## Run

```
python app.py
```

## Streamlit UI

```
streamlit run app.py
```

## Example questions

- What are total sales by month in 2023?
- Which product has the highest sales overall?
- Compare North vs South regions.
- What is the median sales value?
- Show customer segments by age group and gender.
