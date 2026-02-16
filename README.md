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

## Logic flow (data + RAG)

```mermaid
flowchart TD
	A([Start]) --> B[load_dotenv()]
	B --> C[Resolve sales_data.csv path]
	C --> D{CSV exists?}
	D -- No --> E[Raise FileNotFoundError]
	D -- Yes --> F[load_data()]
	F --> G[pd.read_csv]
	G --> H[pd.to_datetime(Date)]
	H --> I[build_knowledge_base(df)]
	I --> J[build_fact_list(kb)]
	J --> K[SimpleRetriever(facts)]
	K --> L{OPENAI_API_KEY set?}
	L -- No --> M[Raise ValueError]
	L -- Yes --> N[OpenAILLM()]
	N --> O[RAGEngine(retriever, llm)]
	O --> P[Ready: rag, kb]

	subgraph RAG_Answer_Path
		Q[User query] --> R[rag.answer(query, memory)]
		R --> S[Retriever selects relevant facts]
		S --> T[LLM generates response]
		T --> U[Return answer]
	end

	P --> Q
```

## Example questions

- What are total sales by month in 2023?
- Which product has the highest sales overall?
- Compare North vs South regions.
- What is the median sales value?
- Show customer segments by age group and gender.
