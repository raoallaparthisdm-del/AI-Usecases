import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from kb import AGE_BINS, AGE_LABELS, build_knowledge_base, build_fact_list
from retriever import SimpleRetriever
from rag import RAGEngine
from llm import OpenAILLM


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    return df


def build_rag_engine() -> tuple[RAGEngine, dict]:
    load_dotenv()
    data_path = Path(__file__).parent / "sales_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    df = load_data(data_path)
    kb = build_knowledge_base(df)
    facts = build_fact_list(kb)
    retriever = SimpleRetriever(facts)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. Please set it before running this script."
        )

    try:
        llm = OpenAILLM()
    except Exception as exc:  # noqa: BLE001
        print("Failed to initialize OpenAI client.")
        print("Set OPENAI_API_KEY and try again.")
        raise exc

    rag = RAGEngine(retriever=retriever, llm=llm)
    return rag, kb


def build_eval_examples(kb: dict) -> list[dict]:
    top_product = kb["product_sales"].index[0]
    top_product_sales = kb["product_sales"].iloc[0]
    top_region = kb["region_sales"].index[0]
    top_region_sales = kb["region_sales"].iloc[0]
    stats = kb["stats"]
    meta = kb["meta"]

    return [
        {
            "query": "What is the dataset date range?",
            "answer": f"{meta['start_date']} to {meta['end_date']}",
        },
        {
            "query": "How many rows are in the dataset?",
            "answer": str(meta["rows"]),
        },
        {
            "query": f"What is total sales for {top_product}?",
            "answer": f"{top_product_sales:.2f}",
        },
        {
            "query": f"What is total sales for the {top_region} region?",
            "answer": f"{top_region_sales:.2f}",
        },
        {
            "query": "What is the average customer satisfaction?",
            "answer": f"{stats['satisfaction_mean']:.2f}",
        },
        {
            "query": "What is the average customer age?",
            "answer": f"{stats['age_mean']:.2f}",
        },
    ]


def _extract_grade_text(grade: dict | str) -> str:
    if isinstance(grade, str):
        return grade
    for key in ("text", "results", "result", "output"):
        if key in grade and grade[key]:
            return str(grade[key])
    return str(grade)


def run_qa_evaluation(rag: RAGEngine, kb: dict) -> pd.DataFrame:
    try:
        from langchain.evaluation import QAEvalChain
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "QAEvalChain requires langchain and langchain-openai. "
            "Install them with 'pip install langchain langchain-openai'."
        ) from exc

    examples = build_eval_examples(kb)
    predictions = []
    for example in examples:
        prediction = rag.answer(example["query"], memory=[])
        predictions.append({"result": prediction})

    eval_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    chain = QAEvalChain.from_llm(eval_llm)
    graded = chain.evaluate(
        examples,
        predictions,
        question_key="query",
        answer_key="answer",
        prediction_key="result",
    )

    rows = []
    for example, prediction, grade in zip(examples, predictions, graded):
        grade_text = _extract_grade_text(grade)
        grade_upper = grade_text.upper()
        if "CORRECT" in grade_upper or "YES" in grade_upper:
            score = 1
        elif "INCORRECT" in grade_upper or "NO" in grade_upper:
            score = 0
        else:
            score = None

        rows.append(
            {
                "question": example["query"],
                "reference": example["answer"],
                "prediction": prediction["result"],
                "grade": grade_text,
                "score": score,
            }
        )

    return pd.DataFrame(rows)


def main_cli() -> None:
    rag, kb = build_rag_engine()

    print("BI Assistant ready. Ask a question or type 'exit'.")
    print(f"Dataset range: {kb['meta']['start_date']} to {kb['meta']['end_date']}")
    print(f"Rows: {kb['meta']['rows']}, Columns: {kb['meta']['columns']}")

    memory = []
    while True:
        user_input = input("\nQuestion> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        response = rag.answer(user_input, memory=memory)
        print("\n" + response)

        memory.append({"role": "user", "content": user_input})
        memory.append({"role": "assistant", "content": response})


def main_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="BI Assistant", layout="wide")
    st.title("BI Assistant")
    st.caption("Ask questions about the sales dataset using RAG.")

    if "rag" not in st.session_state:
        try:
            rag, kb = build_rag_engine()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()
        st.session_state.rag = rag
        st.session_state.kb = kb
        st.session_state.memory = []

    if "dataframe" not in st.session_state:
        data_path = Path(__file__).parent / "sales_data.csv"
        st.session_state.dataframe = load_data(data_path)

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"

    with st.sidebar:
        st.subheader("Dataset")
        kb = st.session_state.kb
        st.write(f"Range: {kb['meta']['start_date']} to {kb['meta']['end_date']}")
        st.write(f"Rows: {kb['meta']['rows']}")
        st.write(f"Columns: {kb['meta']['columns']}")
        if st.button("Go to Evaluation"):
            st.session_state.active_tab = "Evaluation"
            st.rerun()
        if st.button("Clear chat"):
            st.session_state.memory = []
            st.rerun()

    tab_options = ["Chat", "Visualizations", "Evaluation"]
    active_tab = st.radio(
        "View",
        tab_options,
        index=tab_options.index(st.session_state.active_tab)
        if st.session_state.active_tab in tab_options
        else 0,
        horizontal=True,
        key="active_tab",
    )

    if active_tab == "Chat":
        for message in st.session_state.memory:
            role = message.get("role", "assistant")
            with st.chat_message(role):
                st.markdown(message.get("content", ""))

        prompt = st.chat_input("Ask a question about sales data")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag.answer(
                        prompt,
                        memory=st.session_state.memory,
                    )
                st.markdown(response)

            st.session_state.memory.append({"role": "user", "content": prompt})
            st.session_state.memory.append({"role": "assistant", "content": response})

    elif active_tab == "Visualizations":
        df = st.session_state.dataframe.copy()
        df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        monthly_sales = df.groupby("Month", as_index=False)["Sales"].sum()
        product_sales = df.groupby("Product", as_index=False)["Sales"].sum().sort_values(
            "Sales",
            ascending=False,
        )
        region_sales = df.groupby("Region", as_index=False)["Sales"].sum().sort_values(
            "Sales",
            ascending=False,
        )
        df["Age_Group"] = pd.cut(df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        segment = (
            df.groupby(["Age_Group", "Customer_Gender"], as_index=False)
            .agg(
                count=("Sales", "count"),
                avg_sales=("Sales", "mean"),
                avg_satisfaction=("Customer_Satisfaction", "mean"),
            )
            .sort_values("count", ascending=False)
        )

        try:
            import plotly.express as px
        except Exception:  # noqa: BLE001
            st.warning("Plotly is not installed. Showing basic charts instead.")
            st.subheader("Sales trends over time")
            st.line_chart(monthly_sales.set_index("Month")["Sales"])
            st.subheader("Product performance")
            st.bar_chart(product_sales.set_index("Product")["Sales"])
            st.subheader("Regional analysis")
            st.bar_chart(region_sales.set_index("Region")["Sales"])
        else:
            st.subheader("Sales trends over time")
            trend_chart = px.line(
                monthly_sales,
                x="Month",
                y="Sales",
                markers=True,
                title="Monthly Sales",
            )
            st.plotly_chart(trend_chart, use_container_width=True)

            st.subheader("Product performance comparisons")
            product_chart = px.bar(
                product_sales,
                x="Product",
                y="Sales",
                title="Sales by Product",
            )
            st.plotly_chart(product_chart, use_container_width=True)

            st.subheader("Regional analysis")
            region_chart = px.bar(
                region_sales,
                x="Region",
                y="Sales",
                title="Sales by Region",
            )
            st.plotly_chart(region_chart, use_container_width=True)

            st.subheader("Customer demographics and segmentation")
            demo_cols = st.columns(2)
            with demo_cols[0]:
                age_chart = px.histogram(
                    df,
                    x="Customer_Age",
                    nbins=12,
                    title="Customer Age Distribution",
                )
                st.plotly_chart(age_chart, use_container_width=True)

            with demo_cols[1]:
                gender_chart = px.pie(
                    df,
                    names="Customer_Gender",
                    title="Customer Gender Split",
                )
                st.plotly_chart(gender_chart, use_container_width=True)

            segment_chart = px.density_heatmap(
                segment,
                x="Age_Group",
                y="Customer_Gender",
                z="count",
                histfunc="sum",
                title="Customer Segments by Age and Gender",
            )
            st.plotly_chart(segment_chart, use_container_width=True)

    else:
        st.subheader("Model evaluation")
        st.write("Run QAEvalChain to grade model answers against known references.")

        if st.button("Run evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    results = run_qa_evaluation(st.session_state.rag, st.session_state.kb)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                else:
                    st.session_state.eval_results = results

        if "eval_results" in st.session_state:
            results = st.session_state.eval_results
            avg_score = results["score"].dropna().mean() if not results.empty else None
            if avg_score is not None:
                st.metric("Average score", f"{avg_score:.2f}")
            st.dataframe(results, use_container_width=True)


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


def _launch_streamlit() -> None:
    script_path = Path(__file__).resolve()
    command = [sys.executable, "-m", "streamlit", "run", str(script_path)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    if _running_in_streamlit():
        main_streamlit()
    else:
        parser = argparse.ArgumentParser(description="BI Assistant")
        parser.add_argument(
            "--ui",
            action="store_true",
            help="Launch the Streamlit web UI",
        )
        args = parser.parse_args()

        if args.ui:
            _launch_streamlit()
        else:
            main_cli()
