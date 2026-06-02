import os
import atexit
from operator import itemgetter

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool

from langchain_classic.chains import create_sql_query_chain
from langchain_classic.chains.sql_database.prompt import PROMPT_SUFFIX

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

DB_URI = "sqlite:///atliq_tshirts.db"

db = SQLDatabase.from_uri(
    DB_URI,
    sample_rows_in_table_info=3,
    include_tables=["t_shirts", "discounts"],
)

print("--- Database connected ---")
print(f"Dialect : {db.dialect}")
print(f"Tables  : {db.get_usable_table_names()}")
print()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def strip_sql(raw: str) -> str:
    """Remove any label prefix (e.g. 'SQLQuery:') the LLM may prepend."""
    import re
    # Drop everything up to and including any 'SQLQuery:' label
    match = re.search(r'(?i)SQLQuery\s*:\s*', raw)
    if match:
        raw = raw[match.end():]
    # Also strip any leading 'Question: ...' line that leaked through
    raw = re.sub(r'(?i)^Question\s*:.*\n?', '', raw.strip())
    print(f"Stripped SQL: {raw}")
    return raw.strip()

def build_basic_chain(llm, db):
    """natural language → SQL → execute → plain-English answer"""
    write_query   = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    answer_prompt = PromptTemplate.from_template(
        "Based on the question, SQL query, and SQL result, answer the question.\n\n"
        "Question: {question}\n"
        "SQL Query: {query}\n"
        "SQL Result: {result}\n"
        "Answer: "
    )

    return (
        RunnablePassthrough.assign(query=write_query | strip_sql)
        .assign(result=itemgetter("query") | execute_query)
        | answer_prompt
        | llm
        | StrOutputParser()
    )


basic_chain = build_basic_chain(llm, db)


few_shots = [
    {
        "Question": "How many t-shirts do we have left for Nike in XS size and white color?",
        "SQLQuery": (
            "SELECT SUM(stock_quantity) FROM t_shirts "
            "WHERE brand='Nike' AND color='White' AND size='XS'"
        ),
        "SQLResult": "Result of the SQL query",
        "Answer": "91",
    },
    {
        "Question": "How much is the total price of the inventory for all S-size t-shirts?",
        "SQLQuery": "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE size = 'S'",
        "SQLResult": "Result of the SQL query",
        "Answer": "22292",
    },
    {
        "Question": (
            "If we have to sell all the Levi's T-shirts today with discounts applied, "
            "how much revenue will our store generate?"
        ),
        "SQLQuery": (
            "SELECT SUM(a.total_amount * ((100 - COALESCE(d.pct_discount, 0)) / 100)) AS revenue "
            "FROM ("
            "    SELECT SUM(price * stock_quantity) AS total_amount, t_shirt_id "
            "    FROM t_shirts WHERE brand = 'Levi' GROUP BY t_shirt_id"
            ") a LEFT JOIN discounts d ON a.t_shirt_id = d.t_shirt_id"
        ),
        "SQLResult": "Result of the SQL query",
        "Answer": "16725.4",
    },
    {
        "Question": (
            "If we have to sell all the Levi's T-shirts today without discount, "
            "how much revenue will our store generate?"
        ),
        "SQLQuery": "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
        "SQLResult": "Result of the SQL query",
        "Answer": "17462",
    },
    {
        "Question": "How many white color Levi's shirts do I have?",
        "SQLQuery": (
            "SELECT SUM(stock_quantity) FROM t_shirts "
            "WHERE brand='Levi' AND color='White'"
        ),
        "SQLResult": "Result of the SQL query",
        "Answer": "290",
    },
]



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

to_vectorize = [" ".join(ex.values()) for ex in few_shots]
vectorstore  = QdrantVectorStore.from_texts(to_vectorize, embeddings, metadatas=few_shots, path="./VECTOR_DB")
atexit.register(vectorstore.client.close)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

print("--- Example selector smoke test ---")
selected = example_selector.select_examples(
    {"Question": "How many Adidas T-shirts do I have left in my store?"}
)
for ex in selected:
    print(f"  -> {ex['Question']}")
print()


example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
    template=(
        "\nQuestion: {Question}"
        "\nSQLQuery: {SQLQuery}"
        "\nSQLResult: {SQLResult}"
        "\nAnswer: {Answer}"
    ),
)

mysql_prompt = """\
You are a sqlite expert. Given an input question, first create a syntactically \
correct sqlite query to run, then look at the results of the query and return \
the answer to the input question.

Unless the user specifies a number of examples, query for at most {top_k} \
results using the LIMIT clause.

Never query for all columns from a table. Only query the columns needed to \
answer the question. Wrap each column name in backticks (`) to denote them as \
delimited identifiers.

Pay attention to use only column names that exist in the tables below. \
Use CURDATE() to get today's date if the question involves "today".

Use the following format:

Question: <question here>
SQLQuery: <query to run, no preamble>
SQLResult: <result of the query>
Answer: <final answer here>

No preamble.
"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"],
)

def build_few_shot_chain(llm, db, few_shot_prompt):
    write_query   = create_sql_query_chain(llm, db, prompt=few_shot_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)

    answer_prompt = PromptTemplate.from_template(
        "Based on the question, SQL query, and SQL result, answer the question.\n\n"
        "Question: {question}\n"
        "SQL Query: {query}\n"
        "SQL Result: {result}\n"
        "Answer: "
    )

    return (
        RunnablePassthrough.assign(query=write_query | strip_sql)
        .assign(result=itemgetter("query") | execute_query)
        | answer_prompt
        | llm
        | StrOutputParser()
    )


few_shot_chain = build_few_shot_chain(llm, db, few_shot_prompt)


question = "How many white color Levi's shirts do I have?"
print(f"Question           : {question}")
answer = few_shot_chain.invoke({"question": question})
print(f"Answer           : {answer}")