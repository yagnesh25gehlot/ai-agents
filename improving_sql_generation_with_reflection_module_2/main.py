import json
import utils
import pandas as pd
from dotenv import load_dotenv


_ = load_dotenv()

import aisuite as ai
client = ai.Client()
from llm.grok_client import LLMClient


''''
• id → unique event ID (autoincrement).
• product_id, product_name, brand, category, color → identify the product.
• action → type of event (insert, restock, sale, price_update).
• qty_delta → stock change (+ for insert/restock, – for sale, 0 for price updates).
• unit_price → price at that moment (NULL for restock).
• notes → optional description of the event.
• ts → timestamp when the event was logged.
'''
utils.create_transactions_db()
utils.insert_sample_data()
utils.print_html_sql(utils.get_schema('products.db'))



def generate_sql(question: str, schema: str, model: str = None) -> str:
    prompt = f"""
You are an expert SQL assistant.

Task:
- Generate a valid SQLite SQL query
- Use ONLY the tables and columns provided in the schema
- The query MUST run directly in SQLite

STRICT RULES:
- DO NOT include markdown
- DO NOT include ```sql or ```
- DO NOT include explanations, comments, or extra text
- Output ONLY raw SQL

Schema:
{schema}

Question:
{question}

Output:
"""

    res = LLMClient.get_response(prompt).content.strip()
    return res





# Example usage of generate_sql

# We provide the schema as a string
schema = """
Table name: transactions
id (INTEGER)
product_id (INTEGER)
product_name (TEXT)
brand (TEXT)
category (TEXT)
color (TEXT)
action (TEXT)
qty_delta (INTEGER)
unit_price (REAL)
notes (TEXT)
ts (DATETIME)
"""

# We ask a question about the data in natural language
question = "Which color of product has the highest total sales?"

#utils.print_html(question, title="User Question")

# Generate the SQL query using the specified model
#sql_V1 = generate_sql(question, schema, model="openai:gpt-4.1")

# Display the generated SQL query
#utils.print_html(sql_V1, title="SQL Query V1")





# Execute the generated SQL query (sql_V1) against the products.db database.
# The result is returned as a pandas DataFrame.
#df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')

# Render the DataFrame as an HTML table in the notebook.
# This makes the query output easier to read and interpret.
#utils.print_html(df_sql_V1, title="Output of SQL Query V1 - ❌ Does NOT fully answer the question")








def refine_sql(
    question: str,
    sql_query: str,
    schema: str,
    model: str,
) -> tuple[str, str]:
    """
    Reflect on whether a query's *shown output* answers the question,
    and propose an improved SQL if needed.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
You are a SQL reviewer and refiner.

User asked:
{question}

Original SQL:
{sql_query}

Table Schema:
{schema}

Step 1: Briefly evaluate if the SQL OUTPUT fully answers the user's question.
Step 2: If improvement is needed, provide a refined SQL query for SQLite.
If the original SQL is already correct, return it unchanged.

Return STRICT JSON with two fields:
{{
  "feedback": "<1-3 sentences explaining the gap or confirming correctness>",
  "refined_sql": "<final SQL to run>"
}}
"""
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0,
    # )
    content = LLMClient.get_response(prompt).content.strip()
    # content = response.choices[0].message.content
    try:
        obj = json.loads(content)
        feedback = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()
        if not refined_sql:
            refined_sql = sql_query
    except Exception:
        # Fallback if model doesn't return valid JSON
        feedback = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql









# Example: refine the generated SQL (V1 → V2)

# feedback, sql_V2 = refine_sql(
#     question=question,
#     sql_query=sql_V1,   # <- comes from generate_sql() (V1)
#     schema=schema, # <- we reuse the schema from section 3.1
#     model="openai:gpt-4.1"
# )

# Display the original prompt
# utils.print_html(question, title="User Question")

# --- V1 ---
# utils.print_html(sql_V1, title="Generated SQL Query (V1)")

# Execute and show V1 output
# df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')
# utils.print_html(df_sql_V1, title="SQL Output of V1 - ❌ Does NOT fully answer the question")

# --- Feedback + V2 ---
# utils.print_html(feedback, title="Feedback on V1")
# utils.print_html(sql_V2, title="Refined SQL Query (V2)")

# Execute and show V2 output
# df_sql_V2 = utils.execute_sql(sql_V2, db_path='products.db')
# utils.print_html(df_sql_V2, title="SQL Output of V2 - ❌ Does NOT fully answer the question")







def refine_sql_external_feedback(
        question: str,
        sql_query: str,
        df_feedback: pd.DataFrame,
        schema: str,
        model: str,
) -> tuple[str, str]:
    """
    Evaluate whether the SQL result answers the user's question and,
    if necessary, propose a refined version of the query.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
    You are a SQL reviewer and refiner.

    Your job is to evaluate and possibly improve a SQL query.

    INPUTS:
    - User Question
    - Original SQL
    - SQL Execution Output
    - Table Schema

    TASK:
    1. Evaluate whether the SQL output correctly answers the user's question.
    2. If improvements are required, generate ONE improved SQL query.
    3. If no improvement is required, return the original SQL unchanged.

    CRITICAL OUTPUT RULES (MUST FOLLOW):
    - Return EXACTLY ONE JSON object
    - DO NOT include markdown
    - DO NOT include ``` or ```json
    - DO NOT include explanations outside JSON
    - DO NOT include multiple options
    - DO NOT repeat the answer
    - DO NOT add any text before or after JSON
    - refined_sql MUST be executable SQLite SQL
    - Use window functions ONLY if supported by SQLite

    JSON FORMAT (STRICT):
    {{
      "feedback": {{
        "evaluation": "<1–2 sentence evaluation>",
        "suggestions": "<1 sentence suggestion or 'No changes required'>"
      }},
      "refined_sql": "<final SQL query>"
    }}

    User Question:
    {question}

    Original SQL:
    {sql_query}

    SQL Output:
    {df_feedback.to_markdown(index=False)}

    Table Schema:
    {schema}

    Return ONLY the JSON object.
    """

    content =LLMClient.get_response(prompt).content.strip()
    try:
        obj = json.loads(content)
        feedback = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()
        if not refined_sql:
            refined_sql = sql_query
    except Exception:
        # Fallback if the model does not return valid JSON:
        # use the raw content as feedback and keep the original SQL
        feedback = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql










# Example: Refine SQL with External Feedback (V1 → V2)

# Execute the original SQL (V1)
# df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')

# Use external feedback to evaluate and refine
# feedback, sql_V2 = refine_sql_external_feedback(
#     question=question,
#     sql_query=sql_V1,   # V1 query
#     df_feedback=df_sql_V1,    # Output of V1
#     schema=schema,
#     model="openai:gpt-4.1"
# )

# --- V1 ---
# utils.print_html(question, title="User Question")
# utils.print_html(sql_V1, title="Generated SQL Query (V1)")
# utils.print_html(df_sql_V1, title="SQL Output of V1 - ❌ Does NOT fully answer the question")
#
# # --- Feedback & V2 ---
# utils.print_html(feedback, title="Feedback on V1")
# utils.print_html(sql_V2, title="Refined SQL Query (V2)")
#
# # Execute and display V2 results
# df_sql_V2 = utils.execute_sql(sql_V2, db_path='products.db')
# utils.print_html(df_sql_V2, title="SQL Output of V2 (with External Feedback) - ✅ Fully answers the question")







def run_sql_workflow(
    db_path: str,
    question: str,
    model_generation: str = "openai:gpt-4.1",
    model_evaluation: str = "openai:gpt-4.1",
):
    """
    End-to-end workflow to generate, execute, evaluate, and refine SQL queries.

    Steps:
      1) Extract database schema
      2) Generate SQL (V1)
      3) Execute V1 → show output
      4) Reflect on V1 with execution feedback → propose refined SQL (V2)
      5) Execute V2 → show final answer
    """

    # 1) Schema
    schema = utils.get_schema(db_path)
    utils.print_html(
        schema,
        title="📘 Step 1 — Extract Database Schema"
    )

    # 2) Generate SQL (V1)
    sql_v1 = generate_sql(question, schema, model_generation)
    utils.print_html(
        sql_v1,
        title="🧠 Step 2 — Generate SQL (V1)"
    )

    # 3) Execute V1
    df_v1 = utils.execute_sql(sql_v1, db_path)
    utils.print_html(
        df_v1,
        title="🧪 Step 3 — Execute V1 (SQL Output)"
    )

    # 4) Reflect on V1 with execution feedback → refine to V2
    feedback, sql_v2 = refine_sql_external_feedback(
        question=question,
        sql_query=sql_v1,
        df_feedback=df_v1,          # external feedback: real output of V1
        schema=schema,
        model=model_evaluation,
    )
    utils.print_html(
        feedback,
        title="🧭 Step 4 — Reflect on V1 (Feedback)"
    )
    utils.print_html(
        sql_v2,
        title="🔁 Step 4 — Refined SQL (V2)"
    )

    # 5) Execute V2
    df_v2 = utils.execute_sql(sql_v2, db_path)
    utils.print_html(
        df_v2,
        title="✅ Step 5 — Execute V2 (Final Answer)"
    )




if __name__ == '__main__':
    run_sql_workflow('products.db', "Which color of product has the highest total sales?")


