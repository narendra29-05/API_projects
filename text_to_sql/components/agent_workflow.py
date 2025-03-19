from typing import TypedDict, List, Annotated
import operator
import uuid
import re

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Define the state that will be passed between agents
class AgentState(TypedDict):
    question: str
    table_schemas: str
    database: str
    sql: str
    reflect: Annotated[List[str], operator.add]
    accepted: bool
    revision: int
    max_revision: int
    csv_files: List[str]
    db_path: str

# Define agent roles and prompts
agent_roles = {
    "schema_finder": {
        "system": """You are a database schema expert who identifies relevant tables and fields needed to answer a question. You analyze the available tables and determine which ones contain the necessary information.
        
When examining the schema, consider:
1. Which tables contain fields directly mentioned in the question
2. Which tables might need to be joined to provide complete answers
3. Which fields will be needed for filtering, grouping, or in the final output""",
        "expected_output": "Identify only the most relevant tables and columns for this specific question. Be concise and precise."
    },
    "sql_writer": {
        "system": """You are an expert SQLite developer who writes precise SQL queries based on database schemas. Your queries should be optimized and follow SQLite-specific syntax rules.
        
CRITICAL SQLite-specific rules to follow:
1. PRAGMA statements are not tables and cannot be used in FROM clauses.
   - INCORRECT: `SELECT * FROM PRAGMA table_info('tablename')`
   - CORRECT:   `PRAGMA table_info('tablename')`
   
2. To count columns in a table, use one of these approaches:
   - Use `PRAGMA table_info('tablename')` and count the rows
   - Query the sqlite_master table: `SELECT sql FROM sqlite_master WHERE type='table' AND name='tablename'`
   
3. Always use single quotes for string literals, not double quotes.
   - INCORRECT: `WHERE name = "value"`
   - CORRECT:   `WHERE name = 'value'`
   
4. SQLite has limited date/time functions compared to other databases:
   - Use functions like date(), time(), datetime(), strftime()
   - Format strings like '%Y-%m-%d' for dates

5. For aggregations and window functions:
   - SQLite supports COUNT, SUM, AVG, MIN, MAX
   - Window functions require SQLite 3.25+ (OVER clause)
   
6. Be mindful of column and table names that might need to be quoted with [square brackets] if they contain spaces or special characters.

7. To list all tables:
   - `SELECT name FROM sqlite_master WHERE type='table'`""",
        "expected_output": "Write a complete and correct SQLite query that answers the given question. Use only the tables and columns available in the schema."
    },
    "sql_validator": {
        "system": """You are a SQLite validation expert who checks if queries correctly answer the given question. You verify syntax, logic, and whether the query will return the information requested.
        
When validating, be extremely vigilant about these common SQLite errors:
1. PRAGMA misuse - PRAGMA commands cannot be used inside regular SQL queries as tables
   - INCORRECT: `SELECT * FROM PRAGMA table_info('tablename')` 
   - CORRECT:   `PRAGMA table_info('tablename')` (as a standalone command)

2. Using functions not available in SQLite
   - No support for PIVOT, MEDIAN, STDDEV
   - Limited string and date manipulation functions
   
3. Syntax errors specific to SQLite:
   - Use single quotes for strings: `WHERE name = 'value'`
   - SQLite's LIMIT syntax differs from other databases
   
4. Schema references:
   - No information_schema tables exist in SQLite
   - Tables created from CSV imports may need special handling
   
5. Check if columns being referenced actually exist in the schema

Before accepting any query, verify that it doesn't contain any of these issues.""",
        "expected_output": """Review the SQL query thoroughly. Respond with 'ACCEPTED' if the query is correct and will answer the question. Otherwise, explain the specific issues that need to be fixed.

If there are issues with PRAGMA usage, explicitly state that PRAGMA commands can't be used as tables in FROM clauses. Be specific about what exactly needs to be changed."""
    },
    "sql_improver": {
        "system": """You are a senior SQLite database administrator who provides specific feedback to improve SQL queries. You focus on correctness, performance, and alignment with the original question.
        
When improving queries, focus on these critical SQLite-specific issues:

1. If you see `SELECT ... FROM PRAGMA table_info(...)`, this is completely invalid! 
   - PRAGMA is a special command, not a table to select from
   - Instead use `PRAGMA table_info(tablename)` as a standalone command
   - Or use `SELECT count(*) FROM pragma_table_info(tablename)` to count columns
   
2. Handling metadata queries properly:
   - Use `SELECT name FROM sqlite_master WHERE type='table'` to list tables
   - Use `PRAGMA table_info(tablename)` to get column information
   
3. SQLite performance best practices:
   - Ensure proper indexing for large tables
   - Avoid unnecessary subqueries
   - Use LIMIT to restrict result sets
   
4. SQLite function limitations:
   - Suggest alternative approaches when non-supported functions are used
   - Use SQLite's built-in functions appropriately""",
        "expected_output": """Provide detailed, actionable feedback to fix any issues with the SQL query. Be specific about what needs to change and why.

If the query uses PRAGMA incorrectly (trying to select from it as a table), explicitly explain that PRAGMA commands are special directives in SQLite and cannot be used as tables in FROM clauses."""
    }
}

# Extract SQL from response
def extract_sql_from_response(response_text):
    # Try to extract SQL from code blocks first
    if "```sql" in response_text:
        sql = response_text.split("```sql")[1].split("```")[0].strip()
    elif "```" in response_text:
        sql = response_text.split("```")[1].split("```")[0].strip()
    else:
        # If no code blocks, try to extract based on common patterns
        lines = response_text.split('\n')
        sql_lines = []
        capturing = False
        
        for line in lines:
            # Look for likely SQL statement starts
            if re.search(r'^\s*(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE|PRAGMA)\s', line, re.IGNORECASE) and not capturing:
                capturing = True
                sql_lines.append(line)
            elif capturing and line.strip():
                sql_lines.append(line)
            elif capturing and not line.strip():
                # Empty line might end the SQL
                capturing = False
        
        sql = '\n'.join(sql_lines).strip()
    
    # Final cleanup
    sql = re.sub(r'[\n\r]+', ' ', sql)  # Replace multiple newlines with spaces
    sql = re.sub(r'\s{2,}', ' ', sql)   # Replace multiple spaces with single space
    
    return sql

# Define the agent nodes
def schema_finder_node(state: AgentState, model):
    messages = [
        SystemMessage(content=agent_roles['schema_finder']['system']),
        HumanMessage(
            content=f"Based on these database schemas:\n{state['table_schemas']}\n\n"
                   f"Find the relevant schemas to answer this question: {state['question']}\n\n"
                   f"Consider which tables and columns would be needed for this specific query.\n\n"
                   f"{agent_roles['schema_finder']['expected_output']}")
    ]
    response = model.invoke(messages)
    
    return {
        "table_schemas": state['table_schemas'],  # Keep the original schema
        "database": state['db_path']  # Keep the database path
    }

def sql_writer_node(state: AgentState, model):
    instruction = f"Using these database schemas:\n{state['table_schemas']}\n\n"
    
    instruction += "VERY IMPORTANT SQLite RULES:\n"
    instruction += "1. PRAGMA statements are NOT tables and CANNOT be used in FROM clauses!\n"
    instruction += "   - INCORRECT: SELECT * FROM PRAGMA table_info('tablename')\n"
    instruction += "   - CORRECT: PRAGMA table_info('tablename')\n"
    instruction += "   - To count columns: SELECT COUNT(*) FROM pragma_table_info('tablename')\n"
    instruction += "2. Use single quotes for string literals, not double quotes\n"
    instruction += "3. SQLite does NOT support information_schema tables\n\n"
    
    if state['reflect']:
        instruction += f"Consider this feedback from previous attempts:\n{state['reflect'][-1]}\n\n"
        
    instruction += f"Write a SQLite-compatible SQL query to answer: {state['question']}\n\n{agent_roles['sql_writer']['expected_output']}"
    
    messages = [
        SystemMessage(content=agent_roles['sql_writer']['system']),
        HumanMessage(content=instruction)
    ]
    response = model.invoke(messages)
    
    # Extract the SQL from the response
    sql_response = extract_sql_from_response(response.content)
    
    return {
        "sql": sql_response,
        "revision": state['revision'] + 1
    }

def sql_validator_node(state: AgentState, model):
    messages = [
        SystemMessage(content=agent_roles['sql_validator']['system']),
        HumanMessage(
            content=f"Database schemas:\n{state['table_schemas']}\n\n"
                   f"SQL query to validate:\n{state['sql']}\n\n"
                   f"Check if this SQLite query correctly answers: {state['question']}\n\n"
                   f"CRITICAL CHECKS:\n"
                   f"- Is it using PRAGMA incorrectly as a table? (e.g., SELECT * FROM PRAGMA...)\n"
                   f"- Is it using functions not available in SQLite?\n"
                   f"- Is it using proper SQLite syntax for metadata queries?\n"
                   f"- Are all referenced tables and columns present in the schema?\n\n"
                   f"{agent_roles['sql_validator']['expected_output']}")
    ]
    response = model.invoke(messages)
    
    # Check if the response indicates acceptance
    response_text = response.content.upper()
    accepted = ('ACCEPTED' in response_text and 
                'NOT ACCEPTED' not in response_text and 
                'UNACCEPTED' not in response_text)
    
    return {
        "accepted": accepted
    }

def sql_improver_node(state: AgentState, model):
    messages = [
        SystemMessage(content=agent_roles['sql_improver']['system']),
        HumanMessage(
            content=f"Database schemas:\n{state['table_schemas']}\n\n"
                   f"Current SQL query with issues:\n{state['sql']}\n\n"
                   f"Provide specific feedback to better answer: {state['question']}\n\n"
                   f"CRITICAL SQLite ISSUES TO CHECK:\n"
                   f"1. Is PRAGMA being used incorrectly? (e.g., SELECT * FROM PRAGMA...)\n"
                   f"2. Are there any SQLite syntax errors?\n"
                   f"3. Is the query using functions not available in SQLite?\n"
                   f"4. Does the query correctly reference the available tables and columns?\n\n"
                   f"{agent_roles['sql_improver']['expected_output']}")
    ]
    response = model.invoke(messages)
    
    return {
        "reflect": [response.content]
    }

# Build the graph
def build_text2sql_graph(model_name, temperature=0.0):
    # Initialize LLM
    model = ChatGroq(
        model_name=model_name,
        temperature=temperature
    )
    
    # Create wrapper functions that include the model
    def schema_finder_with_model(state):
        return schema_finder_node(state, model)
    
    def sql_writer_with_model(state):
        return sql_writer_node(state, model)
    
    def sql_validator_with_model(state):
        return sql_validator_node(state, model)
    
    def sql_improver_with_model(state):
        return sql_improver_node(state, model)
    
    # Build the graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("schema_finder", schema_finder_with_model)
    builder.add_node("sql_writer", sql_writer_with_model)
    builder.add_node("sql_validator", sql_validator_with_model)
    builder.add_node("sql_improver", sql_improver_with_model)
    
    # Add edges
    builder.add_edge("schema_finder", "sql_writer")
    builder.add_edge("sql_writer", "sql_validator")
    builder.add_edge("sql_improver", "sql_writer")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "sql_validator",
        lambda state: END if state['accepted'] or state['revision'] >= state['max_revision'] else "improve",
        {END: END, "improve": "sql_improver"}
    )
    
    # Set entry point
    builder.set_entry_point("schema_finder")
    
    # Use in-memory checkpoint
    memory = MemorySaver()
    
    # Compile graph
    return builder.compile(checkpointer=memory)