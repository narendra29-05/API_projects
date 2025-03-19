import streamlit as st
import pandas as pd
import os
import warnings
import uuid
from pathlib import Path

from components.agent_workflow import build_text2sql_graph, AgentState
from components.db_utils import create_db_from_csvs, execute_sql
from components.ui_components import render_sidebar, render_data_preview, render_schema_view

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Text to SQL Agent",
    page_icon="🤖",
    layout="wide"
)

# Application title and description
st.title("🤖 Text to SQL Agent")
st.markdown("""
Upload CSV files and ask questions in natural language to get SQL queries and results.
This application uses LangGraph with a Groq-powered LLM to convert your questions to SQL.
""")

# Create data directory for uploads if it doesn't exist
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Render sidebar with configuration options
api_key, model_name, max_revisions, temperature = render_sidebar()

# Main area for file upload and question input
st.header("📁 Upload Your CSV Files")
uploaded_files = st.file_uploader("Upload one or more CSV files", type="csv", accept_multiple_files=True)

# Initialize session state for database info
if 'db_info' not in st.session_state:
    st.session_state.db_info = None
    st.session_state.table_schemas = None
    st.session_state.uploaded = False
    st.session_state.csv_files = []
    st.session_state.file_paths = []
    st.session_state.dataframes = {}

# Process uploaded files
if uploaded_files and not st.session_state.uploaded:
    # Process uploads and create database
    db_path, table_schemas, dataframes = create_db_from_csvs(uploaded_files, DATA_DIR)
    
    # Store in session state
    st.session_state.db_info = db_path
    st.session_state.table_schemas = table_schemas
    st.session_state.dataframes = dataframes
    st.session_state.uploaded = True
    
    st.success(f"Successfully loaded {len(uploaded_files)} CSV files into SQLite database")

# Display loaded files
if st.session_state.uploaded:
    st.header("📊 Loaded Data")
    
    # Display data preview with statistics
    render_data_preview(st.session_state.dataframes)
    
    # Display tables and schemas
    render_schema_view(st.session_state.table_schemas)
    
    # Question input
    st.header("❓ Ask a Question")
    question = st.text_input("Enter your question in natural language")
    
    # Process question when submitted
    if question and api_key:
        # Set API key
        os.environ['GROQ_API_KEY'] = api_key
        
        # Initialize and run the agent workflow
        try:
            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create columns for displaying process and results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔄 Processing")
                processing_area = st.empty()
                
                # Build graph and prepare initial state
                graph = build_text2sql_graph(model_name, temperature)
                
                # Initial state
                initial_state = {
                    'question': question,
                    'table_schemas': st.session_state.table_schemas,
                    'database': "",
                    'db_path': st.session_state.db_info,
                    'sql': "",
                    'accepted': False,
                    'reflect': [],
                    'revision': 0,
                    'max_revision': max_revisions,
                    'csv_files': st.session_state.csv_files
                }
                
                # Track execution
                thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
                steps = []
                
                total_steps = max_revisions * 2 + 2  # Approximate max steps
                step_count = 0
                
                # Process each step in the graph
                for s in graph.stream(initial_state, thread):
                    step_count += 1
                    step_name = list(s.keys())[0] if isinstance(s, dict) else "processing"
                    steps.append(f"Step {step_count}: {step_name}")
                    progress = min(step_count / total_steps, 0.9)
                    progress_bar.progress(progress)
                    status_text.text(f"Running agent: {step_name}")
                    processing_area.text('\n'.join(steps))
                
                # Get final state
                final_state = graph.get_state(thread)
                final_sql = final_state.values['sql']
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
            # Display SQL and results in second column
            with col2:
                st.subheader("🔍 Generated SQL")
                st.code(final_sql, language="sql")
                
                # Execute SQL
                st.subheader("📊 Query Results")
                try:
                    results = execute_sql(st.session_state.db_info, final_sql)
                    if isinstance(results, pd.DataFrame):
                        st.dataframe(results, use_container_width=True)
                        
                        # Add export options
                        st.download_button(
                            label="Download Results as CSV",
                            data=results.to_csv(index=False).encode('utf-8'),
                            file_name=f"query_results_{uuid.uuid4().hex[:8]}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Error executing SQL: {results}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    if not uploaded_files:
        st.info("Please upload at least one CSV file to continue")
    
# Add footer with instructions
st.markdown("---")
st.markdown("""
**How to use this app:**
1. Enter your Groq API key in the sidebar
2. Upload one or more CSV files containing your data
3. Ask a question in natural language about your data
4. The agent will convert your question to SQL and show the results

**Note:** Your data remains on your local machine and is not sent to external servers other than the LLM API.
""")