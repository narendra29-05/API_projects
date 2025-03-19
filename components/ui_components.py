import streamlit as st
import pandas as pd

def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input("Enter your Groq API Key", type="password")
        model_name = st.selectbox(
            "Select Groq Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
        )
        
        st.header("⚙️ Agent Settings")
        max_revisions = st.slider("Maximum SQL Revisions", 1, 5, 2)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
        
        st.header("📊 Sample Questions")
        st.markdown("""
        After uploading CSV files, try questions like:
        - What is the average value in [column]?
        - How many records are in [table]?
        - What are the top 5 [items] by [metric]?
        """)
        
    return api_key, model_name, max_revisions, temperature

def render_data_preview(dataframes):
    """Render data preview with statistics"""
    # Enhanced Data Preview Section
    tables = list(dataframes.keys())
    if tables:
        selected_table = st.selectbox("Select a table to preview", tables)
        
        if selected_table:
            df = dataframes[selected_table]
            
            # Display table statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])  # Fixed from df.shape[0]
            with col3:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                st.metric("Numeric Columns", len(numeric_cols))
            with col4:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Allow user to select how many rows to display
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
            
            # Show data with profiling info
            with st.expander(f"Preview of {selected_table}", expanded=True):
                st.dataframe(df.head(num_rows), use_container_width=True)
                
                # Column information
                st.subheader("Column Information")
                column_info = []
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    unique_vals = df[col].nunique()
                    missing = df[col].isna().sum()
                    
                    column_info.append({
                        "Column": col,
                        "Type": col_type,
                        "Unique Values": unique_vals,
                        "Missing Values": missing,
                        "% Missing": f"{(missing/len(df)*100):.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(column_info), use_container_width=True)

def render_schema_view(table_schemas):
    """Render database schema view"""
    with st.expander("Database Schema", expanded=True):
        st.code(table_schemas)