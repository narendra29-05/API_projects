# Text to SQL Agent

## Overview
Text to SQL Agent is a powerful tool that converts natural language questions into SQL queries. It allows users to upload CSV files, ask questions about their data in plain English, and receive accurate SQL queries and results—all without writing a single line of code.

## Features
- **Natural Language to SQL Conversion**: Ask questions in plain English and get SQL queries automatically.
- **CSV File Support**: Upload and analyze multiple CSV files simultaneously.
- **Interactive Data Preview**: Explore your data with statistics and visualizations before querying.
- **Multi-Agent Processing**: Utilizes a collaborative agent system for accurate query generation.
- **Error Handling**: Intelligent SQL validation and improvement.

## Architecture
The application is built using a LangGraph workflow with multiple specialized agents:

1. **Schema Finder Agent**: Identifies relevant tables and columns needed to answer the question.
2. **SQL Writer Agent**: Crafts precise SQL queries based on the identified schema.
3. **SQL Validator Agent**: Checks query correctness and validates SQLite compatibility.
4. **SQL Improver Agent**: Provides feedback for query improvement when needed.

## Getting Started

### Prerequisites
- Python 3.7+
- Groq API key
- Required Python packages (see `requirements.txt`)

### Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/text-to-sql-agent.git
   cd text-to-sql-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

### Usage
1. Enter your Groq API key in the sidebar.
2. Upload one or more CSV files containing your data.
3. Browse and preview your data in the interactive data explorer.
4. Ask a question in natural language about your data.
5. Review the generated SQL query and results.
6. Download the results as CSV if needed.

## Configuration Options
- **Model Selection**: Choose from different Groq models (Llama, Mixtral).
- **Maximum SQL Revisions**: Set how many times the agent should attempt to improve the SQL query.
- **Temperature**: Adjust the randomness of the LLM responses.

## File Structure
- `app.py`: Main application file with Streamlit UI
- `components/`
  - `agent_workflow.py`: LangGraph agent definitions and workflow
  - `db_utils.py`: Database utilities for handling CSV files and SQLite
  - `ui_components.py`: UI components for the Streamlit interface

## Privacy and Security
Your data remains on your local machine and is not sent to external servers other than the LLM API. The application creates a temporary SQLite database to store and query your data.

## Troubleshooting
- **No results**: Check if your table and column names match the generated SQL query.
- **SQL errors**: Review the error message and adjust your question to be more specific.
- **Slow performance**: Try using a smaller dataset or limit the number of rows in your query.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
