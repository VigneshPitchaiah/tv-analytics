from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import snowflake.connector
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
CORS(app)

# Snowflake configuration from environment variables
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

# Available Cortex models
CORTEX_MODELS = [
    'snowflake-arctic',
    'llama3-8b',
    'claude-4-sonnet',
    'mistral-large',
    'deepseek-r1',
    'openai-gpt-4.1'
]

def get_snowflake_connection():
    """Create and return a Snowflake connection."""
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

def get_table_schema():
    """Get the schema of the TV ratings table for context."""
    return """
    Table: dev_dwdb.analytics.tv_show_ratings
    Columns:
    - Week (DATE): The week of the TV ratings
    - Show_Title (STRING): Name of the TV show
    - Rating (FLOAT): TV rating score
    - Viewers_Millions (FLOAT): Number of viewers in millions
    - Share_Percent (FLOAT): Market share percentage
    - Network (STRING): Broadcasting network (NBC, CBS, ABC)

    Sample data covers TV show ratings from January 2024 to February 2024.
    """

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', models=CORTEX_MODELS)

@app.route('/api/query', methods=['POST'])
def query_cortex():
    """Handle user queries using Snowflake Cortex."""
    try:
        data = request.json
        user_question = data.get('question', '')
        selected_model = data.get('model', 'snowflake-arctic')

        if not user_question:
            return jsonify({'error': 'No question provided'}), 400

        if selected_model not in CORTEX_MODELS:
            return jsonify({'error': 'Invalid model selected'}), 400

        conn = get_snowflake_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to Snowflake'}), 500

        cursor = conn.cursor()
        table_context = get_table_schema()

        enhanced_prompt = f"""
        You are a data analyst with access to a TV show ratings database.

        {table_context}

        User Question: {user_question}

        Please analyze this question and provide a well-structured response with:

        1. **Executive Summary**: A brief 1-2 sentence answer to the question
        2. **Data Analysis**: If the question requires data from the database, write a SQL query to get the relevant data and explain what the data shows
        3. **Key Insights**: Provide 3-5 specific insights based on the data with bullet points
        4. **Recommendations**: If applicable, provide actionable recommendations

        Format your response using markdown with clear headers (##) and bullet points (-) for better readability.

        If you need to write SQL, use the table name: dev_dwdb.analytics.tv_show_ratings and wrap it in ```sql code blocks.

        Be specific with numbers, percentages, and comparisons. Focus on actionable insights.
        """

        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{selected_model}',
            '{enhanced_prompt.replace("'", "''")}'
        ) as response;
        """

        cursor.execute(cortex_query)
        result = cursor.fetchone()

        if result:
            ai_response = result[0]
            data_results = None
            column_names = []

            if 'SELECT' in ai_response.upper():
                try:
                    sql_match = re.search(r'```sql\n(.*?)\n```', ai_response, re.DOTALL | re.IGNORECASE)
                    if not sql_match:
                        sql_match = re.search(r'SELECT.*?;', ai_response, re.DOTALL | re.IGNORECASE)

                    if sql_match:
                        sql_query = sql_match.group(1) if sql_match.group(1) else sql_match.group(0)
                        cursor.execute(sql_query)
                        data_results = cursor.fetchall()
                        column_names = [desc[0] for desc in cursor.description]
                except Exception as sql_error:
                    print(f"SQL execution error: {sql_error}")

            response_data = {
                'response': ai_response,
                'model_used': selected_model,
                'timestamp': datetime.now().isoformat()
            }

            if data_results:
                response_data['data'] = {
                    'columns': column_names,
                    'rows': data_results
                }

            return jsonify(response_data)
        else:
            return jsonify({'error': 'No response from Cortex model'}), 500

    except Exception as e:
        print(f"Error in query_cortex: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()




@app.route('/api/direct-sql', methods=['POST'])
def execute_direct_sql():
    """Execute direct SQL queries for advanced users."""
    try:
        data = request.json
        sql_query = data.get('query', '')

        if not sql_query:
            return jsonify({'error': 'No SQL query provided'}), 400

        # Basic SQL injection protection (enhance for production)
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql_query.upper():
                return jsonify({'error': f'Query contains dangerous keyword: {keyword}'}), 400

        conn = get_snowflake_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to Snowflake'}), 500

        cursor = conn.cursor()
        cursor.execute(sql_query)

        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        return jsonify({
            'columns': column_names,
            'rows': results,
            'row_count': len(results)
        })

    except Exception as e:
        return jsonify({'error': f'SQL execution error: {str(e)}'}), 500

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/api/sample-questions')
def get_sample_questions():
    """Provide sample questions users can ask."""
    questions = [
        "What are the top 5 TV shows by average rating?",
        "How did Sunday Night Football's viewership trend over time?",
        "Which network has the highest average share percentage?",
        "Compare the performance of NBC shows vs ABC shows",
        "What was the best performing week for each show?",
        "Show me the correlation between ratings and viewership",
        "Which show had the most consistent ratings?",
        "What are the viewership trends by network over time?"
    ]
    return jsonify({'questions': questions})

# This is required for Vercel
def application(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
