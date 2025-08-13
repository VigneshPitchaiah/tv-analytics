from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import snowflake.connector
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

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

def get_available_tables():
    """Get list of available tables in the database."""
    try:
        conn = get_snowflake_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # Get tables from the current schema
        query = """
        SELECT TABLE_NAME, TABLE_SCHEMA, TABLE_CATALOG
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        cursor.execute(query)
        tables = cursor.fetchall()
        
        table_list = []
        for table in tables:
            table_name = table[0]
            schema_name = table[1]
            database_name = table[2]
            full_table_name = f"{database_name}.{schema_name}.{table_name}"
            table_list.append({
                'name': table_name,
                'full_name': full_table_name,
                'schema': schema_name,
                'database': database_name
            })
        
        return table_list
        
    except Exception as e:
        print(f"Error getting available tables: {e}")
        return []
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def get_table_schema(table_name=None):
    """Get the schema of a specific table for context."""
    if not table_name:
        # Try to get the first available table
        available_tables = get_available_tables()
        if available_tables:
            table_name = available_tables[0]['full_name']
        else:
            return "No tables available in the current schema."
    
    # If we still don't have a table name, return a generic message
    if not table_name:
        return "Please select a table to analyze."
    
    try:
        conn = get_snowflake_connection()
        if not conn:
            return f"Unable to connect to database to get schema for {table_name}"
        
        cursor = conn.cursor()
        
        # Parse table name components
        table_parts = table_name.split('.')
        if len(table_parts) == 3:
            database_name = table_parts[0]
            schema_name = table_parts[1]
            table_name_only = table_parts[2]
        else:
            # If not fully qualified, try to get current database and schema
            cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
            current_db_schema = cursor.fetchone()
            database_name = current_db_schema[0]
            schema_name = current_db_schema[1]
            table_name_only = table_name
        
        # Get column information for the specified table
        query = """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COMMENT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = %s
        AND TABLE_SCHEMA = %s
        AND TABLE_CATALOG = %s
        ORDER BY ORDINAL_POSITION
        """
        
        cursor.execute(query, (table_name_only, schema_name, database_name))
        columns = cursor.fetchall()
        
        if not columns:
            return f"Table {table_name} not found or no columns available."
        
        # Build schema description
        schema_description = f"\nTable: {table_name}\nColumns:\n"
        
        for column in columns:
            col_name = column[0]
            data_type = column[1]
            is_nullable = column[2]
            default_val = column[3]
            comment = column[4]
            
            nullable_text = "" if is_nullable == 'YES' else " (NOT NULL)"
            comment_text = f" - {comment}" if comment else ""
            
            schema_description += f"- {col_name} ({data_type}){nullable_text}{comment_text}\n"
        
        # Get row count for context
        try:
            # Quote the table name properly to handle reserved keywords
            quoted_table_name = f'"{database_name}"."{schema_name}"."{table_name_only}"'
            cursor.execute(f"SELECT COUNT(*) FROM {quoted_table_name}")
            row_count = cursor.fetchone()[0]
            schema_description += "\nApproximate row count: " + str(f"{row_count:,}") + "\n"
        except Exception as e:
            schema_description += "\nRow count: Unable to determine (" + str(e) + ")\n"
        
        # Get sample data (first 3 rows)
        try:
            cursor.execute(f"SELECT * FROM {quoted_table_name} LIMIT 3")
            sample_rows = cursor.fetchall()
            if sample_rows:
                schema_description += "\nSample data (first 3 rows):\n"
                for i, row in enumerate(sample_rows, 1):
                    row_dict = dict(zip([col[0] for col in columns], row))
                    row_str = str(row_dict)
                    schema_description += "Row " + str(i) + ": " + row_str + "\n"
        except Exception as e:
            schema_description += "\nSample data: Unable to retrieve (" + str(e) + ")\n"
            
        return schema_description
        
    except Exception as e:
        print(f"Error getting table schema: {e}")
        return f"Error retrieving schema for table {table_name}: {str(e)}"
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/')
def index():
    """Serve the main page."""
    tables = get_available_tables()
    return render_template('index.html', models=CORTEX_MODELS, tables=tables)

@app.route('/api/tables')
def get_tables():
    """Get available tables in the database."""
    try:
        tables = get_available_tables()
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': f'Failed to get tables: {str(e)}'}), 500

@app.route('/api/table-schema', methods=['POST'])
def get_table_schema_api():
    """Get schema for a specific table."""
    try:
        data = request.json
        table_name = data.get('table_name')
        
        if not table_name:
            return jsonify({'error': 'Table name is required'}), 400
        
        schema = get_table_schema(table_name)
        return jsonify({'schema': schema, 'table_name': table_name})
        
    except Exception as e:
        return jsonify({'error': f'Failed to get table schema: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def query_cortex():
    """Handle user queries using Snowflake Cortex."""
    try:
        data = request.json
        user_question = data.get('question', '')
        selected_model = data.get('model', 'snowflake-arctic')
        selected_table = data.get('table', 'dev_dwdb.analytics.tv_show_ratings')  # Default table
        
        if not user_question:
            return jsonify({'error': 'No question provided'}), 400
            
        if selected_model not in CORTEX_MODELS:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Connect to Snowflake
        conn = get_snowflake_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to Snowflake'}), 500
        
        cursor = conn.cursor()
        
        # Create context for the AI model with selected table
        table_context = get_table_schema(selected_table)
        
        # Enhanced prompt for business analysis without code
        enhanced_prompt = f"""
        You are an expert business analyst providing insights about data.

        {table_context}

        User Question: {user_question}
        
        Please provide a comprehensive business analysis with:
        
        1. **Executive Summary**: A brief overview answering the user's question
        
        2. **Key Insights**: Provide specific business insights and findings
        
        3. **Important Metrics**: Highlight the most relevant KPIs and metrics for this analysis
        
        4. **Recommendations**: Provide actionable business recommendations
        
        5. **Visualization Suggestions**: Recommend appropriate chart types for presenting this data (without showing any code)
        
        Format your response using clear headers and bullet points for better readability.
        
        IMPORTANT: Do NOT include any SQL queries, code blocks, or technical implementation details in your response. Focus only on business insights, analysis, and strategic recommendations.
        """
        
        # Execute Cortex query
        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{selected_model}',
            '{enhanced_prompt.replace("'", "''")}'  
        ) AS response
        """
        
        cursor.execute(cortex_query)
        result = cursor.fetchone()
        
        if result and result[0]:
            ai_response = result[0]
            
            # Extract and execute SQL queries if mentioned in the response
            extracted_data = extract_and_execute_sql(ai_response, cursor, selected_table)
            
            # Generate KPIs and metrics from the data
            dashboard_data = generate_dashboard_data(extracted_data, ai_response, selected_table, cursor)
            
            return jsonify({
                'response': ai_response,
                'model_used': selected_model,
                'table_used': selected_table,
                'timestamp': datetime.now().isoformat(),
                'dashboard_data': dashboard_data,
                'visualization_data': extracted_data
            })
        else:
            return jsonify({'error': 'No response from Cortex'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Query processing error: {str(e)}'}), 500
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def extract_and_execute_sql(ai_response, cursor, table_name):
    """Extract SQL queries from AI response and execute them for visualization data."""
    import re
    
    extracted_data = {
        'queries_executed': [],
        'data_for_charts': [],
        'summary_stats': {}
    }
    
    try:
        # Get basic table info first
        cursor.execute(f'SELECT COUNT(*) as total_records FROM {table_name}')
        total_count = cursor.fetchone()[0]
        
        extracted_data['queries_executed'].append({
            'name': 'record_count',
            'description': 'Total Records',
            'columns': ['total_records'],
            'data': [(total_count,)],
            'chart_type': 'metric'
        })
        
        # Get sample data to understand the table structure
        cursor.execute(f'SELECT * FROM {table_name} LIMIT 5')
        sample_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        if sample_data and len(sample_data) > 0:
            # Try to find the most relevant columns for visualization
            # Look for numeric columns for aggregation
            numeric_cols = []
            text_cols = []
            
            for i, col_name in enumerate(column_names):
                sample_values = [row[i] for row in sample_data if row[i] is not None]
                if sample_values:
                    # Check if values are numeric
                    try:
                        numeric_values = [float(v) for v in sample_values if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())]
                        if len(numeric_values) > 0:
                            numeric_cols.append((col_name, i))
                    except:
                        pass
                    
                    # Check if values are categorical text
                    if not any(col_name == nc[0] for nc in numeric_cols):
                        text_values = [str(v) for v in sample_values if v is not None]
                        if len(text_values) > 0 and len(set(text_values)) < len(text_values):  # Has some repeated values
                            text_cols.append((col_name, i))
            
            # Generate meaningful charts based on available data
            if text_cols and numeric_cols:
                # Category analysis - group by text column, aggregate numeric
                text_col = text_cols[0][0]
                numeric_col = numeric_cols[0][0]
                
                try:
                    query = f'SELECT {text_col}, AVG({numeric_col}) as avg_value FROM {table_name} WHERE {text_col} IS NOT NULL AND {numeric_col} IS NOT NULL GROUP BY {text_col} ORDER BY avg_value DESC LIMIT 10'
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    if results:
                        extracted_data['queries_executed'].append({
                            'name': 'category_analysis',
                            'description': f'Average {numeric_col} by {text_col}',
                            'columns': [text_col, 'avg_value'],
                            'data': results,
                            'chart_type': 'bar'
                        })
                except Exception as e:
                    print(f"Error executing category analysis: {e}")
            
            if len(text_cols) >= 1:
                # Distribution chart
                text_col = text_cols[0][0]
                try:
                    query = f'SELECT {text_col}, COUNT(*) as count FROM {table_name} WHERE {text_col} IS NOT NULL GROUP BY {text_col} ORDER BY count DESC LIMIT 8'
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    if results:
                        extracted_data['queries_executed'].append({
                            'name': 'distribution_analysis', 
                            'description': f'Distribution by {text_col}',
                            'columns': [text_col, 'count'],
                            'data': results,
                            'chart_type': 'pie'
                        })
                except Exception as e:
                    print(f"Error executing distribution analysis: {e}")
                    
    except Exception as e:
        print(f"Error in extract_and_execute_sql: {e}")
    
    return extracted_data

def generate_basic_analytical_queries(table_name):
    """Generate dynamic analytical queries based on actual table schema."""
    queries = []
    
    try:
        # Get actual table schema dynamically
        conn = get_snowflake_connection()
        if not conn:
            return queries
            
        cursor = conn.cursor()
        
        # Parse table name components
        table_parts = table_name.split('.')
        if len(table_parts) == 3:
            database_name = table_parts[0]
            schema_name = table_parts[1]
            table_name_only = table_parts[2]
        else:
            cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
            current_db_schema = cursor.fetchone()
            database_name = current_db_schema[0]
            schema_name = current_db_schema[1]
            table_name_only = table_name
        
        # Get column information
        schema_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_CATALOG = '{database_name}'
        AND TABLE_SCHEMA = '{schema_name}'
        AND TABLE_NAME = '{table_name_only}'
        ORDER BY ORDINAL_POSITION
        """
        
        cursor.execute(schema_query)
        columns = cursor.fetchall()
        
        if not columns:
            return queries
        
        # Categorize columns by type
        numeric_columns = []
        text_columns = []
        date_columns = []
        
        for col_name, data_type, is_nullable in columns:
            data_type_upper = data_type.upper()
            if any(t in data_type_upper for t in ['NUMBER', 'DECIMAL', 'FLOAT', 'INTEGER', 'BIGINT']):
                numeric_columns.append(col_name)
            elif any(t in data_type_upper for t in ['DATE', 'TIME', 'TIMESTAMP']):
                date_columns.append(col_name)
            elif any(t in data_type_upper for t in ['VARCHAR', 'STRING', 'TEXT', 'CHAR']):
                text_columns.append(col_name)
        
        # Always add basic queries
        queries.extend([
            {
                'name': 'record_count',
                'description': 'Total number of records',
                'query': f'SELECT COUNT(*) as total_records FROM {table_name}',
                'chart_type': 'metric'
            },
            {
                'name': 'sample_data',
                'description': 'Sample of recent data',
                'query': f'SELECT * FROM {table_name} LIMIT 10',
                'chart_type': 'table'
            }
        ])
        
        # Generate dynamic queries based on column types
        if text_columns and numeric_columns:
            # Top values by first numeric column grouped by first text column
            first_text = text_columns[0]
            first_numeric = numeric_columns[0]
            queries.append({
                'name': 'top_categories',
                'description': f'Top {first_text} by {first_numeric}',
                'query': f'SELECT {first_text}, AVG({first_numeric}) as avg_{first_numeric.lower()} FROM {table_name} WHERE {first_text} IS NOT NULL GROUP BY {first_text} ORDER BY avg_{first_numeric.lower()} DESC LIMIT 10',
                'chart_type': 'bar'
            })
        
        if len(text_columns) >= 2:
            # Distribution by second text column if available
            second_text = text_columns[1] if len(text_columns) > 1 else text_columns[0]
            queries.append({
                'name': 'category_distribution',
                'description': f'Distribution by {second_text}',
                'query': f'SELECT {second_text}, COUNT(*) as count FROM {table_name} WHERE {second_text} IS NOT NULL GROUP BY {second_text} ORDER BY count DESC LIMIT 8',
                'chart_type': 'pie'
            })
        
        if date_columns and numeric_columns:
            # Time-based trends
            first_date = date_columns[0]
            first_numeric = numeric_columns[0]
            queries.append({
                'name': 'time_trends',
                'description': f'{first_numeric} trends over time',
                'query': f'SELECT {first_date}, AVG({first_numeric}) as avg_{first_numeric.lower()} FROM {table_name} WHERE {first_date} IS NOT NULL GROUP BY {first_date} ORDER BY {first_date} LIMIT 20',
                'chart_type': 'line'
            })
        
        if len(numeric_columns) >= 2:
            # Correlation between numeric columns
            first_num = numeric_columns[0]
            second_num = numeric_columns[1]
            queries.append({
                'name': 'numeric_comparison',
                'description': f'Comparison of {first_num} vs {second_num}',
                'query': f'SELECT AVG({first_num}) as avg_{first_num.lower()}, AVG({second_num}) as avg_{second_num.lower()}, COUNT(*) as record_count FROM {table_name} WHERE {first_num} IS NOT NULL AND {second_num} IS NOT NULL',
                'chart_type': 'metric'
            })
        
        # Generate summary statistics for numeric columns
        if numeric_columns:
            for num_col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                queries.append({
                    'name': f'{num_col.lower()}_stats',
                    'description': f'Statistics for {num_col}',
                    'query': f'SELECT MIN({num_col}) as min_val, MAX({num_col}) as max_val, AVG({num_col}) as avg_val, COUNT({num_col}) as non_null_count FROM {table_name} WHERE {num_col} IS NOT NULL',
                    'chart_type': 'metric'
                })
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error generating dynamic queries: {e}")
        # Fallback to basic queries only
        queries = [
            {
                'name': 'record_count',
                'description': 'Total number of records',
                'query': f'SELECT COUNT(*) as total_records FROM {table_name}',
                'chart_type': 'metric'
            }
        ]
    
    return queries

def generate_ai_dashboard_config(table_name, cursor):
    """Generate AI-driven dashboard configuration based on table structure."""
    try:
        # Get table schema information
        schema_info = get_table_schema(table_name)
        
        # Use AI to determine best KPIs and chart types for this table
        ai_prompt = f"""
        Based on this table structure:
        {schema_info}
        
        Generate a JSON configuration for dashboard KPIs and chart types that would be most relevant for this data.
        Return only valid JSON in this format:
        {{
            "kpis": [
                {{"type": "count", "title": "Total Records", "icon": "storage", "color": "blue"}},
                {{"type": "average", "column": "column_name", "title": "Average Value", "icon": "analytics", "color": "green"}}
            ],
            "charts": [
                {{"type": "bar", "title": "Distribution Chart", "group_by": "column_name", "aggregate": "count"}},
                {{"type": "line", "title": "Trend Chart", "x_axis": "date_column", "y_axis": "numeric_column"}}
            ]
        }}
        """
        
        # Execute AI query to get dashboard config
        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'snowflake-arctic',
            '{ai_prompt.replace("'", "''")}'
        ) AS config
        """
        
        cursor.execute(cortex_query)
        result = cursor.fetchone()
        
        if result and result[0]:
            import json
            try:
                # Extract JSON from AI response
                ai_response = result[0]
                # Find JSON block in response
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = ai_response[json_start:json_end]
                    return json.loads(json_str)
            except:
                pass
        
        # Fallback to default configuration
        return {
            "kpis": [
                {"type": "count", "title": "Total Records", "icon": "storage", "color": "blue"},
                {"type": "columns", "title": "Data Fields", "icon": "view_column", "color": "green"}
            ],
            "charts": [
                {"type": "bar", "title": "Data Distribution", "group_by": "first_string_column", "aggregate": "count"},
                {"type": "pie", "title": "Category Breakdown", "group_by": "first_categorical_column", "aggregate": "count"}
            ]
        }
        
    except Exception as e:
        print(f"Error generating AI dashboard config: {e}")
        return {"kpis": [], "charts": []}

def generate_dynamic_dashboard_from_data(extracted_data, dashboard_config, table_name, cursor):
    """Generate dashboard using AI configuration and actual data."""
    dashboard = {
        'kpis': [],
        'charts': [],
        'metrics': {},
        'insights': [],
        'forecast': None
    }
    
    try:
        # Generate KPIs based on AI configuration
        for kpi_config in dashboard_config.get('kpis', []):
            if kpi_config['type'] == 'count':
                # Get total record count
                for query_result in extracted_data.get('queries_executed', []):
                    if 'count' in query_result.get('name', '').lower():
                        dashboard['kpis'].append({
                            'title': kpi_config['title'],
                            'value': query_result['data'][0][0] if query_result.get('data') else 0,
                            'icon': kpi_config['icon'],
                            'color': kpi_config['color'],
                            'format': 'number'
                        })
                        break
            
            elif kpi_config['type'] == 'average' and 'column' in kpi_config:
                # Calculate average for specified column
                for query_result in extracted_data.get('queries_executed', []):
                    if query_result.get('data'):
                        # Try to find numeric data to average
                        numeric_values = []
                        for row in query_result['data']:
                            for val in row:
                                if isinstance(val, (int, float)):
                                    numeric_values.append(val)
                        
                        if numeric_values:
                            avg_value = sum(numeric_values) / len(numeric_values)
                            dashboard['kpis'].append({
                                'title': kpi_config['title'],
                                'value': round(avg_value, 2),
                                'icon': kpi_config['icon'],
                                'color': kpi_config['color'],
                                'format': 'decimal'
                            })
                            break
        
        # Generate charts based on AI configuration and available data
        chart_id = 1
        for chart_config in dashboard_config.get('charts', []):
            for query_result in extracted_data.get('queries_executed', []):
                if query_result.get('data'):
                    data = query_result['data']
                    
                    # Generate chart based on configuration
                    chart_data = {
                        'id': f'ai_generated_chart_{chart_id}',
                        'title': chart_config['title'],
                        'type': chart_config['type'],
                        'data': {
                            'labels': [str(row[0]) for row in data[:10]],  # First column as labels
                            'values': []
                        },
                        'config': get_chart_styling(chart_config['type'])
                    }
                    
                    # Extract numeric values for the chart
                    for row in data[:10]:
                        for val in row[1:]:  # Skip first column (labels)
                            if isinstance(val, (int, float)):
                                chart_data['data']['values'].append(float(val))
                                break
                        if len(chart_data['data']['values']) == 0:
                            chart_data['data']['values'].append(0)
                    
                    # Only add chart if we have valid data
                    if chart_data['data']['values'] and any(v != 0 for v in chart_data['data']['values']):
                        dashboard['charts'].append(chart_data)
                        chart_id += 1
                        break
        
        # Generate insights using AI
        dashboard['insights'] = generate_insights_from_data(extracted_data)
        
    except Exception as e:
        print(f"Error generating dynamic dashboard: {e}")
    
    return dashboard

def get_chart_styling(chart_type):
    """Get appropriate styling for different chart types."""
    color_schemes = {
        'bar': {
            'backgroundColor': 'rgba(26, 115, 232, 0.8)',
            'borderColor': 'rgba(26, 115, 232, 1)'
        },
        'line': {
            'borderColor': 'rgba(52, 168, 83, 1)',
            'backgroundColor': 'rgba(52, 168, 83, 0.1)',
            'fill': True
        },
        'pie': {
            'backgroundColor': [
                'rgba(26, 115, 232, 0.8)',
                'rgba(52, 168, 83, 0.8)', 
                'rgba(251, 188, 4, 0.8)',
                'rgba(234, 67, 53, 0.8)',
                'rgba(255, 152, 0, 0.8)'
            ]
        }
    }
    
    return color_schemes.get(chart_type, color_schemes['bar'])

def generate_insights_from_data(extracted_data):
    """Generate actionable insights from the analyzed data."""
    insights = []
    
    try:
        queries_executed = extracted_data.get('queries_executed', [])
        
        for query_result in queries_executed:
            if query_result.get('data'):
                data = query_result['data']
                query_name = query_result.get('name', '')
                
                if 'top' in query_name.lower() and len(data) > 0:
                    top_item = data[0]
                    insights.append(f"The top performer is '{top_item[0]}' with a value of {top_item[1]}.")
                
                elif 'trend' in query_name.lower() and len(data) > 1:
                    first_value = data[0][1]
                    last_value = data[-1][1]
                    trend = "increasing" if last_value > first_value else "decreasing"
                    change = abs(last_value - first_value)
                    insights.append(f"The data shows a {trend} trend with a change of {change:.2f}.")
                
                elif 'count' in query_name.lower():
                    total = data[0][0] if data else 0
                    insights.append(f"Total records analyzed: {total:,}.")
    
    except Exception as e:
        print(f"Error generating insights: {e}")
        insights = ["Unable to generate insights from the current data."]
    
    return insights

def generate_dashboard_data(extracted_data, ai_response, table_name, cursor):
    """Generate comprehensive dashboard data including KPIs, metrics, and chart configurations."""
    dashboard = {
        'kpis': [],
        'charts': [],
        'metrics': {},
        'insights': [],
        'forecast': None
    }
    
    # Use AI to generate dynamic dashboard configuration
    dashboard_config = generate_ai_dashboard_config(table_name, cursor)
    
    try:
        # Generate dynamic KPIs and charts using AI-driven analysis
        dashboard = generate_dynamic_dashboard_from_data(extracted_data, dashboard_config, table_name, cursor)
        
        # Generate forecast if time-series data is available
        for query_result in extracted_data.get('queries_executed', []):
            if 'trend' in query_result.get('name', '').lower() and query_result.get('data'):
                if len(query_result['data']) > 2:
                    dashboard['forecast'] = generate_simple_forecast(query_result['data'])
                    break
        
        # Add additional KPIs if available
        if dashboard['charts']:
            dashboard['kpis'].append({
                'title': 'Chart Types',
                'value': len(dashboard['charts']),
                'icon': 'insert_chart',
                'color': 'green',
                'format': 'number'
            })
        
        # Generate insights based on the data
        dashboard['insights'] = generate_insights_from_data(extracted_data)
        
        # Calculate summary metrics
        dashboard['metrics'] = {
            'data_freshness': 'Real-time',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'table_analyzed': table_name.split('.')[-1].replace('_', ' ').title()
        }
        
    except Exception as e:
        print(f"Error generating dashboard data: {e}")
        dashboard['kpis'] = [{
            'title': 'Error',
            'value': 'Data unavailable',
            'icon': 'error',
            'color': 'red',
            'format': 'text'
        }]
    
    return dashboard

def generate_simple_forecast(time_series_data):
    """Generate a simple forecast based on linear trend."""
    try:
        if len(time_series_data) < 3:
            return None
            
        # Simple linear regression for trend
        values = [float(row[1]) for row in time_series_data]
        n = len(values)
        
        # Calculate trend
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i**2 for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum**2)
        intercept = (y_sum - slope * x_sum) / n
        
        # Generate forecast for next 3 periods
        forecast_values = []
        for i in range(n, n + 3):
            forecast_values.append(round(slope * i + intercept, 2))
        
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        return {
            'next_values': forecast_values,
            'trend': trend_direction,
            'confidence': 'Medium',
            'slope': round(slope, 4)
        }
    except:
        return None

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
    table_name = request.args.get('table', 'dev_dwdb.analytics.tv_show_ratings')
    
    # Get table schema to generate relevant questions
    try:
        schema_info = get_table_schema(table_name)
        questions = generate_sample_questions_for_table(table_name, schema_info)
    except:
        # Fallback to default questions
        questions = [
            "What are the top 5 records by the first numeric column?",
            "Show me the distribution of data by the main categorical column",
            "What are the trends over time if there's a date column?",
            "Give me a summary of the key metrics in this table",
            "What insights can you provide from this dataset?"
        ]
    
    return jsonify({'questions': questions})

def generate_sample_questions_for_table(table_name, schema_info):
    """Generate relevant sample questions based on table schema."""
    questions = []
    
    # Extract table name for context
    table_short_name = table_name.split('.')[-1].replace('_', ' ').title()
    
    # Default questions that work for most tables
    questions.extend([
        f"What are the key insights from the {table_short_name} data?",
        f"Show me a summary of the {table_short_name} table",
        f"What are the top 10 records in {table_short_name}?"
    ])
    
    # Analyze schema to generate specific questions
    schema_lower = schema_info.lower()
    
    # Check for common column patterns and generate relevant questions
    if 'date' in schema_lower or 'time' in schema_lower:
        questions.extend([
            f"What are the trends over time in {table_short_name}?",
            f"Show me the monthly/yearly patterns in the data"
        ])
    
    if 'revenue' in schema_lower or 'sales' in schema_lower or 'amount' in schema_lower:
        questions.extend([
            f"What are the revenue trends?",
            f"Which periods had the highest sales?"
        ])
    
    if 'customer' in schema_lower or 'user' in schema_lower:
        questions.extend([
            f"What are the customer behavior patterns?",
            f"Which customers are the most valuable?"
        ])
    
    if 'product' in schema_lower or 'item' in schema_lower:
        questions.extend([
            f"Which products perform best?",
            f"What is the product performance analysis?"
        ])
    
    if 'region' in schema_lower or 'location' in schema_lower or 'country' in schema_lower:
        questions.extend([
            f"How does performance vary by location?",
            f"Which regions show the best results?"
        ])
    
    if 'rating' in schema_lower or 'score' in schema_lower:
        questions.extend([
            f"What are the highest rated items?",
            f"How do ratings correlate with other metrics?"
        ])
    
    # Limit to 8 questions to avoid overwhelming the user
    return questions[:8]



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
