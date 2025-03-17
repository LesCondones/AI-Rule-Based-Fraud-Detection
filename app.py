from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
import os
import io
import base64
import uuid
import re
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from werkzeug.utils import secure_filename

# Import the fraud detection classes from AIRuleBasedFraudDetection.py
from AIRuleBasedFraudDetection import AIDocumentProcessor, EnhancedFraudDetectionSystem, FraudDetectionApp

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['RESULT_FOLDER'] = 'results'

# Create upload and result directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'pdf', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate a session ID for this analysis
        session_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        # Store the file path and original name in session
        session['file_path'] = file_path
        session['original_filename'] = filename
        session['session_id'] = session_id
        
        # Get large amount threshold from form if provided
        threshold = request.form.get('largeAmountThreshold', 2000)
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = 2000
        
        session['threshold'] = threshold
        
        # Redirect to document processing route
        return redirect(url_for('process_document'))
    
    flash('File type not allowed. Please upload a CSV, Excel, PDF, or image file.')
    return redirect(url_for('index'))

@app.route('/process')
def process_document():
    """Process the uploaded document with AI."""
    file_path = session.get('file_path')
    session_id = session.get('session_id')
    threshold = session.get('threshold', 2000)
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.')
        return redirect(url_for('index'))
    
    try:
        # Initialize the fraud detection app
        app_instance = FraudDetectionApp()
        app_instance.user_profile['large_amount_threshold'] = threshold
        
        # Process document
        document_data = app_instance.process_document(file_path)
        
        # Generate insights
        insights = app_instance.document_processor.generate_insights(
            document_data['data'], 
            document_data.get('category_mapping', {})
        )
        
        # Automatically map fields
        app_instance.map_fields()
        
        # Store field mapping in session for display
        field_mapping = app_instance.field_mapping
        
        # Process and analyze transactions
        transactions = app_instance.process_transactions()
        results = app_instance.analyze_transactions()
        
        # Generate charts
        charts = generate_visualizations(app_instance, session_id)
        
        # Generate AI explanation
        explanation = app_instance.generate_ai_explanation()
        
        # Export results to CSV
        output_file = os.path.join(app.config['RESULT_FOLDER'], f"{session_id}_fraud_analysis.csv")
        app_instance.export_analysis_to_csv(output_file)
        
        # Store data in session for templates
        session['document_data'] = {
            'format': document_data.get('format', 'unknown'),
            'headers': document_data.get('headers', []),
            'record_count': len(document_data.get('data', [])),
            'category_mapping': {k: v for k, v in document_data.get('category_mapping', {}).items()},
            'field_mapping': field_mapping
        }
        
        # Prepare summary stats
        total = len(results)
        high_risk = sum(1 for r in results if r['fraud_likelihood'] == 'High')
        medium_risk = sum(1 for r in results if r['fraud_likelihood'] == 'Medium')
        low_risk = sum(1 for r in results if r['fraud_likelihood'] == 'Low')
        
        # Calculate percentages
        high_risk_pct = (high_risk / total) * 100 if total > 0 else 0
        medium_risk_pct = (medium_risk / total) * 100 if total > 0 else 0
        low_risk_pct = (low_risk / total) * 100 if total > 0 else 0
        
        # Calculate amount statistics
        total_amount = sum(float(tx.get('amount', 0)) for tx in transactions)
        high_risk_amount = sum(
            float(tx.get('amount', 0)) 
            for tx, result in zip(transactions, results) 
            if result['fraud_likelihood'] == 'High'
        )
        high_risk_amount_pct = (high_risk_amount / total_amount) * 100 if total_amount > 0 else 0
        
        # Count AI-enhanced detections
        ai_enhanced = sum(1 for r in results if r.get('ai_enhanced', False))
        
        # Get most triggered rules
        rule_counts = {}
        for result in results:
            for rule in result['triggered_rules']:
                rule_name = rule['rule_name']
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        # Sort and get top rules
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Store summary in session
        session['summary'] = {
            'total': total,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'high_risk_pct': high_risk_pct,
            'medium_risk_pct': medium_risk_pct,
            'low_risk_pct': low_risk_pct,
            'total_amount': total_amount,
            'high_risk_amount': high_risk_amount,
            'high_risk_amount_pct': high_risk_amount_pct,
            'ai_enhanced': ai_enhanced,
            'top_rules': top_rules
        }
        
        # Format and store transactions for display
        formatted_transactions = []
        for tx, result in zip(transactions, results):
            formatted_tx = {
                'date': format_timestamp(tx.get('timestamp')),
                'description': tx.get('description') or tx.get('merchantCategory', 'Unknown'),
                'amount': float(tx.get('amount', 0)),
                'risk_level': result['fraud_likelihood'],
                'risk_score': result['risk_score'],
                'rules': [rule['rule_name'] for rule in result['triggered_rules']],
                'rules_text': ', '.join(rule['rule_name'] for rule in result['triggered_rules']) if result['triggered_rules'] else 'None',
                'ai_enhanced': result.get('ai_enhanced', False),
                'original_data': tx.get('originalData', {})
            }
            formatted_transactions.append(formatted_tx)
        
        # Store transactions in session
        session['transactions'] = formatted_transactions
        
        # Store insights and explanation
        session['insights'] = insights
        session['explanation'] = explanation
        session['charts'] = charts
        session['output_file'] = output_file
        
        return redirect(url_for('show_results'))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error processing document: {str(e)}")
        return redirect(url_for('index'))

def format_timestamp(timestamp):
    """Format timestamp to readable date format."""
    if not timestamp:
        return "Unknown"
    
    try:
        if isinstance(timestamp, str):
            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y",
                "%Y-%m-%d"
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.strftime("%m/%d/%Y %I:%M %p")
                except ValueError:
                    continue
            
            # If all formats fail, return as is
            return timestamp
            
        elif isinstance(timestamp, datetime):
            return timestamp.strftime("%m/%d/%Y %I:%M %p")
            
        return str(timestamp)
    except:
        return str(timestamp)

def generate_visualizations(app_instance, session_id):
    """Generate visualizations and return paths to saved images."""
    charts = {}
    
    try:
        # Get results and transactions
        results = app_instance.results
        
        # 1. Risk distribution pie chart
        plt.figure(figsize=(8, 6))
        risk_counts = {
            'High': sum(1 for r in results if r['fraud_likelihood'] == 'High'),
            'Medium': sum(1 for r in results if r['fraud_likelihood'] == 'Medium'),
            'Low': sum(1 for r in results if r['fraud_likelihood'] == 'Low')
        }
        
        colors = ['#ff6b6b', '#feca57', '#1dd1a1']
        plt.pie(
            risk_counts.values(),
            labels=risk_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        plt.title('Transaction Risk Distribution', fontweight='bold')
        
        # Save chart for download
        pie_chart_path = os.path.join(app.config['RESULT_FOLDER'], f"{session_id}_risk_distribution.png")
        plt.savefig(pie_chart_path, dpi=100, bbox_inches='tight')
        
        # Save as base64 for web display
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        pie_chart_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()
        
        # 2. Top triggered rules bar chart
        rule_counts = {}
        for result in results:
            for rule in result['triggered_rules']:
                rule_name = rule['rule_name']
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        # Sort and get top rules
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_rules:  # Only create chart if we have rules
            plt.figure(figsize=(10, 6))
            rule_names = [r[0] for r in top_rules]
            rule_values = [r[1] for r in top_rules]
            
            bars = plt.barh(rule_names, rule_values, color='#3498db')
            
            # Add count labels to bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(
                    width + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.0f}',
                    va='center',
                    fontweight='bold'
                )
            
            plt.title('Top Triggered Rules', fontweight='bold')
            plt.xlabel('Number of Occurrences')
            
            # Save chart for download
            bar_chart_path = os.path.join(app.config['RESULT_FOLDER'], f"{session_id}_top_rules.png")
            plt.savefig(bar_chart_path, dpi=100, bbox_inches='tight')
            
            # Save as base64 for web display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            bar_chart_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()
            
            charts['bar_chart'] = {
                'path': bar_chart_path,
                'b64': bar_chart_b64
            }
        
        charts['pie_chart'] = {
            'path': pie_chart_path,
            'b64': pie_chart_b64
        }
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    return charts

@app.route('/results')
def show_results():
    """Show fraud detection results."""
    # Check if we have results in session
    if not session.get('transactions'):
        flash('No analysis results available. Please upload a document first.')
        return redirect(url_for('index'))
    
    return render_template(
        'results.html',
        filename=session.get('original_filename', 'Unknown'),
        document_data=session.get('document_data', {}),
        summary=session.get('summary', {}),
        transactions=session.get('transactions', []),
        insights=session.get('insights', []),
        explanation=session.get('explanation', ''),
        charts=session.get('charts', {})
    )

@app.route('/transaction_details/<int:index>')
def transaction_details(index):
    """Get details for a specific transaction."""
    transactions = session.get('transactions', [])
    
    if index >= len(transactions):
        return jsonify({'error': 'Transaction not found'})
    
    return jsonify(transactions[index])

@app.route('/download_report')
def download_report():
    """Download CSV report of fraud analysis."""
    output_file = session.get('output_file')
    
    if not output_file or not os.path.exists(output_file):
        flash('Report file not found.')
        return redirect(url_for('show_results'))
    
    return send_file(
        output_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
    )

@app.route('/download_chart/<chart_type>')
def download_chart(chart_type):
    """Download visualization chart."""
    charts = session.get('charts', {})
    
    if not charts or chart_type not in charts:
        flash('Chart not found.')
        return redirect(url_for('show_results'))
    
    chart_path = charts[chart_type]['path']
    
    if not os.path.exists(chart_path):
        flash('Chart file not found.')
        return redirect(url_for('show_results'))
    
    return send_file(
        chart_path,
        mimetype='image/png',
        as_attachment=True,
        download_name=f"fraud_analysis_{chart_type}_{datetime.now().strftime('%Y%m%d')}.png"
    )

@app.route('/clear')
def clear_session():
    """Clear session data and return to upload page."""
    # Clean up files
    if 'file_path' in session and os.path.exists(session['file_path']):
        try:
            os.remove(session['file_path'])
        except:
            pass
    
    if 'output_file' in session and os.path.exists(session['output_file']):
        try:
            os.remove(session['output_file'])
        except:
            pass
    
    if 'charts' in session:
        for chart_type, chart_data in session['charts'].items():
            if 'path' in chart_data and os.path.exists(chart_data['path']):
                try:
                    os.remove(chart_data['path'])
                except:
                    pass
    
    # Clear session
    session.clear()
    
    flash('Analysis cleared. Upload a new document to begin.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template if it doesn't exist
    index_template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(index_template_path):
        with open(index_template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Enhanced Bank Statement Fraud Detection</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <!-- Custom CSS -->
    <style>
        .fraud-high {
            background-color: #ffdddd;
        }
        .fraud-medium {
            background-color: #ffffdd;
        }
        .fraud-low {
            background-color: #ddffdd;
        }
        .drag-area {
            border: 2px dashed #ccc;
            border-radius: 5px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
        }
        .drag-area.active {
            border-color: #0d6efd;
            background-color: #faf8f8;
        }
        .ai-badge {
            background-color: #8c43ff;
        }
        .processing-step {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #0d6efd;
            background-color: #f8f9fa;
        }
        .detected-field {
            background-color: #e8f4ff;
            border-radius: 3px;
            padding: 2px 5px;
            margin: 2px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                        <h3>AI-Enhanced Bank Statement Fraud Detection</h3>
                        <span class="badge ai-badge">AI Powered</span>
                    </div>
                    <div class="card-body">
                        {% with messages = get_flashed_messages() %}
                            {% if messages %}
                                {% for message in messages %}
                                    <div class="alert alert-warning">{{ message }}</div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}
                        
                        <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data">
                            <div class="drag-area" id="dropArea">
                                <h5>Drag & Drop any bank document here</h5>
                                <p class="text-muted">Accepts CSV, Excel, PDF, Images of statements</p>
                                <button type="button" class="btn btn-primary" id="browseBtn">Browse Files</button>
                                <input type="file" id="fileInput" name="file" accept=".csv,.xlsx,.xls,.pdf,.jpg,.jpeg,.png" hidden>
                                <div id="filePreview" class="mt-3" style="display: none;">
                                    <div class="alert alert-info d-flex align-items-center">
                                        <div>Selected file: <span id="fileName"></span></div>
                                        <button type="button" class="btn-close ms-auto" aria-label="Clear" id="clearFile"></button>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="largeAmountThreshold" class="form-label">Large Amount Threshold</label>
                                    <div class="input-group">
                                        <span class="input-group-text">$</span>
                                        <input type="number" class="form-control" id="largeAmountThreshold" name="largeAmountThreshold" value="2000">
                                    </div>
                                    <small class="form-text text-muted">Transactions above this amount will be flagged</small>
                                </div>
                            </div>
                            
                            <div class="text-center">
                                <button type="submit" class="btn btn-success btn-lg" id="submitBtn" disabled>
                                    Analyze Document with AI
                                </button>
                            </div>
                        </form>
                        
                        <div class="mt-5">
                            <h4>How it works</h4>
                            <div class="row">
                                <div class="col-md-4">
                                    <div class="card h-100">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">1. Document Processing</h5>
                                            <p class="card-text">AI analyzes your document structure and extracts transaction data, regardless of format</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card h-100">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">2. AI-Enhanced Detection</h5>
                                            <p class="card-text">Advanced algorithms identify suspicious patterns and potential fraud indicators</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card h-100">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">3. Detailed Analysis</h5>
                                            <p class="card-text">Get insights, visualizations, and explanations of detected fraud patterns</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('fileInput');
            const browseBtn = document.getElementById('browseBtn');
            const submitBtn = document.getElementById('submitBtn');
            const filePreview = document.getElementById('filePreview');
            const fileName = document.getElementById('fileName');
            const clearFile = document.getElementById('clearFile');
            
            // Browse button click
            browseBtn.addEventListener('click', () => {
                fileInput.click();
            });
            
            // File input change
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    updateFilePreview(this.files[0]);
                }
            });
            
            // Clear file selection
            clearFile.addEventListener('click', function(e) {
                e.stopPropagation();
                fileInput.value = '';
                filePreview.style.display = 'none';
                submitBtn.disabled = true;
            });
            
            // Drag and drop functionality
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                dropArea.classList.add('active');
            }
            
            function unhighlight() {
                dropArea.classList.remove('active');
            }
            
            dropArea.addEventListener('drop', function(e) {
                const dt = e.dataTransfer;
                const file = dt.files[0];
                fileInput.files = dt.files;
                updateFilePreview(file);
            });
            
            function updateFilePreview(file) {
                fileName.textContent = file.name;
                filePreview.style.display = 'block';
                submitBtn.disabled = false;
            }
        });
    </script>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")
    
    # Create results.html template if it doesn't exist
    results_template_path = os.path.join('templates', 'results.html')
    if not os.path.exists(results_template_path):
        with open(results_template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection Results</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <!-- Custom CSS -->
    <style>
        .fraud-high {
            background-color: #ffdddd;
        }
        .fraud-medium {
            background-color: #ffffdd;
        }
        .fraud-low {
            background-color: #ddffdd;
        }
        .ai-badge {
            background-color: #8c43ff;
        }
        .detected-field {
            background-color: #e8f4ff;
            border-radius: 3px;
            padding: 2px 5px;
            margin: 2px;
            display: inline-block;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .markdown h2 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .markdown h3 {
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
            font-weight: 600;
        }
        .markdown p {
            margin-bottom: 1rem;
        }
        .markdown ul {
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container mt-5 mb-5">
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                        <h3>Analysis Results: {{ filename }}</h3>
                        <div>
                            <a href="{{ url_for('clear') }}" class="btn btn-sm btn-outline-light">New Analysis</a>
                        </div>
                    </div>
                    
                    <div class="card-body">
                        {% with messages = get_flashed_messages() %}
                            {% if messages %}
                                {% for message in messages %}
                                    <div class="alert alert-warning">{{ message }}</div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}
                        
                        <!-- AI Insights -->
                        {% if insights %}
                        <div class="card mb-4">
                            <div class="card-header bg-info text-white">
                                <h5><i class="bi bi-lightbulb"></i> AI Insights</h5>
                            </div>
                            <div class="card-body">
                                {% for insight in insights %}
                                <p>{{ insight }}</p>
                                {% endfor %}
                            </div>
                        </div>
                        {% endif %}
                        
                        <!-- Summary Statistics -->
                        <h4>Summary</h4>
                        <div class="row mb-4">
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Transaction Count</h5>
                                        <p class="card-text">Total: {{ summary.total }}</p>
                                        <div class="d-flex justify-content-between">
                                            <span class="text-danger">High Risk: {{ summary.high_risk }} ({{ "%.1f"|format(summary.high_risk_pct) }}%)</span>
                                            <span class="text-warning">Medium Risk: {{ summary.medium_risk }} ({{ "%.1f"|format(summary.medium_risk_pct) }}%)</span>
                                            <span class="text-success">Low Risk: {{ summary.low_risk }} ({{ "%.1f"|format(summary.low_risk_pct) }}%)</span>
                                        </div>
                                        {% if summary.ai_enhanced > 0 %}
                                        <div class="mt-2"><span class="badge ai-badge">AI</span> Enhanced detection: {{ summary.ai_enhanced }} transactions</div>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Transaction Volume</h5>
                                        <p class="card-text">Total: ${{ "%.2f"|format(summary.total_amount) }}</p>
                                        <p class="text-danger">High Risk Amount: ${{ "%.2f"|format(summary.high_risk_amount) }} ({{ "%.1f"|format(summary.high_risk_amount_pct) }}%)</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Risk Distribution</h5>
                                        {% if charts.pie_chart %}
                                        <img src="data:image/png;base64,{{ charts.pie_chart.b64 }}" alt="Risk Distribution" class="img-fluid">
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Top Triggered Rules -->
                        {% if summary.top_rules %}
                        <div class="row mb-4">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>Top Triggered Rules</h5>
                                    </div>
                                    <div class="card-body">
                                        {% if charts.bar_chart %}
                                        <img src="data:image/png;base64,{{ charts.bar_chart.b64 }}" alt="Top Rules" class="img-fluid">
                                        {% else %}
                                        <ul>
                                            {% for rule, count in summary.top_rules %}
                                            <li>{{ rule }}: {{ count }} transactions</li>
                                            {% endfor %}
                                        </ul>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endif %}
                        
                        <!-- AI Explanation -->
                        {% if explanation %}
                        <div class="card mb-4">
                            <div class="card-header bg-info text-white">
                                <h5>AI Explanation</h5>
                            </div>
                            <div class="card-body markdown">
                                {{ explanation|replace("\n\n", "</p><p>")|replace("\n", "<br>")|replace("## ", "<h2>")|replace("### ", "<h3>")|replace("**", "<strong>")|replace("**", "</strong>")|safe }}
                            </div>
                        </div>
                        {% endif %}
                        
                        <!-- Results Table -->
                        <h4>Analysis Results</h4>
                        <div class="d-flex justify-content-between mb-3">
                            <div>
                                <span class="badge bg-danger me-2">High Risk</span>
                                <span class="badge bg-warning text-dark me-2">Medium Risk</span>
                                <span class="badge bg-success">Low Risk</span>
                            </div>
                            <a href="{{ url_for('download_report') }}" class="btn btn-sm btn-outline-primary">Download Report</a>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Date/Time</th>
                                        <th>Description</th>
                                        <th>Amount</th>
                                        <th>Risk Level</th>
                                        <th>Risk Score</th>
                                        <th>Triggered Rules</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for tx in transactions %}
                                    <tr class="fraud-{{ tx.risk_level|lower }}">
                                        <td>{{ tx.date }}</td>
                                        <td>{{ tx.description }}</td>
                                        <td>${{ "%.2f"|format(tx.amount) }}</td>
                                        <td>
                                            {% if tx.risk_level == 'High' %}
                                            <span class="badge bg-danger">High</span>
                                            {% elif tx.risk_level == 'Medium' %}
                                            <span class="badge bg-warning text-dark">Medium</span>
                                            {% else %}
                                            <span class="badge bg-success">Low</span>
                                            {% endif %}
                                        </td>
                                        <td>{{ tx.risk_score }}</td>
                                        <td>
                                            {{ tx.rules_text }}
                                            {% if tx.ai_enhanced %}
                                            <span class="badge ai-badge ms-1">AI</span>
                                            {% endif %}
                                        </td>
                                        <td>
                                            <button class="btn btn-sm btn-outline-info view-details" data-index="{{ loop.index0 }}">View</button>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Transaction Details Modal -->
    <div class="modal fade" id="transactionDetailsModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Transaction Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body" id="transactionDetailsBody">
                    <!-- Content will be dynamically inserted here -->
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Transaction details modal
            const detailsModal = new bootstrap.Modal(document.getElementById('transactionDetailsModal'));
            const detailsBody = document.getElementById('transactionDetailsBody');
            
            // Detail buttons
            const detailButtons = document.querySelectorAll('.view-details');
            detailButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const index = this.dataset.index;
                    fetch(`/transaction_details/${index}`)
                        .then(response => response.json())
                        .then(data => {
                            let riskBadgeClass = 'bg-success';
                            if (data.risk_level === 'High') {
                                riskBadgeClass = 'bg-danger';
                            } else if (data.risk_level === 'Medium') {
                                riskBadgeClass = 'bg-warning text-dark';
                            }
                            
                            // Format triggered rules
                            let triggeredRulesHtml = '<p>No rules triggered</p>';
                            if (data.rules && data.rules.length > 0) {
                                triggeredRulesHtml = '<ul class="list-group">';
                                data.rules.forEach(rule => {
                                    const aiTag = rule.includes('AI:') ? '<span class="badge ai-badge ms-2">AI</span>' : '';
                                    
                                    triggeredRulesHtml += `
                                        <li class="list-group-item">
                                            <div class="d-flex justify-content-between">
                                                <strong>${rule} ${aiTag}</strong>
                                            </div>
                                        </li>
                                    `;
                                });
                                triggeredRulesHtml += '</ul>';
                            }
                            
                            // Format original transaction data
                            let originalDataHtml = '<table class="table table-sm">';
                            for (const [key, value] of Object.entries(data.original_data)) {
                                originalDataHtml += `
                                    <tr>
                                        <td><strong>${key}</strong></td>
                                        <td>${value}</td>
                                    </tr>
                                `;
                            }
                            originalDataHtml += '</table>';
                            
                            detailsBody.innerHTML = `
                                <div class="row">
                                    <div class="col-md-6">
                                        <h5>Risk Assessment</h5>
                                        <div class="mb-3">
                                            <span class="badge ${riskBadgeClass} fs-6">${data.risk_level} Risk</span>
                                            <span class="ms-2">Score: ${data.risk_score}</span>
                                            ${data.ai_enhanced ? '<span class="badge ai-badge ms-2">AI Enhanced</span>' : ''}
                                        </div>
                                        
                                        <h6>Triggered Rules:</h6>
                                        ${triggeredRulesHtml}
                                    </div>
                                    <div class="col-md-6">
                                        <h5>Transaction Data</h5>
                                        ${originalDataHtml}
                                    </div>
                                </div>
                            `;
                            
                            detailsModal.show();
                        })
                        .catch(error => {
                            console.error('Error fetching transaction details:', error);
                            detailsBody.innerHTML = '<div class="alert alert-danger">Error loading transaction details</div>';
                            detailsModal.show();
                        });
                });
            });
        });
    </script>
</body>
</html>""")
    
    app.run(debug=True)