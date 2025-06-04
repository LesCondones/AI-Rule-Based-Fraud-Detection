#!/usr/bin/env python3
"""
Docker-optimized AI Fraud Detection Web Application
Simplified version to ensure compatibility in Docker environment
"""

from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify, session, send_file
import os
import io
import base64
import uuid
import re
import json
from datetime import datetime
from werkzeug.utils import secure_filename

# Try to import ML libraries with graceful fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Pandas not available - using fallback")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - using fallback")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - charts disabled")

# Try to import our fraud detection modules
try:
    from AIRuleBasedFraudDetection import (
        FraudDetectionApp,
        TENSORFLOW_AVAILABLE,
        NLP_AVAILABLE,
        OPENAI_AVAILABLE
    )
    FRAUD_DETECTION_AVAILABLE = True
except ImportError as e:
    FRAUD_DETECTION_AVAILABLE = False
    print(f"Fraud detection modules not available: {e}")

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'docker-development-key')
app.config['UPLOAD_FOLDER'] = '/app/uploads'
app.config['RESULT_FOLDER'] = '/app/results'
app.config['MODEL_FOLDER'] = '/app/models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['MODEL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'pdf', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML Templates
DOCKER_INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fraud Detection - Docker Version</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .docker-badge { background: linear-gradient(45deg, #0db7ed, #48c9fd); color: white; }
        .ai-badge { background: linear-gradient(45deg, #8c43ff, #ff6b9d); color: white; }
        .drag-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 20px; }
        .drag-area.active { border-color: #0d6efd; background-color: #f8f9ff; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
        <div class="container">
            <a class="navbar-brand" href="#"><i class="fab fa-docker me-2"></i>AI Fraud Detection</a>
            <div class="navbar-nav ms-auto">
                <span class="badge docker-badge me-2">Docker</span>
                <span class="badge ai-badge">AI Powered</span>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card shadow-lg border-0">
                    <div class="card-header bg-gradient text-white" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <h3 class="mb-1"><i class="fas fa-rocket me-2"></i>AI Fraud Detection - Docker Environment</h3>
                        <p class="mb-0 opacity-75">Running in isolated Docker container with optimized ML libraries</p>
                    </div>
                    
                    <div class="card-body p-4">
                        <!-- System Status -->
                        <div class="alert alert-info mb-4">
                            <h5><i class="fas fa-info-circle me-2"></i>System Status</h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <ul class="list-unstyled mb-0">
                                        <li><i class="fas fa-check status-good"></i> Docker Environment: Active</li>
                                        <li><i class="fas fa-{% if PANDAS_AVAILABLE %}check status-good{% else %}times status-error{% endif %}"></i> Pandas: {{ 'Available' if PANDAS_AVAILABLE else 'Not Available' }}</li>
                                        <li><i class="fas fa-{% if NUMPY_AVAILABLE %}check status-good{% else %}times status-error{% endif %}"></i> NumPy: {{ 'Available' if NUMPY_AVAILABLE else 'Not Available' }}</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <ul class="list-unstyled mb-0">
                                        <li><i class="fas fa-{% if FRAUD_DETECTION_AVAILABLE %}check status-good{% else %}times status-error{% endif %}"></i> ML Models: {{ 'Available' if FRAUD_DETECTION_AVAILABLE else 'Limited Mode' }}</li>
                                        <li><i class="fas fa-{% if TENSORFLOW_AVAILABLE %}check status-good{% else %}times status-warning{% endif %}"></i> TensorFlow: {{ 'Available' if TENSORFLOW_AVAILABLE else 'Not Available' }}</li>
                                        <li><i class="fas fa-{% if NLP_AVAILABLE %}check status-good{% else %}times status-warning{% endif %}"></i> NLP: {{ 'Available' if NLP_AVAILABLE else 'Not Available' }}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        {% with messages = get_flashed_messages() %}
                            {% if messages %}
                                {% for message in messages %}
                                    <div class="alert alert-warning alert-dismissible fade show" role="alert">
                                        <i class="fas fa-exclamation-circle me-2"></i>{{ message }}
                                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                    </div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}
                        
                        <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data" id="uploadForm">
                            <div class="drag-area" id="dropArea">
                                <div class="mb-3">
                                    <i class="fas fa-cloud-upload-alt" style="font-size: 3rem; color: #0d6efd; margin-bottom: 1rem;"></i>
                                    <h5>Drag & Drop Bank Statement Here</h5>
                                    <p class="text-muted mb-3">Supports CSV, Excel, PDF, and Image files</p>
                                    <button type="button" class="btn btn-primary btn-lg" id="browseBtn">
                                        <i class="fas fa-folder-open me-2"></i>Browse Files
                                    </button>
                                </div>
                                <input type="file" id="fileInput" name="file" accept=".csv,.xlsx,.xls,.pdf,.jpg,.jpeg,.png" hidden>
                                <div id="filePreview" class="mt-3" style="display: none;">
                                    <div class="alert alert-success d-flex align-items-center">
                                        <i class="fas fa-file-alt me-3 fs-4"></i>
                                        <div class="flex-grow-1 text-start">
                                            <strong>Selected file:</strong> <span id="fileName"></span>
                                            <br><small class="text-muted" id="fileSize"></small>
                                        </div>
                                        <button type="button" class="btn-close" aria-label="Clear" id="clearFile"></button>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mb-4">
                                <div class="col-md-6">
                                    <label for="largeAmountThreshold" class="form-label">
                                        <i class="fas fa-dollar-sign me-1"></i>Large Amount Threshold
                                    </label>
                                    <div class="input-group">
                                        <span class="input-group-text">$</span>
                                        <input type="number" class="form-control" id="largeAmountThreshold" 
                                               name="largeAmountThreshold" value="2000" min="0" step="100">
                                    </div>
                                    <small class="form-text text-muted">Transactions above this amount will be flagged as high-risk</small>
                                </div>
                            </div>
                            
                            <div class="d-grid">
                                <button type="submit" class="btn btn-success btn-lg" id="submitBtn" disabled>
                                    <i class="fas fa-rocket me-2"></i>
                                    <span id="submitText">Analyze Document with AI</span>
                                    <div class="spinner-border spinner-border-sm ms-2" id="submitSpinner" style="display: none;"></div>
                                </button>
                            </div>
                        </form>

                        <!-- Docker Info -->
                        <div class="mt-5 p-4 bg-light rounded">
                            <h5><i class="fab fa-docker me-2"></i>Docker Environment Benefits</h5>
                            <div class="row">
                                <div class="col-md-4">
                                    <div class="text-center p-3">
                                        <i class="fas fa-shield-alt text-primary mb-2" style="font-size: 2rem;"></i>
                                        <h6>Isolated Environment</h6>
                                        <small class="text-muted">No conflicts with your system</small>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-center p-3">
                                        <i class="fas fa-cogs text-success mb-2" style="font-size: 2rem;"></i>
                                        <h6>Pre-configured Setup</h6>
                                        <small class="text-muted">All dependencies included</small>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-center p-3">
                                        <i class="fas fa-rocket text-info mb-2" style="font-size: 2rem;"></i>
                                        <h6>Consistent Performance</h6>
                                        <small class="text-muted">Same results everywhere</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('fileInput');
            const browseBtn = document.getElementById('browseBtn');
            const submitBtn = document.getElementById('submitBtn');
            const filePreview = document.getElementById('filePreview');
            const fileName = document.getElementById('fileName');
            const clearFile = document.getElementById('clearFile');
            
            browseBtn.addEventListener('click', () => fileInput.click());
            
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    updateFilePreview(this.files[0]);
                }
            });
            
            clearFile.addEventListener('click', function(e) {
                e.stopPropagation();
                fileInput.value = '';
                filePreview.style.display = 'none';
                updateSubmitButton();
            });
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
            
            dropArea.addEventListener('drop', function(e) {
                const dt = e.dataTransfer;
                const file = dt.files[0];
                if (file) {
                    fileInput.files = dt.files;
                    updateFilePreview(file);
                }
            });
            
            function updateFilePreview(file) {
                fileName.textContent = file.name;
                filePreview.style.display = 'block';
                updateSubmitButton();
            }
            
            function updateSubmitButton() {
                const hasFile = fileInput.files.length > 0;
                submitBtn.disabled = !hasFile;
            }
            
            updateSubmitButton();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DOCKER_INDEX_TEMPLATE,
                                 PANDAS_AVAILABLE=PANDAS_AVAILABLE,
                                 NUMPY_AVAILABLE=NUMPY_AVAILABLE,
                                 FRAUD_DETECTION_AVAILABLE=FRAUD_DETECTION_AVAILABLE,
                                 TENSORFLOW_AVAILABLE=TENSORFLOW_AVAILABLE if FRAUD_DETECTION_AVAILABLE else False,
                                 NLP_AVAILABLE=NLP_AVAILABLE if FRAUD_DETECTION_AVAILABLE else False)

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
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        session['file_path'] = file_path
        session['original_filename'] = filename
        session['session_id'] = session_id
        session['threshold'] = float(request.form.get('largeAmountThreshold', 2000))
        
        return redirect(url_for('process_document'))
    
    flash('File type not allowed. Please upload a CSV, Excel, PDF, or image file.')
    return redirect(url_for('index'))

@app.route('/process')
def process_document():
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.')
        return redirect(url_for('index'))
    
    if FRAUD_DETECTION_AVAILABLE and PANDAS_AVAILABLE:
        # Full fraud detection processing
        try:
            app_instance = FraudDetectionApp()
            app_instance.user_profile['large_amount_threshold'] = session.get('threshold', 2000)
            
            document_data = app_instance.process_document(file_path)
            app_instance.map_fields()
            transactions = app_instance.process_transactions()
            results = app_instance.analyze_transactions_rule_based()
            
            session['results'] = results
            session['transactions'] = transactions
            
            flash('✅ Analysis complete! Full ML fraud detection performed.')
            return redirect(url_for('show_results'))
            
        except Exception as e:
            flash(f'Error in full analysis: {str(e)}. Using fallback mode.')
            return process_simple_csv()
    else:
        # Fallback simple processing
        return process_simple_csv()

def process_simple_csv():
    """Simple CSV processing fallback when full ML is not available"""
    file_path = session.get('file_path')
    
    if file_path.lower().endswith('.csv'):
        try:
            # Simple CSV analysis without pandas
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                flash('CSV file appears to be empty or has no data rows.')
                return redirect(url_for('index'))
            
            # Parse header
            header = lines[0].strip().split(',')
            
            # Find amount column
            amount_col = None
            for i, col in enumerate(header):
                if 'amount' in col.lower() or 'sum' in col.lower():
                    amount_col = i
                    break
            
            if amount_col is None:
                flash('Could not find amount column in CSV. Please ensure there is a column with "amount" in the name.')
                return redirect(url_for('index'))
            
            # Analyze transactions
            threshold = session.get('threshold', 2000)
            flagged = 0
            total = 0
            
            for line in lines[1:]:
                if line.strip():
                    try:
                        row = line.strip().split(',')
                        if len(row) > amount_col:
                            amount = float(re.sub(r'[^\d.-]', '', row[amount_col]))
                            total += 1
                            if amount > threshold:
                                flagged += 1
                    except (ValueError, IndexError):
                        continue
            
            session['simple_results'] = {
                'total': total,
                'flagged': flagged,
                'threshold': threshold,
                'rate': (flagged / total * 100) if total > 0 else 0
            }
            
            flash(f'✅ Simple analysis complete! Processed {total} transactions.')
            return redirect(url_for('show_simple_results'))
            
        except Exception as e:
            flash(f'Error processing CSV: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('In fallback mode, only CSV files are supported.')
        return redirect(url_for('index'))

@app.route('/results')
def show_results():
    if FRAUD_DETECTION_AVAILABLE and session.get('results'):
        # Show full results (implement full results template here)
        flash('Full ML results would be displayed here.')
        return redirect(url_for('index'))
    else:
        return redirect(url_for('show_simple_results'))

@app.route('/simple_results')
def show_simple_results():
    results = session.get('simple_results')
    if not results:
        flash('No analysis results available.')
        return redirect(url_for('index'))
    
    simple_results_template = f"""
    <div class="container mt-5">
        <div class="card">
            <div class="card-header bg-success text-white">
                <h3>Analysis Results - {session.get('original_filename', 'Unknown')}</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Summary</h5>
                        <ul>
                            <li>Total Transactions: {results['total']}</li>
                            <li>Large Transactions (>${results['threshold']}): {results['flagged']}</li>
                            <li>Risk Rate: {results['rate']:.1f}%</li>
                        </ul>
                    </div>
                </div>
                <a href="{url_for('index')}" class="btn btn-primary">Analyze Another File</a>
            </div>
        </div>
    </div>
    """
    
    return render_template_string(simple_results_template)

@app.route('/health')
def health_check():
    """Health check endpoint for Docker"""
    return jsonify({
        'status': 'healthy',
        'pandas': PANDAS_AVAILABLE,
        'numpy': NUMPY_AVAILABLE,
        'fraud_detection': FRAUD_DETECTION_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🐳 Starting AI Fraud Detection in Docker...")
    print(f"📊 Pandas Available: {'✅' if PANDAS_AVAILABLE else '❌'}")
    print(f"🔢 NumPy Available: {'✅' if NUMPY_AVAILABLE else '❌'}")
    print(f"🧠 ML Models Available: {'✅' if FRAUD_DETECTION_AVAILABLE else '❌'}")
    print("🌐 Server starting on http://0.0.0.0:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)