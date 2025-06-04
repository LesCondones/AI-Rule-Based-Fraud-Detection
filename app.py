"""
AI-Enhanced Bank Statement Fraud Detection Web Application
Self-contained version with embedded HTML templates

Author: Lester L. Artis Jr.
Created: 03/15/2025
Enhanced: Added Deep Learning Autoencoder Integration + Embedded Templates

This Flask web application is completely self-contained and only requires
the AIRuleBasedFraudDetection.py file to function.
"""

from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify, session, send_file
import os
import io
import base64
import uuid
import re
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from werkzeug.utils import secure_filename

# Import our enhanced fraud detection classes
from AIRuleBasedFraudDetection import (
    AIDocumentProcessor, 
    EnhancedFraudDetectionSystem, 
    FraudDetectionApp,
    AutoencoderFraudDetector,
    NLPFraudDetector,
    FraudDetectionChatbot,
    TENSORFLOW_AVAILABLE,
    NLP_AVAILABLE,
    OPENAI_AVAILABLE
)

# Create Flask app with enhanced configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['RESULT_FOLDER'] = 'results'
app.config['MODEL_FOLDER'] = 'models'  # For saving autoencoder models

# Create directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['MODEL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'pdf', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# EMBEDDED HTML TEMPLATES
# ============================================================================

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Enhanced Bank Statement Fraud Detection</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .ai-badge { background: linear-gradient(45deg, #8c43ff, #ff6b9d); color: white; }
        .ml-badge { background: linear-gradient(45deg, #00d2ff, #3a7bd5); color: white; }
        .drag-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 20px; transition: all 0.3s ease; }
        .drag-area.active { border-color: #0d6efd; background-color: #f8f9ff; transform: scale(1.02); }
        .detection-method-card { border: 2px solid transparent; transition: all 0.3s ease; cursor: pointer; }
        .detection-method-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
        .detection-method-card.selected { border-color: #0d6efd; background-color: #f8f9ff; }
        .feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
        .advanced-options { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; padding: 20px; margin-top: 20px; }
        .neural-network-icon { background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .status-indicator { display: inline-flex; align-items-center; gap: 8px; padding: 8px 16px; border-radius: 20px; font-weight: 500; }
        .status-available { background-color: #d1edff; color: #0066cc; }
        .status-unavailable { background-color: #ffe6e6; color: #cc0000; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
        <div class="container">
            <a class="navbar-brand" href="#"><i class="fas fa-shield-alt me-2"></i>AI Fraud Detection System</a>
            <div class="navbar-nav ms-auto">
                <span class="badge ai-badge me-2">AI Powered</span>
                <span class="badge ml-badge">Deep Learning</span>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card shadow-lg border-0">
                    <div class="card-header bg-gradient text-white" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h3 class="mb-1"><i class="fas fa-brain me-2"></i>AI-Enhanced Fraud Detection</h3>
                                <p class="mb-0 opacity-75">Upload bank statements for intelligent fraud analysis using traditional rules and deep learning</p>
                            </div>
                            <div class="text-end">
                                {% if tensorflow_available %}
                                <div class="status-indicator status-available">
                                    <i class="fas fa-check-circle"></i><span>Deep Learning Ready</span>
                                </div>
                                {% else %}
                                <div class="status-indicator status-unavailable">
                                    <i class="fas fa-exclamation-triangle"></i><span>TensorFlow Not Available</span>
                                </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <div class="card-body p-4">
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
                                    <i class="fas fa-cloud-upload-alt feature-icon text-primary"></i>
                                    <h5>Drag & Drop Bank Statement Here</h5>
                                    <p class="text-muted mb-3">Supports CSV, Excel, PDF, and Image files</p>
                                    <button type="button" class="btn btn-primary btn-lg" id="browseBtn">
                                        <i class="fas fa-folder-open me-2"></i>Browse Files
                                    </button>
                                </div>
                                <input type="file" id="fileInput" name="file" accept=".csv,.xlsx,.xls,.pdf,.jpg,.jpeg,.png" hidden>
                                <div id="filePreview" class="mt-3" style="display: none;">
                                    <div class="alert alert-info d-flex align-items-center">
                                        <i class="fas fa-file-alt me-3 fs-4"></i>
                                        <div class="flex-grow-1 text-start">
                                            <strong>Selected file:</strong> <span id="fileName"></span>
                                            <br><small class="text-muted" id="fileSize"></small>
                                        </div>
                                        <button type="button" class="btn-close" aria-label="Clear" id="clearFile"></button>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mb-4">
                                <h5 class="mb-3"><i class="fas fa-cogs me-2"></i>Choose Detection Method</h5>
                                <div class="row">
                                    <div class="col-md-4 mb-3">
                                        <div class="card detection-method-card h-100" data-method="rules">
                                            <div class="card-body text-center">
                                                <i class="fas fa-list-check feature-icon text-warning"></i>
                                                <h6 class="card-title">Rule-Based Only</h6>
                                                <p class="card-text small">Traditional fraud detection using predefined rules and patterns</p>
                                                <div class="form-check">
                                                    <input class="form-check-input" type="radio" name="detectionMethod" value="rules" id="methodRules">
                                                    <label class="form-check-label" for="methodRules">Select</label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="card detection-method-card h-100" data-method="autoencoder" 
                                             {% if not tensorflow_available %}style="opacity: 0.6;" title="Requires TensorFlow"{% endif %}>
                                            <div class="card-body text-center">
                                                <i class="fas fa-project-diagram feature-icon neural-network-icon"></i>
                                                <h6 class="card-title">Deep Learning Only
                                                    {% if not tensorflow_available %}
                                                    <i class="fas fa-lock text-muted ms-1" data-bs-toggle="tooltip" title="TensorFlow required"></i>
                                                    {% endif %}
                                                </h6>
                                                <p class="card-text small">Unsupervised anomaly detection using neural network autoencoder</p>
                                                <div class="form-check">
                                                    <input class="form-check-input" type="radio" name="detectionMethod" value="autoencoder" 
                                                           id="methodAutoencoder" {% if not tensorflow_available %}disabled{% endif %}>
                                                    <label class="form-check-label" for="methodAutoencoder">Select</label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="card detection-method-card h-100 selected" data-method="both">
                                            <div class="card-body text-center">
                                                <i class="fas fa-layer-group feature-icon text-success"></i>
                                                <h6 class="card-title">Both Methods <span class="badge bg-success ms-1">Recommended</span></h6>
                                                <p class="card-text small">Comprehensive analysis with rule-based + AI comparison</p>
                                                <div class="form-check">
                                                    <input class="form-check-input" type="radio" name="detectionMethod" value="both" 
                                                           id="methodBoth" checked {% if not tensorflow_available %}disabled{% endif %}>
                                                    <label class="form-check-label" for="methodBoth">Select</label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                {% if not tensorflow_available %}
                                <div class="alert alert-info">
                                    <i class="fas fa-info-circle me-2"></i>
                                    <strong>Note:</strong> Deep learning features require TensorFlow. Install with: <code>pip install tensorflow</code>
                                </div>
                                {% endif %}
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
                            
                            <div id="autoencoderOptions" class="advanced-options" style="display: none;">
                                <h6 class="mb-3"><i class="fas fa-brain me-2"></i>Deep Learning Configuration</h6>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="aeThresholdStrategy" class="form-label">Anomaly Threshold Strategy</label>
                                        <select class="form-select" id="aeThresholdStrategy" name="aeThresholdStrategy">
                                            <option value="percentile" selected>Percentile-based</option>
                                            <option value="std_dev">Standard Deviation</option>
                                            <option value="manual">Manual Threshold</option>
                                        </select>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="aeThresholdValue" class="form-label">Threshold Value</label>
                                        <input type="number" class="form-control" id="aeThresholdValue" 
                                               name="aeThresholdValue" value="95" min="1" max="99" step="1">
                                        <small class="form-text text-muted" id="thresholdHelp">95th percentile (top 5% flagged as anomalies)</small>
                                    </div>
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
            const autoencoderOptions = document.getElementById('autoencoderOptions');
            
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
            
            document.querySelectorAll('.detection-method-card').forEach(card => {
                card.addEventListener('click', function() {
                    const method = this.dataset.method;
                    const radio = document.getElementById(`method${method.charAt(0).toUpperCase() + method.slice(1)}`);
                    
                    if (!radio.disabled) {
                        document.querySelectorAll('.detection-method-card').forEach(c => c.classList.remove('selected'));
                        this.classList.add('selected');
                        radio.checked = true;
                        updateAutoencoderOptions(method);
                    }
                });
            });
            
            function updateAutoencoderOptions(method) {
                if (method === 'autoencoder' || method === 'both') {
                    autoencoderOptions.style.display = 'block';
                } else {
                    autoencoderOptions.style.display = 'none';
                }
            }
            
            function updateSubmitButton() {
                const hasFile = fileInput.files.length > 0;
                submitBtn.disabled = !hasFile;
            }
            
            updateAutoencoderOptions('both');
            updateSubmitButton();
        });
    </script>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection Results - {{ filename }}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .fraud-high { background-color: #ffdddd; }
        .fraud-medium { background-color: #ffffdd; }
        .fraud-low { background-color: #ddffdd; }
        .anomaly-detected { background-color: #ffe6e6; }
        .anomaly-normal { background-color: #e6ffe6; }
        .ai-badge { background: linear-gradient(45deg, #8c43ff, #ff6b9d); color: white; }
        .ml-badge { background: linear-gradient(45deg, #00d2ff, #3a7bd5); color: white; }
        .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; transition: transform 0.3s ease; }
        .metric-card:hover { transform: translateY(-5px); }
        .interactive-controls { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; padding: 20px; }
        .table-container { max-height: 600px; overflow-y: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-shield-alt me-2"></i>AI Fraud Detection System
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text me-3">Analysis: {{ filename }}</span>
                <a href="{{ url_for('clear') }}" class="btn btn-outline-light btn-sm">
                    <i class="fas fa-plus me-1"></i>New Analysis
                </a>
            </div>
        </div>
    </nav>

    <div class="container-fluid">
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
        
        <!-- Summary Section -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Analysis Summary</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            {% if rule_summary %}
                            <div class="col-md-6">
                                <h6><i class="fas fa-list-check me-2"></i>Rule-Based Detection</h6>
                                <div class="row">
                                    <div class="col-4">
                                        <div class="metric-card p-3 text-center">
                                            <h4 class="text-danger">{{ rule_summary.high_risk }}</h4>
                                            <small>High Risk</small>
                                        </div>
                                    </div>
                                    <div class="col-4">
                                        <div class="metric-card p-3 text-center">
                                            <h4 class="text-warning">{{ rule_summary.medium_risk }}</h4>
                                            <small>Medium Risk</small>
                                        </div>
                                    </div>
                                    <div class="col-4">
                                        <div class="metric-card p-3 text-center">
                                            <h4 class="text-success">{{ rule_summary.low_risk }}</h4>
                                            <small>Low Risk</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            {% endif %}
                            
                            {% if ae_summary %}
                            <div class="col-md-6">
                                <h6><i class="fas fa-brain me-2"></i>Autoencoder Detection</h6>
                                <div class="row">
                                    <div class="col-6">
                                        <div class="metric-card p-3 text-center">
                                            <h4 class="text-danger">{{ ae_summary.anomalies }}</h4>
                                            <small>Anomalies</small>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="metric-card p-3 text-center">
                                            <h4 class="text-success">{{ ae_summary.total - ae_summary.anomalies }}</h4>
                                            <small>Normal</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            {% endif %}
                        </div>
                        
                        {% if charts %}
                        <div class="row mt-4">
                            {% for chart_type, chart_data in charts.items() %}
                            <div class="col-md-6 mb-3">
                                <div class="text-center">
                                    <img src="data:image/png;base64,{{ chart_data.b64 }}" alt="{{ chart_type }}" class="img-fluid" style="max-height: 300px;">
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Interactive Threshold Tuning (if autoencoder) -->
        {% if ae_summary and tensorflow_available %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="interactive-controls">
                    <h6 class="mb-3"><i class="fas fa-sliders-h me-2"></i>Interactive Threshold Tuning</h6>
                    <div class="row align-items-end">
                        <div class="col-md-3 mb-3">
                            <label for="thresholdStrategy" class="form-label">Strategy</label>
                            <select class="form-select" id="thresholdStrategy">
                                <option value="percentile">Percentile</option>
                                <option value="std_dev">Std Deviation</option>
                                <option value="manual">Manual</option>
                            </select>
                        </div>
                        <div class="col-md-3 mb-3">
                            <label for="thresholdValue" class="form-label">Value</label>
                            <input type="number" class="form-control" id="thresholdValue" value="95" step="0.1">
                        </div>
                        <div class="col-md-4 mb-3">
                            <label class="form-label">Current Threshold</label>
                            <div class="form-control bg-light" id="currentThreshold">{{ "%.6f"|format(current_ae_threshold) }}</div>
                        </div>
                        <div class="col-md-2 mb-3">
                            <button class="btn btn-light w-100" id="updateThreshold">
                                <i class="fas fa-sync"></i> Update
                            </button>
                        </div>
                    </div>
                    <div id="thresholdUpdateStatus" class="mt-2" style="display: none;"></div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <!-- Transactions Table -->
        <div class="row">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-secondary text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0"><i class="fas fa-table me-2"></i>Transaction Analysis</h5>
                        <a href="{{ url_for('download_report') }}" class="btn btn-outline-light btn-sm">
                            <i class="fas fa-download me-1"></i>Download Report
                        </a>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-container">
                            <table class="table table-hover table-sm mb-0">
                                <thead class="table-dark sticky-top">
                                    <tr>
                                        <th>Date/Time</th>
                                        <th>Description</th>
                                        <th>Amount</th>
                                        {% if rule_summary %}
                                        <th>Rule Risk</th>
                                        {% endif %}
                                        {% if ae_summary %}
                                        <th>Anomaly</th>
                                        <th>Error</th>
                                        {% endif %}
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for tx in (rule_transactions or ae_transactions) %}
                                    <tr class="{% if tx.get('risk_level') == 'High' %}fraud-high{% elif tx.get('risk_level') == 'Medium' %}fraud-medium{% elif tx.get('risk_level') == 'Low' %}fraud-low{% elif tx.get('is_anomaly') %}anomaly-detected{% else %}anomaly-normal{% endif %}">
                                        <td>{{ tx.date }}</td>
                                        <td>{{ tx.description[:50] }}{% if tx.description|length > 50 %}...{% endif %}</td>
                                        <td>${{ "%.2f"|format(tx.amount) }}</td>
                                        {% if rule_summary %}
                                        <td>
                                            {% if tx.get('risk_level') == 'High' %}
                                            <span class="badge bg-danger">High</span>
                                            {% elif tx.get('risk_level') == 'Medium' %}
                                            <span class="badge bg-warning text-dark">Medium</span>
                                            {% else %}
                                            <span class="badge bg-success">Low</span>
                                            {% endif %}
                                        </td>
                                        {% endif %}
                                        {% if ae_summary %}
                                        <td>
                                            {% if tx.get('is_anomaly') %}
                                            <span class="badge bg-danger">Yes</span>
                                            {% else %}
                                            <span class="badge bg-success">No</span>
                                            {% endif %}
                                        </td>
                                        <td>{{ "%.6f"|format(tx.get('reconstruction_error', 0)) }}</td>
                                        {% endif %}
                                        <td>
                                            <button class="btn btn-sm btn-outline-info view-details" data-index="{{ loop.index0 }}">
                                                <i class="fas fa-eye"></i>
                                            </button>
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
    <div class="modal fade" id="transactionDetailsModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Transaction Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="transactionDetailsBody"></div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const detailsModal = new bootstrap.Modal(document.getElementById('transactionDetailsModal'));
            const detailsBody = document.getElementById('transactionDetailsBody');
            
            document.querySelectorAll('.view-details').forEach(button => {
                button.addEventListener('click', function() {
                    const index = this.dataset.index;
                    // Simple transaction details display
                    detailsBody.innerHTML = `<p>Transaction ${index} details would be displayed here.</p>`;
                    detailsModal.show();
                });
            });
            
            const updateThresholdBtn = document.getElementById('updateThreshold');
            if (updateThresholdBtn) {
                updateThresholdBtn.addEventListener('click', function() {
                    const strategy = document.getElementById('thresholdStrategy').value;
                    const value = parseFloat(document.getElementById('thresholdValue').value);
                    
                    fetch('/update_autoencoder_threshold', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({strategy: strategy, value: value})
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('currentThreshold').textContent = data.new_threshold.toFixed(6);
                            alert('Threshold updated successfully! Page will reload to show updated results.');
                            setTimeout(() => window.location.reload(), 2000);
                        } else {
                            alert('Error updating threshold: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Network error occurred');
                    });
                });
            }
        });
    </script>
</body>
</html>
"""

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE, tensorflow_available=TENSORFLOW_AVAILABLE)

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
        
        # Get form parameters
        threshold = float(request.form.get('largeAmountThreshold', 2000))
        detection_method = request.form.get('detectionMethod', 'both')
        
        session['threshold'] = threshold
        session['detection_method'] = detection_method
        
        if detection_method in ['autoencoder', 'both'] and TENSORFLOW_AVAILABLE:
            session['ae_threshold_strategy'] = request.form.get('aeThresholdStrategy', 'percentile')
            session['ae_threshold_value'] = float(request.form.get('aeThresholdValue', 95))
        
        return redirect(url_for('process_document'))
    
    flash('File type not allowed. Please upload a CSV, Excel, PDF, or image file.')
    return redirect(url_for('index'))

@app.route('/process')
def process_document():
    file_path = session.get('file_path')
    session_id = session.get('session_id')
    detection_method = session.get('detection_method', 'both')
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.')
        return redirect(url_for('index'))
    
    try:
        # Initialize fraud detection app
        app_instance = FraudDetectionApp()
        app_instance.user_profile['large_amount_threshold'] = session.get('threshold', 2000)
        
        # Process document
        document_data = app_instance.process_document(file_path)
        app_instance.map_fields()
        transactions = app_instance.process_transactions()
        
        # Store basic info
        session['total_transactions'] = len(transactions)
        
        # Run rule-based detection
        rule_results = None
        if detection_method in ['rules', 'both']:
            rule_results = app_instance.analyze_transactions_rule_based()
            
            # Process rule-based results
            total = len(rule_results)
            high_risk = sum(1 for r in rule_results if r['fraud_likelihood'] == 'High')
            medium_risk = sum(1 for r in rule_results if r['fraud_likelihood'] == 'Medium')
            low_risk = sum(1 for r in rule_results if r['fraud_likelihood'] == 'Low')
            
            session['rule_summary'] = {
                'total': total,
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk,
                'high_risk_pct': (high_risk / total) * 100 if total > 0 else 0,
                'medium_risk_pct': (medium_risk / total) * 100 if total > 0 else 0,
                'low_risk_pct': (low_risk / total) * 100 if total > 0 else 0
            }
            
            # Format transactions for display
            session['rule_transactions'] = format_transactions_for_display(transactions, rule_results, 'rule_based')
        
        # Run NLP detection
        nlp_results = None
        if detection_method in ['nlp', 'both', 'all'] and NLP_AVAILABLE:
            try:
                nlp_results = app_instance.analyze_transactions_nlp()
                
                if nlp_results:
                    summary = nlp_results['summary']
                    session['nlp_summary'] = {
                        'total': summary['total_transactions'],
                        'anomalies': summary['anomalies_detected'],
                        'anomaly_rate': summary['anomaly_rate'],
                        'clusters': summary['unique_clusters']
                    }
                    
                    session['nlp_transactions'] = format_transactions_for_display(transactions, nlp_results, 'nlp')
            except Exception as e:
                print(f"Error in NLP analysis: {str(e)}")
        
        # Run autoencoder detection
        ae_results = None
        if detection_method in ['autoencoder', 'both', 'all'] and TENSORFLOW_AVAILABLE:
            if app_instance.autoencoder_detector:
                app_instance.autoencoder_detector.threshold_strategy = session.get('ae_threshold_strategy', 'percentile')
                app_instance.autoencoder_detector.threshold_value = session.get('ae_threshold_value', 95)
                
                ae_results = app_instance.analyze_transactions_autoencoder(train_on_all=True)
                
                if ae_results:
                    summary = ae_results['summary']
                    session['ae_summary'] = {
                        'total': summary['total_transactions'],
                        'anomalies': summary['anomalies_detected'],
                        'anomaly_rate': summary['anomaly_rate_percent'],
                        'threshold': summary['threshold_used'],
                        'mean_error': summary['mean_reconstruction_error']
                    }
                    
                    session['current_ae_threshold'] = app_instance.autoencoder_detector.anomaly_threshold
                    session['reconstruction_errors'] = ae_results['reconstruction_errors']
                    session['ae_transactions'] = format_transactions_for_display(transactions, ae_results, 'autoencoder')
        
        # Generate charts
        charts = generate_enhanced_visualizations(app_instance, session_id, rule_results, ae_results)
        session['charts'] = charts
        
        # Export results
        output_file = generate_detailed_report(session_id, transactions, rule_results, ae_results)
        session['output_file'] = output_file
        
        return redirect(url_for('show_results'))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error processing document: {str(e)}")
        return redirect(url_for('index'))

@app.route('/results')
def show_results():
    if not (session.get('rule_transactions') or session.get('ae_transactions')):
        flash('No analysis results available. Please upload and process a document first.')
        return redirect(url_for('index'))
    
    template_data = {
        'filename': session.get('original_filename', 'Unknown'),
        'detection_method': session.get('detection_method', 'both'),
        'total_transactions': session.get('total_transactions', 0),
        'charts': session.get('charts', {}),
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'rule_summary': session.get('rule_summary'),
        'rule_transactions': session.get('rule_transactions', []),
        'ae_summary': session.get('ae_summary'),
        'ae_transactions': session.get('ae_transactions', []),
        'current_ae_threshold': session.get('current_ae_threshold')
    }
    
    return render_template_string(RESULTS_TEMPLATE, **template_data)

@app.route('/update_autoencoder_threshold', methods=['POST'])
def update_autoencoder_threshold():
    if not TENSORFLOW_AVAILABLE:
        return jsonify({'error': 'TensorFlow not available'}), 400
    
    try:
        strategy = request.json.get('strategy', 'percentile')
        value = float(request.json.get('value', 95))
        
        reconstruction_errors = session.get('reconstruction_errors')
        if not reconstruction_errors:
            return jsonify({'error': 'No reconstruction errors available'}), 400
        
        # Calculate new threshold
        if strategy == 'percentile':
            new_threshold = np.percentile(reconstruction_errors, value)
        elif strategy == 'std_dev':
            mean_error = np.mean(reconstruction_errors)
            std_error = np.std(reconstruction_errors)
            new_threshold = mean_error + (value * std_error)
        elif strategy == 'manual':
            new_threshold = value
        else:
            return jsonify({'error': 'Invalid threshold strategy'}), 400
        
        # Recalculate anomaly flags
        anomaly_flags = [error > new_threshold for error in reconstruction_errors]
        anomalies_detected = sum(anomaly_flags)
        anomaly_rate = (anomalies_detected / len(reconstruction_errors)) * 100
        
        # Update session
        session['current_ae_threshold'] = new_threshold
        
        return jsonify({
            'success': True,
            'new_threshold': new_threshold,
            'anomalies_detected': anomalies_detected,
            'anomaly_rate': anomaly_rate,
            'total_transactions': len(reconstruction_errors)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report')
def download_report():
    output_file = session.get('output_file')
    
    if not output_file or not os.path.exists(output_file):
        flash('Report file not found.')
        return redirect(url_for('show_results'))
    
    return send_file(
        output_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """Handle chatbot interactions"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Initialize chatbot if not exists
        if 'chatbot' not in session:
            session['chatbot'] = True
            
        # Create a temporary FraudDetectionApp to handle the chat
        app_instance = FraudDetectionApp()
        
        # Set context if available
        if session.get('transactions'):
            app_instance.transactions = session['transactions']
            app_instance.rule_based_results = session.get('rule_results', [])
            app_instance.nlp_results = session.get('nlp_results')
            app_instance.autoencoder_results = session.get('autoencoder_results')
        
        # Get response
        response = app_instance.chat_with_ai(message)
        
        return jsonify({
            'success': True,
            'response': response,
            'message': message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear')
def clear_session():
    # Clean up files
    files_to_clean = [
        session.get('file_path'),
        session.get('output_file')
    ]
    
    for file_path in files_to_clean:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
    
    session.clear()
    flash('Analysis cleared. Upload a new document to begin.')
    return redirect(url_for('index'))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_transactions_for_display(transactions: list, results, result_type: str) -> list:
    """Format transactions with results for web display."""
    formatted_transactions = []
    
    for i, tx in enumerate(transactions):
        formatted_tx = {
            'date': format_timestamp(tx.get('timestamp')),
            'description': tx.get('description') or tx.get('merchantCategory', 'Unknown'),
            'amount': float(tx.get('amount', 0)),
            'original_data': tx.get('originalData', {})
        }
        
        if result_type == 'rule_based' and i < len(results):
            result = results[i]
            formatted_tx.update({
                'risk_level': result['fraud_likelihood'],
                'risk_score': result['risk_score'],
                'rules': [rule['rule_name'] for rule in result['triggered_rules']],
                'rules_text': ', '.join(rule['rule_name'] for rule in result['triggered_rules']) if result['triggered_rules'] else 'None',
                'ai_enhanced': result.get('ai_enhanced', False)
            })
            
        elif result_type == 'autoencoder' and results and i < len(results['detailed_results']):
            result = results['detailed_results'][i]
            formatted_tx.update({
                'reconstruction_error': result['reconstruction_error'],
                'is_anomaly': result['is_anomaly'],
                'anomaly_score': result['anomaly_score'],
                'confidence': result['confidence'],
                'error_ratio': result['error_ratio']
            })
        
        formatted_transactions.append(formatted_tx)
    
    return formatted_transactions

def format_timestamp(timestamp):
    """Format timestamp to readable date format."""
    if not timestamp:
        return "Unknown"
    
    try:
        if isinstance(timestamp, str):
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
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
            
            return timestamp
            
        elif isinstance(timestamp, datetime):
            return timestamp.strftime("%m/%d/%Y %I:%M %p")
            
        return str(timestamp)
    except:
        return str(timestamp)

def generate_enhanced_visualizations(app_instance, session_id, rule_results=None, ae_results=None):
    """Generate visualizations and return base64 encoded images."""
    charts = {}
    
    try:
        # Rule-based risk distribution
        if rule_results:
            plt.figure(figsize=(8, 6))
            risk_counts = {
                'High': sum(1 for r in rule_results if r['fraud_likelihood'] == 'High'),
                'Medium': sum(1 for r in rule_results if r['fraud_likelihood'] == 'Medium'),
                'Low': sum(1 for r in rule_results if r['fraud_likelihood'] == 'Low')
            }
            
            colors = ['#ff6b6b', '#feca57', '#1dd1a1']
            wedges, texts, autotexts = plt.pie(
                risk_counts.values(),
                labels=risk_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title('Rule-Based Risk Distribution', fontweight='bold', fontsize=14)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            rule_pie_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()
            
            charts['rule_pie_chart'] = {'b64': rule_pie_b64}
        
        # Autoencoder error distribution
        if ae_results:
            plt.figure(figsize=(10, 6))
            
            errors = ae_results['reconstruction_errors']
            threshold = ae_results['summary']['threshold_used']
            
            plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='Reconstruction Errors')
            plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Anomaly Threshold: {threshold:.4f}')
            
            mean_error = np.mean(errors)
            plt.axvline(mean_error, color='green', linestyle='-', linewidth=1, alpha=0.7,
                       label=f'Mean: {mean_error:.4f}')
            
            plt.title('Autoencoder Reconstruction Error Distribution', fontweight='bold', fontsize=14)
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            ae_hist_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()
            
            charts['ae_histogram'] = {'b64': ae_hist_b64}
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    return charts

def generate_detailed_report(session_id, transactions, rule_results=None, ae_results=None):
    """Generate a comprehensive CSV report."""
    try:
        data = []
        
        for i, tx in enumerate(transactions):
            row = tx.get('originalData', {}).copy()
            
            # Add basic transaction info
            row['Transaction_ID'] = tx.get('id', f'tx_{i}')
            row['Amount'] = tx.get('amount', 0)
            row['Timestamp'] = tx.get('timestamp', '')
            row['Description'] = tx.get('description', '')
            
            # Add rule-based results
            if rule_results and i < len(rule_results):
                rule_result = rule_results[i]
                row['Rule_Risk_Level'] = rule_result['fraud_likelihood']
                row['Rule_Risk_Score'] = rule_result['risk_score']
                row['Rule_Triggered_Rules'] = '; '.join(
                    rule['rule_name'] for rule in rule_result['triggered_rules']
                )
            
            # Add autoencoder results
            if ae_results and i < len(ae_results['detailed_results']):
                ae_detail = ae_results['detailed_results'][i]
                row['AE_Reconstruction_Error'] = ae_detail['reconstruction_error']
                row['AE_Is_Anomaly'] = 'Yes' if ae_detail['is_anomaly'] else 'No'
                row['AE_Anomaly_Score'] = ae_detail['anomaly_score']
                row['AE_Confidence'] = ae_detail['confidence']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        output_file = os.path.join(app.config['RESULT_FOLDER'], f"{session_id}_fraud_analysis.csv")
        df.to_csv(output_file, index=False)
        
        return output_file
        
    except Exception as e:
        print(f"Error generating detailed report: {str(e)}")
        return None

if __name__ == '__main__':
    print("🚀 Starting AI-Enhanced Fraud Detection Web Application")
    print(f"📊 TensorFlow Available: {'✅ Yes' if TENSORFLOW_AVAILABLE else '❌ No'}")
    if not TENSORFLOW_AVAILABLE:
        print("   Install TensorFlow for deep learning features: pip install tensorflow")
    print("🌐 Open your browser and navigate to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)