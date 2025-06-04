# AI-Enhanced Bank Statement Fraud Detection with ML & Chatbot

An intelligent system for analyzing bank statements and financial documents to detect potential fraud using advanced AI techniques including unsupervised learning, NLP, and conversational AI.

## Overview

This project provides a complete end-to-end solution for fraud detection in financial documents. It can process virtually any type of bank statement or transaction data (CSV, Excel, PDF, images, etc.), automatically extract transaction information, and apply multiple AI-powered detection algorithms to identify suspicious activities.

## Features

### 🔍 **Multi-Modal Fraud Detection**
- **Rule-Based Detection**: Traditional pattern-based fraud detection with customizable rules
- **NLP-Based Analysis**: Text analysis of transaction descriptions using TF-IDF and clustering
- **Deep Learning**: Autoencoder neural networks for unsupervised anomaly detection
- **Ensemble Methods**: Combine multiple approaches for comprehensive analysis

### 🤖 **AI-Powered Chatbot**
- **Interactive Analysis**: Ask questions about fraud detection results in natural language
- **Intelligent Explanations**: Get detailed explanations of why transactions were flagged
- **Actionable Insights**: Receive recommendations for further investigation
- **Conversation History**: Maintain context throughout your analysis session

### 📊 **Advanced Analytics**
- **Universal Document Processing**: Automatically handles any file format containing transaction data
- **Intelligent Field Detection**: AI automatically identifies and maps transaction fields
- **Visual Analysis**: Interactive charts and visualizations of fraud patterns
- **Clustering Analysis**: Identify transaction patterns and group similar behaviors
- **Anomaly Scoring**: Quantitative risk assessment for each transaction

### 🌐 **User Interface**
- **Web Interface**: User-friendly web application for uploading and analyzing documents
- **Command Line Interface**: Scriptable interface for automated processing
- **Interactive Chat Mode**: Real-time conversation with AI about your results
- **Comprehensive Reporting**: Detailed fraud reports available for download

## Installation

### Requirements

- Python 3.8 or higher
- Flask (for web interface)
- pandas, numpy
- matplotlib
- PyPDF2 (for PDF processing)
- pdfplumber (for advanced PDF parsing)
- pytesseract and PIL (for image processing)
- Various optional dependencies for extended file format support

### Basic Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bank-fraud-detection.git
   cd bank-fraud-detection
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. For image processing support (optional):
   - Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
   - Install additional dependencies: `pip install pytesseract pillow`

### File Structure

- `AIRuleBasedFraudDetection.py`: Core fraud detection and document processing code
- `app.py`: Flask web application for the user interface
- `uploads/`: Directory for uploaded documents
- `results/`: Directory for analysis results
- `templates/`: HTML templates for the web interface

## Usage

### Command Line Interface

To analyze a document from the command line:

```bash
# Use all detection methods
python AIRuleBasedFraudDetection.py path/to/your/document.csv --method all

# Use specific detection method
python AIRuleBasedFraudDetection.py path/to/your/document.csv --method nlp

# Enable interactive chat mode after analysis
python AIRuleBasedFraudDetection.py path/to/your/document.csv --chat

# Use OpenAI for enhanced chatbot responses
python AIRuleBasedFraudDetection.py path/to/your/document.csv --chat --openai-key YOUR_API_KEY
```

**Available Arguments:**
- `--method`: Detection method (`rules`, `nlp`, `autoencoder`, `all`) - default: `all`
- `--threshold`: Amount threshold for large transaction detection (default: 2000)
- `--nlp-contamination`: Expected fraction of outliers for NLP method (default: 0.1)
- `--autoencoder-threshold`: Autoencoder anomaly threshold percentile (default: 95)
- `--chat`: Enable interactive chat mode after analysis
- `--openai-key`: OpenAI API key for enhanced chatbot functionality
- `--save-model`: Path to save trained autoencoder model
- `--load-model`: Path to load pretrained autoencoder model
- `--output`: Output file path for analysis report
- `--debug`: Enable detailed debug output

**Example Chat Session:**
```
🤖 === Interactive AI Assistant Mode ===
💬 You: What are the main fraud risks in this dataset?
🤖 AI Assistant: Based on the analysis, I found 3 high-risk transactions flagged by rule-based detection, 
primarily due to large amounts and unusual transaction hours. The NLP analysis also identified 2 
transactions with suspicious keywords like "cash advance" and "foreign transaction"...

💬 You: Which transactions should I investigate first?
🤖 AI Assistant: I recommend prioritizing these transactions for investigation:
1. Transaction TX0045 - $5,200 at 3:14 AM (high amount + unusual hour)
2. Transaction TX0167 - Contains "bitcoin purchase" keyword
...
```

### Web Interface

To start the web application:

```
python app.py
```

Then open a web browser and navigate to `http://127.0.0.1:5000/`

The web interface allows you to:
1. Upload any bank statement document
2. Set custom fraud detection parameters
3. View detailed analysis results and visualizations
4. Download analysis reports and charts

## Supported File Formats

The system can process virtually any type of financial document including:

### Structured Data
- CSV files (with any delimiter)
- Excel spreadsheets (XLSX, XLS, XLSM, XLSB)
- JSON and XML data files
- HTML tables

### Documents
- PDF bank statements
- Word documents (DOCX, DOC)
- Rich Text Format (RTF)
- Text files

### Images
- Scanned bank statements (JPG, PNG, TIFF, BMP)
- Photos of statements taken with smartphones

### Other
- Email statements (EML, MSG)
- Archive files (ZIP, TAR, GZ, 7Z) containing statements

## How It Works

### 1. Document Processing

The system uses a multi-layered approach to process documents:

1. **File Type Detection**: Identifies the file format using extensions, MIME types, and content analysis
2. **Content Extraction**: Extracts text and tabular data using format-specific processors
3. **Field Identification**: AI algorithms identify banking-related fields (dates, amounts, descriptions)
4. **Transaction Mapping**: Maps extracted data to standardized transaction fields

### 2. Fraud Detection

Multiple AI-powered fraud detection techniques are applied:

1. **Rule-Based Detection**: Applies predefined rules like:
   - Large transaction amounts
   - Unusual transaction hours
   - High transaction frequency
   - Geographical velocity anomalies
   - Unusual penny amounts
   - New merchant categories

2. **NLP-Based Detection**: 
   - **Text Analysis**: Uses TF-IDF vectorization of transaction descriptions
   - **Clustering**: DBSCAN clustering to identify transaction patterns
   - **Anomaly Detection**: Isolation Forest for detecting unusual text patterns
   - **Keyword Analysis**: Identifies suspicious keywords and phrases
   - **Feature Engineering**: Combines text and numerical features

3. **Deep Learning Detection**:
   - **Autoencoder Neural Networks**: Unsupervised learning to detect anomalies
   - **Feature Learning**: Automatically learns relevant patterns from data
   - **Reconstruction Error**: Measures how well normal patterns can be reconstructed
   - **Threshold Tuning**: Adaptive anomaly thresholds based on training data

4. **Ensemble Analysis**:
   - **Method Comparison**: Compares results across all detection methods
   - **Consensus Scoring**: Identifies transactions flagged by multiple methods
   - **Confidence Assessment**: Provides confidence levels for each detection

### 3. Analysis & Reporting

The system provides comprehensive analysis:

1. **Risk Scoring**: Each transaction receives a risk score and classification
2. **Visualization**: Charts show risk distribution and top fraud indicators
3. **Method Comparison**: Side-by-side comparison of all detection methods
4. **Detailed Reports**: Complete transaction analysis available as CSV

### 4. AI Assistant & Chatbot

The integrated AI assistant provides interactive analysis:

1. **Contextual Understanding**: Automatically understands your fraud detection results
2. **Natural Language Interface**: Ask questions in plain English about your data
3. **Intelligent Explanations**: Get detailed explanations of fraud patterns and anomalies
4. **Actionable Recommendations**: Receive specific advice on investigation priorities
5. **Learning Capability**: Uses OpenAI GPT models for advanced understanding (optional)
6. **Fallback Responses**: Works without internet connection using built-in knowledge

## Architecture

The project consists of five main components:

1. **AIDocumentProcessor**: Handles all document parsing and field detection
2. **EnhancedFraudDetectionSystem**: Implements rule-based fraud detection algorithms
3. **NLPFraudDetector**: Implements NLP and clustering-based anomaly detection
4. **AutoencoderFraudDetector**: Implements deep learning-based anomaly detection
5. **FraudDetectionChatbot**: Provides AI-powered conversational interface
6. **FraudDetectionApp**: Orchestrates the overall process and integrates all components

The web application (`app.py`) provides a Flask-based user interface that integrates these components with interactive features including real-time chatbot functionality.

## Dependencies

### Core Dependencies
- pandas, numpy: Data processing
- matplotlib: Visualization
- Flask: Web interface

### File Processing Dependencies
- PyPDF2, pdfplumber: PDF processing
- pytesseract, PIL: Image processing and OCR
- python-docx: Word document processing
- python-magic: Enhanced file type detection
- chardet: Character encoding detection

### Optional Dependencies
- BeautifulSoup: HTML parsing
- dateutil: Advanced date parsing
- py7zr: 7z archive support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Developed by Lester L. Artis Jr.
