# AI-Enhanced Bank Statement Fraud Detection

An intelligent system for analyzing bank statements and financial documents to detect potential fraud using advanced AI techniques.

## Overview

This project provides a complete end-to-end solution for fraud detection in financial documents. It can process virtually any type of bank statement or transaction data (CSV, Excel, PDF, images, etc.), automatically extract transaction information, and apply AI-enhanced fraud detection algorithms to identify suspicious activities.

## Features

- **Universal Document Processing**: Automatically handles any file format containing transaction data
- **Intelligent Field Detection**: AI automatically identifies and maps transaction fields
- **Advanced Fraud Detection**: Multiple rule-based and AI-powered detection algorithms
- **Visual Analysis**: Interactive charts and visualizations of fraud patterns
- **Detailed Explanations**: AI-generated natural language explanations of fraud findings
- **Web Interface**: User-friendly web application for uploading and analyzing documents
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

```
python AIRuleBasedFraudDetection.py path/to/your/document.csv
```

Optional arguments:
- `--threshold`: Amount threshold for large transaction detection (default: 2000)
- `--output`: Output file path for analysis report (default: fraud_analysis_report.csv)
- `--no-viz`: Disable visualization generation
- `--debug`: Enable detailed debug output

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

### Real-time Transaction Monitoring

To start real-time transaction monitoring:

```
python realtime_transaction_tracker.py
```

This component provides continuous monitoring of transaction patterns and real-time fraud detection capabilities.

<<<<<<< HEAD
### Real-time Transaction Monitoring

To start real-time transaction monitoring:

```
python realtime_transaction_tracker.py
```

This component provides continuous monitoring of transaction patterns and real-time fraud detection capabilities.

=======
>>>>>>> 8537c05c (Create README.md)
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

Multiple fraud detection techniques are applied:

1. **Rule-Based Detection**: Applies predefined rules like:
   - Large transaction amounts
   - Unusual transaction hours
   - High transaction frequency
   - Geographical velocity anomalies
   - Unusual penny amounts
   - New merchant categories

2. **AI-Enhanced Detection**:
   - Unusual amount pattern detection using statistical methods
   - Suspicious transaction sequence identification
   - Description anomaly detection using NLP techniques

### 3. Analysis & Reporting

The system provides comprehensive analysis:

1. **Risk Scoring**: Each transaction receives a risk score and classification
2. **Visualization**: Charts show risk distribution and top fraud indicators
3. **AI Explanation**: Natural language explanation of identified patterns
4. **Detailed Reports**: Complete transaction analysis available as CSV

## Architecture

The project consists of three main components:

1. **AIDocumentProcessor**: Handles all document parsing and field detection
2. **EnhancedFraudDetectionSystem**: Implements fraud detection algorithms
3. **FraudDetectionApp**: Orchestrates the overall process and provides interfaces

The web application (`app.py`) provides a Flask-based user interface that integrates these components.

## Dependencies

### Core Dependencies
- pandas, numpy: Data processing
- matplotlib: Visualization
- Flask: Web interface

### File Processing Dependencies
- PyPDF2>=3.0.0: PDF processing
- pdfplumber>=0.7.0: Advanced PDF parsing
- python-docx>=0.8.11: Word document processing
- pytesseract>=0.3.10: OCR text extraction
- Pillow>=9.0.0: Image processing
- python-magic>=0.4.27: File type detection
- chardet>=5.0.0: Character encoding detection

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
