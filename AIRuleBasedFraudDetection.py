"""
Enhanced Bank Statement Fraud Detection System
with improved file type handling and parsing capabilities

Author: Lester L. Artis Jr. 
Created: 03/15/2025
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings
from pathlib import Path
import json
import sys
import io
import mimetypes

# Optional dependencies - import with try/except
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PyPDF2 not found. PDF support disabled.")

try:
    import pytesseract
    from PIL import Image
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("Tesseract OCR dependencies not found. Image support disabled.")

try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    print("pdfplumber not found. Advanced PDF parsing disabled.")

# Try to import file type detection libraries
try:
    import magic
    MAGIC_SUPPORT = True
except ImportError:
    MAGIC_SUPPORT = False
    print("python-magic not found. Advanced file type detection disabled.")

# Try to import Word document support
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("python-docx not found. Word document support disabled.")

# Try to import XML support
try:
    import xml.etree.ElementTree as ET
    XML_SUPPORT = True
except ImportError:
    XML_SUPPORT = False
    print("XML support disabled.")

# Try to import character encoding detection
try:
    import chardet
    CHARDET_SUPPORT = True
except ImportError:
    CHARDET_SUPPORT = False
    print("chardet not found. Advanced text encoding detection disabled.")


class AIDocumentProcessor:
    """
    AI-powered processor for various document types that extracts transaction data
    and identifies banking information. Now with enhanced file type support.
    """
    
    def __init__(self):
        """Initialize the document processor with banking categories and patterns."""
        # Banking keyword categories for AI detection
        self.banking_categories = {
            'transaction_date': ['date', 'datetime', 'transaction date', 'post date', 'posting date'],
            'transaction_id': ['id', 'reference', 'confirmation', 'transaction id', 'ref number', 'confirmation number'],
            'amount': ['amount', 'sum', 'debit', 'credit', 'transaction amount', 'payment', 'deposit', 'withdrawal'],
            'balance': ['balance', 'current balance', 'available balance', 'ending balance', 'new balance'],
            'description': ['description', 'details', 'memo', 'narrative', 'payee', 'transaction description'],
            'category': ['category', 'transaction type', 'type', 'classification', 'merchant category'],
            'location': ['location', 'merchant location', 'place', 'address', 'merchant address']
        }

        # Common bank statement patterns
        self.statement_patterns = {
            'account_info': re.compile(r'(account\s*number|acct\s*#|account\s*#|Account Holder)[:.\s]*([^$\n]+)', re.IGNORECASE),
            'date_range': re.compile(r'(statement\s*period|from|statement\s*dates|Account Statement)[:.\s]*([^-]+)\s*-\s*([^$\n]+)', re.IGNORECASE),
            'balance': re.compile(r'(closing\s*balance|ending\s*balance|available\s*balance|current\s*balance|Balance)[:.\s]*[$£€]?(\d{1,3}(,\d{3})*\.\d{2})', re.IGNORECASE),
            'transaction_date': re.compile(r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})', re.IGNORECASE)
        }
        
        # Expanded supported file formats and their processor methods
        self.supported_formats = {
            # Standard formats
            '.csv': self.process_csv,
            '.tsv': self.process_csv,  # Tab-separated values
            '.xlsx': self.process_excel,
            '.xls': self.process_excel,
            '.xlsm': self.process_excel,  # Macro-enabled Excel
            '.xlsb': self.process_excel,  # Binary Excel
            '.pdf': self.process_pdf,
            
            # Image formats
            '.jpg': self.process_image,
            '.jpeg': self.process_image,
            '.png': self.process_image,
            '.gif': self.process_image,
            '.bmp': self.process_image,
            '.tiff': self.process_image,
            '.tif': self.process_image,
            
            # Text formats
            '.txt': self.process_text,
            '.text': self.process_text,
            '.md': self.process_text,
            '.rtf': self.process_text,  # Rich Text Format
            
            # Document formats
            '.docx': self.process_docx,
            '.doc': self.process_docx,  # Will try to convert
            
            # Data formats
            '.json': self.process_json,
            '.xml': self.process_xml,
            '.html': self.process_html,
            '.htm': self.process_html,
            
            # Archive formats (will extract and process contents)
            '.zip': self.process_archive,
            '.tar': self.process_archive,
            '.gz': self.process_archive,
            '.7z': self.process_archive,
            
            # Email formats
            '.eml': self.process_email,
            '.msg': self.process_email,
        }
        
        # MIME type to processor mapping
        self.mime_processors = {
            'text/csv': self.process_csv,
            'text/tab-separated-values': self.process_csv,
            'application/vnd.ms-excel': self.process_excel,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self.process_excel,
            'application/pdf': self.process_pdf,
            'image/jpeg': self.process_image,
            'image/png': self.process_image,
            'image/gif': self.process_image,
            'image/bmp': self.process_image,
            'image/tiff': self.process_image,
            'text/plain': self.process_text,
            'text/markdown': self.process_text,
            'text/rtf': self.process_text,
            'application/msword': self.process_docx,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self.process_docx,
            'application/json': self.process_json,
            'application/xml': self.process_xml,
            'text/xml': self.process_xml,
            'text/html': self.process_html,
            'application/zip': self.process_archive,
            'application/x-tar': self.process_archive,
            'application/gzip': self.process_archive,
            'application/x-7z-compressed': self.process_archive,
            'message/rfc822': self.process_email,
        }
        
        # Fallback processors in order of attempt
        self.fallback_processors = [
            self.process_csv,
            self.process_excel,
            self.process_text,
            self.process_json,
            self.process_xml,
            self.process_pdf,
            self.process_image
        ]
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type using multiple methods.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected MIME type
        """
        mime_type = None
        
        # 1. Try using file extension first (fastest)
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_type = mimetypes.guess_type(file_path)[0]
        
        # 2. If extension-based detection failed, try python-magic
        if not mime_type and MAGIC_SUPPORT:
            try:
                mime_type = magic.Magic(mime=True).from_file(file_path)
            except Exception as e:
                print(f"Magic-based file type detection failed: {str(e)}")
        
        # 3. If magic-based detection failed, try to open and analyze file content
        if not mime_type:
            try:
                # Try to read the first few bytes
                with open(file_path, 'rb') as f:
                    header = f.read(4096)
                
                # Check for common file signatures
                if header.startswith(b'%PDF'):
                    mime_type = 'application/pdf'
                elif header.startswith(b'PK\x03\x04'):
                    # ZIP-based formats (XLSX, DOCX, etc.)
                    mime_type = 'application/zip'
                elif header.startswith(b'\xFF\xD8\xFF'):
                    mime_type = 'image/jpeg'
                elif header.startswith(b'\x89PNG\r\n\x1A\n'):
                    mime_type = 'image/png'
                elif b'<?xml' in header:
                    mime_type = 'application/xml'
                elif b'<html' in header.lower() or b'<!doctype html' in header.lower():
                    mime_type = 'text/html'
                elif b'{' in header and b'}' in header:
                    # Possible JSON
                    try:
                        json.loads(header.decode('utf-8'))
                        mime_type = 'application/json'
                    except:
                        pass
                
                # If still nothing, try to detect if it's text
                if not mime_type and CHARDET_SUPPORT:
                    encoding = chardet.detect(header)
                    if encoding['confidence'] > 0.8:
                        # It's probably text
                        try:
                            decoded = header.decode(encoding['encoding'])
                            # Check for CSV-like content
                            if ',' in decoded and '\n' in decoded:
                                mime_type = 'text/csv'
                            else:
                                mime_type = 'text/plain'
                        except:
                            pass
            except Exception as e:
                print(f"Content-based file type detection failed: {str(e)}")
        
        # 4. If all detection methods failed, default to binary
        if not mime_type:
            mime_type = 'application/octet-stream'
            
        return mime_type
    
    def process_document(self, file_path: str, debug=False) -> Dict:
        """
        Process a document file and extract transaction data.
        Now with enhanced file type detection and fallback mechanisms.
        
        Args:
            file_path: Path to the document file
            debug: Whether to run debug extraction for PDFs
            
        Returns:
            dict: Document data including transactions and headers
        """
        print(f"Processing document: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Run debug mode for PDFs if requested
        if debug and file_path.lower().endswith('.pdf'):
            self.debug_pdf_extraction(file_path)
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Enhanced file type detection
        mime_type = self.detect_file_type(file_path)
        print(f"Detected MIME type: {mime_type}")
        
        try:
            print(f"Step 1/4: File type analysis...")
            # Try extension-based processor first
            if file_ext in self.supported_formats:
                processor = self.supported_formats[file_ext]
                processor_name = f"extension-based ({file_ext})"
            # If not found, try MIME type based processor
            elif mime_type in self.mime_processors:
                processor = self.mime_processors[mime_type]
                processor_name = f"MIME-based ({mime_type})"
            else:
                processor = None
                processor_name = None
            
            print(f"Step 2/4: Selecting processor method ({processor_name if processor_name else 'unknown'})...")
            
            # Process the document based on its type
            if processor:
                try:
                    print(f"Step 3/4: Document parsing...")
                    data = processor(file_path)
                    processing_method = processor_name
                except Exception as e:
                    print(f"Primary processor failed: {str(e)}. Trying fallback processors...")
                    # Try fallback processors if primary fails
                    data = self._try_fallback_processors(file_path)
                    processing_method = "fallback"
            else:
                print(f"Unsupported file type. Trying fallback processors...")
                data = self._try_fallback_processors(file_path)
                processing_method = "fallback"
            
            # If all processing methods failed
            if not data:
                raise ValueError(f"Could not process file: {file_path}. All processing methods failed.")
            
            print(f"Step 4/4: AI classification of fields...")
            # Identify banking categories in the data
            enhanced_data = self.identify_banking_categories(data)
            
            # Add processing method to output
            enhanced_data['processing_method'] = processing_method
            
            print(f"AI processing complete. Found {len(enhanced_data['data'])} transactions and {len(enhanced_data.get('category_mapping', {}))} categorized fields.")
            return enhanced_data
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise
    
    def _try_fallback_processors(self, file_path: str) -> Optional[Dict]:
        """Try multiple fallback processors until one succeeds."""
        for i, processor in enumerate(self.fallback_processors):
            try:
                print(f"  Trying fallback processor {i+1}/{len(self.fallback_processors)}...")
                data = processor(file_path)
                if data and data.get('data') and len(data.get('data')) > 0:
                    print(f"  Fallback processor {i+1} succeeded!")
                    return data
            except Exception as e:
                print(f"  Fallback processor {i+1} failed: {str(e)}")
        
        return None
    
    def process_csv(self, file_path: str) -> Dict:
        """
        Process a CSV-like file and extract transaction data.
        Now with enhanced CSV detection capabilities.
        """
        try:
            # First attempt: Try standard comma delimiter
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    return {
                        'data': df.to_dict('records'),
                        'headers': list(df.columns),
                        'format': 'csv',
                        'shape': df.shape
                    }
            except Exception as e:
                print(f"Standard CSV parsing failed: {str(e)}. Trying alternative delimiters...")
            
            # Second attempt: Try to determine delimiter
            with open(file_path, 'rb') as f:
                sample = f.read(4096)
                if CHARDET_SUPPORT:
                    encoding = chardet.detect(sample)['encoding'] or 'utf-8'
                else:
                    encoding = 'utf-8'
                
            sample_text = sample.decode(encoding, errors='replace')
            
            # Count potential delimiters
            delimiters = [',', ';', '\t', '|', ':']
            delimiter_counts = {d: sample_text.count(d) for d in delimiters}
            best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
            
            # Try with the best delimiter
            df = pd.read_csv(file_path, delimiter=best_delimiter)
            
            # Basic data validation
            if df.empty:
                raise ValueError("CSV file contains no data")
            
            # If we have only one column, it might be fixed width
            if len(df.columns) == 1:
                # Try fixed width
                try:
                    df = pd.read_fwf(file_path)
                    if len(df.columns) > 1:
                        return {
                            'data': df.to_dict('records'),
                            'headers': list(df.columns),
                            'format': 'fixed-width',
                            'shape': df.shape
                        }
                except:
                    pass
            
            return {
                'data': df.to_dict('records'),
                'headers': list(df.columns),
                'format': f'csv (delimiter: {best_delimiter})',
                'shape': df.shape
            }
        except Exception as e:
            raise ValueError(f"Error processing CSV-like file: {str(e)}")
    
    def process_excel(self, file_path: str) -> Dict:
        """Process an Excel file and extract transaction data."""
        try:
            # Try to read Excel file with pandas
            # Check if it has multiple sheets
            excel_file = pd.ExcelFile(file_path)
            
            if len(excel_file.sheet_names) > 1:
                print(f"Excel file has multiple sheets: {excel_file.sheet_names}")
                
                # Check all sheets and use the one with most data
                best_sheet = None
                max_rows = 0
                
                sheet_data = {}
                
                for sheet_name in excel_file.sheet_names:
                    try:
                        sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                        sheet_data[sheet_name] = {
                            'rows': len(sheet_df),
                            'df': sheet_df
                        }
                        
                        if len(sheet_df) > max_rows:
                            max_rows = len(sheet_df)
                            best_sheet = sheet_name
                    except Exception as e:
                        print(f"Error reading sheet {sheet_name}: {str(e)}")
                
                if best_sheet:
                    print(f"Using sheet with most data: {best_sheet} ({max_rows} rows)")
                    df = sheet_data[best_sheet]['df']
                    
                    # Add metadata about other sheets
                    metadata = {
                        'all_sheets': excel_file.sheet_names,
                        'selected_sheet': best_sheet,
                        'sheet_rows': {name: data['rows'] for name, data in sheet_data.items()}
                    }
                else:
                    raise ValueError("No valid data found in any sheet")
            else:
                # Just one sheet, read it directly
                df = pd.read_excel(file_path)
                metadata = {'sheet_name': excel_file.sheet_names[0]}
            
            # Basic data validation
            if df.empty:
                raise ValueError("Excel file contains no data")
            
            # Clean column names (Excel can have unnamed columns)
            df.columns = [f"Column{i}" if pd.isna(col) or col == '' else col 
                         for i, col in enumerate(df.columns)]
            
            # Look for potential header rows
            # Sometimes the first row is not the header but part of a title or description
            if len(df) > 1:
                # Check if the first row has different data types than the rest
                first_row_dtypes = [type(x) for x in df.iloc[0]]
                second_row_dtypes = [type(x) for x in df.iloc[1]]
                
                if first_row_dtypes != second_row_dtypes and all(isinstance(x, str) for x in df.iloc[0]):
                    # First row might be a header
                    # Try to use it as header
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = [str(x) if x else f"Column{i}" for i, x in enumerate(new_header)]
                    print("Detected and used first row as header")
            
            return {
                'data': df.to_dict('records'),
                'headers': list(df.columns),
                'format': 'excel',
                'shape': df.shape,
                'metadata': metadata
            }
        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
    
    def _preprocess_pdf_text(self, text):
        """Preprocess PDF text to handle formatting issues like split dates."""
        # Fix split dates (e.g., "2024-03-\n01" becomes "2024-03-01")
        fixed_text = re.sub(r'(\d{4}-\d{2}-)\s*\n(\d{2})', r'\1\2', text)
        
        # Reconstruct table rows that might be split
        lines = fixed_text.split('\n')
        reconstructed_lines = []
        current_line = ""
        
        for line in lines:
            # If line starts with a date fragment, it's likely a new row
            if re.match(r'^\d{4}-\d{2}-\d{2}', line):
                if current_line:
                    reconstructed_lines.append(current_line)
                current_line = line
            # If line starts with whitespace and previous line exists, it might be a continuation
            elif line.strip() and current_line:
                # Check if this line might contain amount or balance information
                if re.search(r'[$]?[\d,.]+', line):
                    current_line += " " + line.strip()
                else:
                    reconstructed_lines.append(current_line)
                    current_line = line
            elif line.strip():
                current_line = line
        
        if current_line:
            reconstructed_lines.append(current_line)
        
        return "\n".join(reconstructed_lines)
    
    def process_pdf(self, file_path: str) -> Dict:
        """Process a PDF file with advanced extraction techniques."""
        if not PDF_SUPPORT:
            raise ImportError("PDF support requires PyPDF2. Please install it with 'pip install PyPDF2'")
                
        try:
            # Always try pdfplumber first for better results
            if PDFPLUMBER_SUPPORT:
                with pdfplumber.open(file_path) as pdf:
                    # Extract metadata first
                    metadata = self._extract_pdf_metadata(pdf)
                    
                    # Try coordinate-based table extraction first
                    table_data = self._extract_tables_by_coordinates(pdf)
                    
                    if table_data and len(table_data) > 5:  # Ensure we found a reasonable number of rows
                        print(f"Successfully extracted {len(table_data)} transactions using coordinate-based extraction")
                        return {
                            'data': table_data,
                            'headers': list(table_data[0].keys()) if table_data else [],
                            'format': 'pdf-coordinates',
                            'metadata': metadata
                        }
                    
                    # If coordinate extraction failed, try standard table extraction
                    print("Coordinate-based extraction failed, trying standard table extraction")
                    tables = []
                    page_texts = []
                    
                    # Extract text and tables from all pages
                    for page in pdf.pages:
                        page_texts.append(page.extract_text())
                        extracted_tables = page.extract_tables()
                        if extracted_tables:
                            tables.extend(extracted_tables)
                    
                    # Combine all page texts
                    raw_text = "\n".join(page_texts)
                    
                    # Preprocess the text to handle formatting issues
                    processed_text = self._preprocess_pdf_text(raw_text)
                    
                    if tables:
                        # Process extracted tables from pdfplumber
                        return self._process_pdfplumber_tables(tables, processed_text)
                    else:
                        # No tables found, try text-based extraction with the processed text
                        return self._extract_bank_statement_structure(processed_text)
            
            # Fall back to PyPDF2 approach
            extracted_text = ""
            table_data = []
            
            # Extract text from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Process each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    extracted_text += page_text + "\n"
                    
                    # Look for tabular data with improved pattern matching
                    lines = page_text.split('\n')
                    for line in lines:
                        # Look for transaction date patterns and amounts
                        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', line)
                        amount_match = re.search(r'[-+]?\$?\d+\.\d{2}', line)
                        
                        if date_match and amount_match:
                            # Correctly parse the bank statement format
                            parts = self._parse_bank_statement_line(line)
                            if parts:
                                table_data.append(parts)
            
            # Preprocess the extracted text
            processed_text = self._preprocess_pdf_text(extracted_text)
            
            # Create structured data from extracted text
            if not table_data:
                # If table extraction failed, try specialized extraction
                return self._extract_bank_statement_structure(processed_text)
            
            # Process the extracted table data
            return self._process_extracted_table_data(table_data, processed_text)
                
        except Exception as e:
            print(f"Error in PDF processing: {str(e)}")
            # Add more detailed error information
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error processing PDF file: {str(e)}")
    
    def _extract_pdf_metadata(self, pdf):
        """Extract metadata from PDF file."""
        metadata = {}
        
        # Try to extract common metadata from the first page text
        first_page_text = pdf.pages[0].extract_text() if pdf.pages else ""
        
        # Extract account information
        account_match = re.search(r'Account\s*(Number|#|No\.?)[:.\s]*([A-Za-z0-9*_\-]+)', first_page_text, re.IGNORECASE)
        if account_match:
            metadata['account_number'] = account_match.group(2).strip()
        
        # Extract account holder
        holder_match = re.search(r'Account\s*Holder[:.\s]*([A-Za-z0-9\s]+)', first_page_text, re.IGNORECASE)
        if holder_match:
            metadata['account_holder'] = holder_match.group(1).strip()
        
        # Extract statement period
        period_match = re.search(r'(Statement|Period)[\s:]*([A-Za-z0-9,\s]+\s+to\s+[A-Za-z0-9,\s]+)', first_page_text, re.IGNORECASE)
        if period_match:
            metadata['statement_period'] = period_match.group(2).strip()
        
        # Add PDF document info if available
        if hasattr(pdf, 'metadata') and pdf.metadata:
            metadata['pdf_title'] = pdf.metadata.get('Title', '')
            metadata['pdf_author'] = pdf.metadata.get('Author', '')
            metadata['pdf_subject'] = pdf.metadata.get('Subject', '')
            metadata['pdf_producer'] = pdf.metadata.get('Producer', '')
            metadata['pdf_creator'] = pdf.metadata.get('Creator', '')
        
        return metadata
    
    def _extract_tables_by_coordinates(self, pdf):
        """Extract table data using coordinate-based analysis."""
        transactions = []
        
        # Identify column boundaries by analyzing first few pages
        column_bounds = self._identify_column_bounds(pdf)
        if not column_bounds:
            print("Could not identify column boundaries")
            return []
        
        # Process each page
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num+1}/{len(pdf.pages)} using coordinate extraction")
            
            # Extract all text with coordinates
            texts = page.extract_words(x_tolerance=3, y_tolerance=3)
            
            # Group texts by row (y-coordinate)
            rows = self._group_texts_by_row(texts)
            
            # Extract data from each row using column boundaries
            for row_y, row_texts in rows.items():
                row_data = {}
                
                # Map texts to columns based on x-coordinates
                for text_obj in row_texts:
                    x0 = text_obj['x0']
                    text = text_obj['text']
                    
                    # Find which column this text belongs to
                    for col_name, (start_x, end_x) in column_bounds.items():
                        if start_x <= x0 <= end_x:
                            if col_name in row_data:
                                row_data[col_name] += " " + text
                            else:
                                row_data[col_name] = text
                            break
                
                # Only add row if it has enough data (e.g., date and amount at minimum)
                if 'Date' in row_data and ('Amount' in row_data or 'Balance' in row_data):
                    # Clean date format if needed
                    if 'Date' in row_data and re.match(r'^\d{4}-\d{2}-$', row_data['Date']):
                        # Look for the rest of the date in the Description
                        if 'Description' in row_data and row_data['Description'].startswith('\d{2}'):
                            match = re.match(r'^(\d{2})\s*(.*)', row_data['Description'])
                            if match:
                                row_data['Date'] += match.group(1)
                                row_data['Description'] = match.group(2)
                
                    transactions.append(row_data)
        
        return transactions
    
    def _identify_column_bounds(self, pdf):
        """Identify table column boundaries based on text positioning."""
        try:
            # Use the first page as reference for column identification
            page = pdf.pages[0]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            
            # Look for header row by finding common column headers
            header_candidates = []
            column_positions = {}
            
            # Common headers in bank statements
            common_headers = ['Date', 'Description', 'Category', 'Amount', 'Balance']
            
            for word in words:
                text = word['text']
                
                # Check if this word matches any common headers
                for header in common_headers:
                    if text.lower() == header.lower() or text.lower().startswith(header.lower()):
                        header_candidates.append({
                            'header': header,
                            'x0': word['x0'],
                            'x1': word['x1'],
                            'top': word['top'],
                            'bottom': word['bottom']
                        })
            
            # If we don't find at least 3 headers, look for date patterns to identify transaction rows
            if len(header_candidates) < 3:
                date_positions = []
                amount_positions = []
                
                # Find patterns that look like dates and amounts
                for word in words:
                    if re.match(r'\d{4}-\d{2}-\d{2}', word['text']) or re.match(r'\d{2}/\d{2}/\d{4}', word['text']):
                        date_positions.append(word['x0'])
                    elif re.match(r'[$]?[\d,.]+\.\d{2}', word['text']):
                        amount_positions.append(word['x0'])
                
                # If we find consistent positions, use them to define columns
                if date_positions and amount_positions:
                    avg_date_pos = sum(date_positions) / len(date_positions)
                    avg_amount_pos = sum(amount_positions) / len(amount_positions)
                    
                    column_positions = {
                        'Date': (avg_date_pos - 10, avg_date_pos + 50),
                        'Description': (avg_date_pos + 50, avg_amount_pos - 10),
                        'Amount': (avg_amount_pos - 10, avg_amount_pos + 50),
                        'Balance': (avg_amount_pos + 50, page.width)
                    }
                    
                    return column_positions
            
            # Sort headers by x-position to determine column order
            header_candidates.sort(key=lambda h: h['x0'])
            
            # If we found headers, define column boundaries
            if header_candidates:
                for i, header in enumerate(header_candidates):
                    if i < len(header_candidates) - 1:
                        column_positions[header['header']] = (header['x0'], header_candidates[i+1]['x0'])
                    else:
                        column_positions[header['header']] = (header['x0'], page.width)
                
                # Add a reasonable start for the first column
                if header_candidates:
                    first_header = header_candidates[0]['header']
                    column_positions[first_header] = (0, column_positions[first_header][1])
                    
                return column_positions
            
            return None
        except Exception as e:
            print(f"Error identifying column bounds: {str(e)}")
            return None
    
    def _group_texts_by_row(self, texts, tolerance=5):
        """Group text objects by their y-coordinate to identify rows."""
        rows = {}
        
        for text in texts:
            y_pos = text['top']
            
            # Find the closest existing row
            closest_row = None
            min_distance = tolerance + 1
            
            for row_y in rows.keys():
                distance = abs(row_y - y_pos)
                if distance < min_distance and distance <= tolerance:
                    closest_row = row_y
                    min_distance = distance
            
            if closest_row is not None:
                # Add to existing row
                rows[closest_row].append(text)
            else:
                # Create new row
                rows[y_pos] = [text]
        
        # Sort texts within each row by x-coordinate
        for row_y in rows:
            rows[row_y].sort(key=lambda t: t['x0'])
        
        return rows
    
    def _process_pdfplumber_tables(self, tables, full_text=""):
        """Process tables extracted by pdfplumber."""
        transactions = []
        headers = []
        
        # Identify headers from the first table
        if tables and tables[0]:
            potential_headers = tables[0][0]
            # Clean and normalize headers
            headers = [
                str(header).strip().lower() 
                for header in potential_headers if header is not None
            ]
            
            # Process each table row as a transaction
            for table in tables:
                for row_idx, row in enumerate(table):
                    # Skip the header row
                    if row_idx == 0 and table == tables[0]:
                        continue
                        
                    # Skip empty rows
                    if not any(cell for cell in row):
                        continue
                        
                    # Create transaction dictionary
                    transaction = {}
                    for col_idx, value in enumerate(row):
                        if col_idx < len(headers) and value is not None:
                            # Clean the value
                            clean_value = str(value).strip()
                            if clean_value:
                                transaction[headers[col_idx]] = clean_value
                    
                    # Only add non-empty transactions
                    if transaction:
                        transactions.append(transaction)
        
        # Extract metadata from full text
        metadata = {}
        if full_text:
            # Try to extract account information
            account_match = re.search(r'Account\s*(Number|#|No\.?)[:.\s]*([A-Za-z0-9*_\-]+)', full_text, re.IGNORECASE)
            if account_match:
                metadata['account_number'] = account_match.group(2).strip()
            
            # Try to extract statement period
            period_match = re.search(r'(Statement|Period)[\s:]*([A-Za-z0-9,\s]+\s+to\s+[A-Za-z0-9,\s]+)', full_text, re.IGNORECASE)
            if period_match:
                metadata['statement_period'] = period_match.group(2).strip()
            
            # Try to extract customer name
            name_match = re.search(r'(Name|Customer|Account\s*Holder)[\s:]*([A-Za-z\s,\.]+)', full_text, re.IGNORECASE)
            if name_match:
                metadata['customer_name'] = name_match.group(2).strip()
        
        return {
            'data': transactions,
            'headers': headers,
            'format': 'pdf',
            'metadata': metadata
        }

    def _parse_bank_statement_line(self, line):
        """Parse a transaction line from bank statement with enhanced pattern matching."""
        # Enhanced pattern matching for various date formats
        date_patterns = [
            # Standard formats
            r'(\d{4}-\d{2}-\d{2})\s+(.*?)\s+([A-Za-z &]+)\s+([-+]?\$?\d+\.\d{2})\s+(\$?\d+\.\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})\s+(.*?)\s+([A-Za-z &]+)\s+([-+]?\$?\d+\.\d{2})\s+(\$?\d+\.\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})\s+(.*?)\s+([-+]?\$?\d+\.\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\s+(.*?)\s+([-+]?\$?\d+\.\d{2})',
            
            # Handle split date formats
            r'(\d{4}-\d{2}-\d{2})\s+(.*?)\s+([A-Za-z &]+)\s+([-+]?\$?\d+\.\d{2})',
            r'(\d{4}-\d{2}-\d{2})(.*?)([-+]?\$?\d+\.\d{2})',
            
            # Very flexible pattern just looking for date and amount
            r'.*?(\d{4}-\d{2}-\d{2}).*?([-+]?\$?\d+\.\d{2}).*?(\$?\d+\.\d{2})?'
        ]
        
        # Try each pattern
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                result = {}
                
                # Always extract date
                result['date'] = groups[0]
                
                # Extract other fields based on available groups
                if len(groups) >= 3:
                    # For formats with description and amount
                    result['description'] = groups[1].strip()
                    if len(groups) >= 4:
                        # Format with category
                        result['category'] = groups[2].strip()
                        result['amount'] = groups[3]
                        if len(groups) >= 5:
                            # Format with balance
                            result['balance'] = groups[4]
                    else:
                        # Format without category
                        result['amount'] = groups[2]
                
                return result
        
        # If no standard pattern matches, try more flexible pattern matching
        # Look for any date and amount on the same line
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})', line)
        amount_match = re.search(r'([-+]?\$?\d+\.\d{2})', line)
        
        if date_match and amount_match:
            date = date_match.group(1)
            amount = amount_match.group(1)
            
            # Extract description as everything between date and amount
            date_end = date_match.end()
            amount_start = amount_match.start()
            
            if amount_start > date_end:
                description = line[date_end:amount_start].strip()
            else:
                # Try to get everything that's not the date or amount
                description = line.replace(date, "").replace(amount, "").strip()
            
            # Look for potential balance after amount
            balance_match = re.search(r'(\$?\d+\.\d{2})', line[amount_match.end():])
            balance = balance_match.group(1) if balance_match else None
            
            result = {
                'date': date,
                'description': description,
                'amount': amount
            }
            
            if balance:
                result['balance'] = balance
            
            return result
        
        return None

    def _extract_bank_statement_structure(self, text):
        """Extract structured data from bank statement text."""
        # Get header info
        statement_match = re.search(r'(Account|Statement)\s*(Period|Date)?\s*:?\s*(.*)', text, re.IGNORECASE)
        statement_period = statement_match.group(3) if statement_match else ''
        
        account_holder_match = re.search(r'(Account\s*Holder|Customer|Name)\s*:?\s*(.*)', text, re.IGNORECASE)
        account_holder = account_holder_match.group(2) if account_holder_match else ''
        
        # Extract transactions using date pattern as separators
        transactions = []
        
        # Try different date patterns
        date_patterns = [
            re.compile(r'(\d{4}-\d{2}-\d{2})'),  # YYYY-MM-DD
            re.compile(r'(\d{1,2}/\d{1,2}/\d{4})'),  # MM/DD/YYYY or DD/MM/YYYY
            re.compile(r'(\d{1,2}/\d{1,2}/\d{2})')   # MM/DD/YY or DD/MM/YY
        ]
        
        # Find all potential transaction blocks
        lines = text.split('\n')
        for line in lines:
            # Check if line contains a date
            has_date = False
            for pattern in date_patterns:
                if pattern.search(line):
                    has_date = True
                    break
            
            if has_date:
                # Try to parse it as a transaction line
                parts = self._parse_bank_statement_line(line)
                if parts:
                    transactions.append(parts)
        
        # If no transactions found, try to extract tabular data
        if not transactions:
            # Look for patterns that might indicate a table structure
            table_start_idx = -1
            headers = []
            
            # Look for potential table headers
            header_patterns = [
                r'date.*amount.*balance',
                r'transaction.*date.*description',
                r'date.*description.*debit.*credit',
                r'date.*details.*amount'
            ]
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                for pattern in header_patterns:
                    if re.search(pattern, line_lower):
                        table_start_idx = i
                        # Try to split header line into columns
                        headers = [h.strip() for h in re.split(r'\s{2,}', line) if h.strip()]
                        break
                if table_start_idx != -1:
                    break
            
            # If we found a table header, parse rows below it
            if table_start_idx != -1 and headers:
                for i in range(table_start_idx + 1, len(lines)):
                    line = lines[i]
                    if not line.strip():
                        continue
                    
                    # Try to split by multiple spaces
                    values = [v.strip() for v in re.split(r'\s{2,}', line) if v.strip()]
                    
                    # Check if we have a reasonable number of values
                    if len(values) >= min(2, len(headers)) and len(values) <= len(headers) + 1:
                        row_data = {}
                        # Map values to headers
                        for j, value in enumerate(values):
                            if j < len(headers):
                                row_data[headers[j]] = value
                            else:
                                # Extra values go into the last header
                                row_data[headers[-1]] = row_data.get(headers[-1], '') + ' ' + value
                        
                        transactions.append(row_data)
        
        # Create common header names if possible
        common_headers = set()
        for tx in transactions:
            common_headers.update(tx.keys())
        
        return {
            'data': transactions,
            'headers': list(common_headers),
            'format': 'pdf',
            'metadata': {
                'statement_period': statement_period,
                'account_holder': account_holder
            }
        }

    def _process_extracted_table_data(self, table_data, extracted_text):
        """Process the extracted table data with additional context."""
        # Get header information
        statement_match = re.search(r'Account Statement: (.*)', extracted_text)
        statement_period = statement_match.group(1) if statement_match else ''
        
        account_holder_match = re.search(r'Account Holder: (.*)', extracted_text)
        account_holder = account_holder_match.group(1) if account_holder_match else ''
        
        # Determine headers from the data structure
        headers = []
        if table_data and isinstance(table_data[0], dict):
            headers = list(table_data[0].keys())
        
        return {
            'data': table_data,
            'headers': headers,
            'format': 'pdf',
            'metadata': {
                'statement_period': statement_period,
                'account_holder': account_holder
            }
        }
    
    def debug_pdf_extraction(self, file_path: str):
        """Debug PDF extraction by showing detailed output at each step."""
        print("\n=== PDF Extraction Debugging ===")
        
        try:
            # Open with pdfplumber for raw analysis
            if PDFPLUMBER_SUPPORT:
                with pdfplumber.open(file_path) as pdf:
                    print(f"PDF Info: {len(pdf.pages)} pages")
                    
                    # Sample first page
                    first_page = pdf.pages[0]
                    
                    # Extract raw text
                    raw_text = first_page.extract_text()
                    print("\n=== Raw Text Sample (first 500 chars) ===")
                    print(raw_text[:500])
                    print("...")
                    
                    # Extract words with positions
                    words = first_page.extract_words()
                    print("\n=== Word Objects Sample (first 10) ===")
                    for i, word in enumerate(words[:10]):
                        print(f"{i}. '{word['text']}' at x0={word['x0']}, y0={word['top']}")
                    
                    # Try to extract tables
                    tables = first_page.extract_tables()
                    print(f"\n=== Tables Detected: {len(tables)} ===")
                    if tables:
                        print("First table sample (first 3 rows):")
                        for i, row in enumerate(tables[0][:3]):
                            print(f"Row {i}: {row}")
                    
                    # Try our preprocessing
                    processed_text = self._preprocess_pdf_text(raw_text)
                    print("\n=== Preprocessed Text Sample ===")
                    lines = processed_text.split("\n")
                    for i, line in enumerate(lines[:10]):
                        print(f"Line {i}: {line}")
                    
                    # Try parsing with our patterns
                    print("\n=== Pattern Matching Test ===")
                    for i, line in enumerate(lines[:10]):
                        result = self._parse_bank_statement_line(line)
                        if result:
                            print(f"Line {i} successfully parsed: {result}")
                        else:
                            print(f"Line {i} failed to parse: '{line}'")
                    
                    # Test coordinate extraction
                    print("\n=== Column Bounds Detection ===")
                    column_bounds = self._identify_column_bounds(pdf)
                    if column_bounds:
                        print("Detected columns:")
                        for col, (start, end) in column_bounds.items():
                            print(f"  {col}: {start:.1f} to {end:.1f}")
                    else:
                        print("Failed to detect column bounds")
            
            return "Debug complete"
        except Exception as e:
            print(f"Debug Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Debug failed: {str(e)}"
    
    def process_image(self, file_path: str) -> Dict:
        """Process an image file using OCR to extract transaction data."""
        if not OCR_SUPPORT:
            raise ImportError("Image processing requires pytesseract and PIL. Please install them with 'pip install pytesseract pillow'")
            
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Try to improve image quality for OCR
            try:
                # Convert to grayscale
                img = img.convert('L')
                
                # Optional: Apply some image enhancements
                from PIL import ImageEnhance, ImageFilter
                
                # Increase contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)
                
                # Apply sharpening filter
                img = img.filter(ImageFilter.SHARPEN)
                
                # Optional: Resize if image is very large
                if img.width > 3000 or img.height > 3000:
                    ratio = min(3000/img.width, 3000/img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
            except Exception as e:
                print(f"Image enhancement failed: {str(e)}")
            
            # Extract text using OCR with improved settings
            try:
                # Try to use more aggressive OCR settings
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                extracted_text = pytesseract.image_to_string(img, config=custom_config)
            except:
                # Fall back to default settings
                extracted_text = pytesseract.image_to_string(img)
            
            # Try to extract table structure
            try:
                # Use pytesseract's image_to_data to get positioning information
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                # Group text by lines based on top position
                lines = {}
                for i, text in enumerate(data['text']):
                    if not text.strip():
                        continue
                    
                    top = data['top'][i]
                    # Group within ±5 pixels as the same line
                    line_key = top // 5 * 5
                    if line_key not in lines:
                        lines[line_key] = []
                    
                    lines[line_key].append({
                        'text': text,
                        'left': data['left'][i],
                        'conf': data['conf'][i]
                    })
                
                # Sort each line by left position
                for key in lines:
                    lines[key].sort(key=lambda x: x['left'])
                
                # Convert to text lines
                text_lines = []
                for key in sorted(lines.keys()):
                    line_text = ' '.join(item['text'] for item in lines[key])
                    text_lines.append(line_text)
                
                # Join lines with newlines
                structured_text = '\n'.join(text_lines)
                
                # Check if structured extraction improved results
                if len(structured_text) > len(extracted_text) * 0.8:
                    extracted_text = structured_text
            except Exception as e:
                print(f"Structured OCR extraction failed: {str(e)}")
            
            # Extract transaction data similar to PDF approach
            transactions = self.extract_transactions_from_pdf(extracted_text, [])
            
            if not transactions:
                # Try more aggressive pattern matching
                date_amounts = []
                
                # Look for dates and amounts in the extracted text
                lines = extracted_text.split('\n')
                for line in lines:
                    # Look for date patterns
                    date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})', line)
                    amount_match = re.search(r'[$€£]?\s*(\d{1,3}(,\d{3})*\.\d{2})', line)
                    
                    if date_match and amount_match:
                        # Extract text between date and amount as description
                        date_end = date_match.end()
                        amount_start = amount_match.start()
                        
                        if date_end < amount_start:
                            description = line[date_end:amount_start].strip()
                            
                            date_amounts.append({
                                'date': date_match.group(0),
                                'description': description,
                                'amount': amount_match.group(0)
                            })
                
                if date_amounts:
                    transactions = date_amounts
            
            if not transactions:
                # If still no transactions found, use a more generic approach
                # Look for any numerical values that could be amounts
                amount_pattern = re.compile(r'[$€£]?\s*(\d{1,3}(,\d{3})*\.\d{2})')
                amount_matches = amount_pattern.finditer(extracted_text)
                
                generic_transactions = []
                for match in amount_matches:
                    # Get the line containing this amount
                    line_start = extracted_text.rfind('\n', 0, match.start()) + 1
                    line_end = extracted_text.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(extracted_text)
                    
                    line = extracted_text[line_start:line_end]
                    
                    # Create a simple transaction record
                    generic_transactions.append({
                        'line': line,
                        'amount': match.group(0)
                    })
                
                if generic_transactions:
                    transactions = generic_transactions
            
            if not transactions:
                # If no transactions found, return the raw OCR text for manual inspection
                return {
                    'data': [],
                    'headers': [],
                    'raw_text': extracted_text,
                    'format': 'image',
                    'warning': 'No structured data could be extracted. Raw OCR text provided.'
                }
            
            # Extract common headers from transactions
            common_headers = set()
            for tx in transactions:
                common_headers.update(tx.keys())
            
            headers = list(common_headers)
            
            return {
                'data': transactions,
                'headers': headers,
                'raw_text': extracted_text,
                'format': 'image'
            }
                
        except Exception as e:
            raise ValueError(f"Error processing image file: {str(e)}")
    
    def extract_transactions_from_pdf(self, text: str, table_data: List[List[str]]) -> List[Dict]:
        """Extract transactions from PDF text using patterns."""
        transactions = []
        
        # Try to identify statement header information
        account_match = self.statement_patterns['account_info'].search(text)
        account_number = account_match.group(2) if account_match else 'Unknown'
        
        # Update pattern to match various date formats
        date_range_match = re.search(r'(Account Statement|Statement Period|Period)[:.\s]*(.*?)(?:\s*-\s*|\s+to\s+)(.*)', text, re.IGNORECASE)
        statement_period = f"{date_range_match.group(2)} to {date_range_match.group(3)}" if date_range_match else ''
        
        # First try to use detected table data
        if table_data:
            for row in table_data:
                if isinstance(row, dict):
                    # Already parsed structured data
                    transactions.append(row)
                else:
                    # Process as array data
                    print(f"Warning: array data processing not implemented in extract_transactions_from_pdf")
        else:
            # Updated regex pattern for various date formats
            date_patterns = [
                re.compile(r'(\d{4}-\d{2}-\d{2})'),  # YYYY-MM-DD
                re.compile(r'(\d{1,2}/\d{1,2}/\d{4})'),  # MM/DD/YYYY or DD/MM/YYYY
                re.compile(r'(\d{1,2}/\d{1,2}/\d{2})')   # MM/DD/YY or DD/MM/YY
            ]
            
            amount_pattern = re.compile(r'([-+]?\$?\d{1,3}(,\d{3})*\.\d{2})')
            
            lines = text.split('\n')
            for line in lines:
                has_date = False
                for date_pattern in date_patterns:
                    if date_pattern.search(line):
                        has_date = True
                        break
                
                has_amount = amount_pattern.search(line)
                
                if has_date and has_amount:
                    # Process transaction line with specialized parsing
                    parsed = self._parse_bank_statement_line(line)
                    if parsed:
                        parsed['account_number'] = account_number
                        parsed['statement_period'] = statement_period
                        transactions.append(parsed)
        
        return transactions
    
    def process_text(self, file_path: str) -> Dict:
        """Process a text file and extract transaction data."""
        try:
            # Try to determine file encoding
            encoding = 'utf-8'
            if CHARDET_SUPPORT:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(4096)
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] or 'utf-8'
            
            # Read the text file
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
            
            # Try to determine if this is a CSV or TSV file first
            if text.count(',') > 10 or text.count('\t') > 10:
                try:
                    # Try CSV processing
                    return self.process_csv(file_path)
                except:
                    pass
            
            # Next, determine if this contains transaction-like data
            # Look for date and amount patterns
            has_dates = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}', text) is not None
            has_amounts = re.search(r'[$€£]?\d{1,3}(,\d{3})*\.\d{2}', text) is not None
            
            if has_dates and has_amounts:
                # Process as a bank statement, similar to PDF processing
                transactions = self.extract_transactions_from_pdf(text, [])
                
                if transactions:
                    headers = list(set().union(*[set(tx.keys()) for tx in transactions]))
                    return {
                        'data': transactions,
                        'headers': headers,
                        'format': 'text',
                        'raw_text': text[:1000] + ('...' if len(text) > 1000 else '')
                    }
            
            # If no structured data is found, try a line-by-line approach
            lines = text.split('\n')
            
            # Try to find a header line
            header_line = -1
            for i, line in enumerate(lines):
                # Check if line contains typical header keywords
                if re.search(r'date|description|amount|balance|transaction', line.lower()):
                    header_line = i
                    break
            
            if header_line >= 0:
                # Found potential header line
                headers = [h.strip() for h in re.split(r'\s{2,}|\t|,', lines[header_line]) if h.strip()]
                
                # Process data rows
                data_rows = []
                for i in range(header_line + 1, len(lines)):
                    line = lines[i].strip()
                    if not line:
                        continue
                    
                    # Split by same pattern as header
                    values = [v.strip() for v in re.split(r'\s{2,}|\t|,', line) if v.strip()]
                    
                    if len(values) >= 2:  # At least two columns of data
                        row_data = {}
                        for j, value in enumerate(values):
                            if j < len(headers):
                                row_data[headers[j]] = value
                            else:
                                # Extra values go to last column
                                last_header = headers[-1]
                                row_data[last_header] = row_data.get(last_header, '') + ' ' + value
                        
                        data_rows.append(row_data)
                
                if data_rows:
                    return {
                        'data': data_rows,
                        'headers': headers,
                        'format': 'text-table',
                        'raw_text': text[:1000] + ('...' if len(text) > 1000 else '')
                    }
            
            # Last resort - try to find chunks that might be transactions
            date_patterns = [
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'\d{4}-\d{2}-\d{2}'
            ]
            
            amount_patterns = [
                r'[$€£]?\d{1,3}(,\d{3})*\.\d{2}',
                r'\d+\.\d{2}'
            ]
            
            transaction_lines = []
            for line in lines:
                has_date = any(re.search(p, line) for p in date_patterns)
                has_amount = any(re.search(p, line) for p in amount_patterns)
                
                if has_date and has_amount:
                    transaction_lines.append(line)
            
            if transaction_lines:
                # Try to parse these lines as transactions
                transactions = []
                for line in transaction_lines:
                    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', line)
                    amount_match = re.search(r'([$€£]?\d{1,3}(,\d{3})*\.\d{2}|\d+\.\d{2})', line)
                    
                    if date_match and amount_match:
                        date_end = date_match.end()
                        amount_start = amount_match.start()
                        
                        if date_end < amount_start:
                            description = line[date_end:amount_start].strip()
                        else:
                            description = line.replace(date_match.group(), '').replace(amount_match.group(), '').strip()
                        
                        transactions.append({
                            'date': date_match.group(),
                            'description': description,
                            'amount': amount_match.group()
                        })
                
                if transactions:
                    return {
                        'data': transactions,
                        'headers': ['date', 'description', 'amount'],
                        'format': 'text-transactions',
                        'raw_text': text[:1000] + ('...' if len(text) > 1000 else '')
                    }
            
            # If we still can't find structured data, return the raw text
            return {
                'data': [{'line': line} for line in lines if line.strip()],
                'headers': ['line'],
                'format': 'raw-text',
                'raw_text': text[:1000] + ('...' if len(text) > 1000 else '')
            }
            
        except Exception as e:
            raise ValueError(f"Error processing text file: {str(e)}")
    
    def process_json(self, file_path: str) -> Dict:
        """Process a JSON file and extract transaction data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Already a list of records
                records = data
                
                # Check if records are flat or nested
                if records and isinstance(records[0], dict):
                    # Check for nested data structures
                    for key in records[0]:
                        if isinstance(records[0][key], list) and len(records[0][key]) > 0:
                            # Possibly nested transactions
                            nested_data = []
                            for record in records:
                                nested_records = record.get(key, [])
                                for nested_record in nested_records:
                                    # Add parent record info to nested record
                                    combined_record = {f"parent_{k}": v for k, v in record.items() if k != key}
                                    combined_record.update(nested_record)
                                    nested_data.append(combined_record)
                            
                            if nested_data:
                                records = nested_data
                                print(f"Found and processed nested data in '{key}' field")
            
            elif isinstance(data, dict):
                # Look for array fields that might contain transactions
                candidate_fields = []
                
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if items look like transactions
                        if isinstance(value[0], dict):
                            candidate_fields.append((key, len(value)))
                
                if candidate_fields:
                    # Sort by array size (largest first)
                    candidate_fields.sort(key=lambda x: x[1], reverse=True)
                    
                    # Use the largest array
                    key = candidate_fields[0][0]
                    records = data[key]
                    
                    # Add metadata from parent
                    metadata = {k: v for k, v in data.items() if k != key}
                    
                    # Add parent data to each record
                    for record in records:
                        for meta_key, meta_value in metadata.items():
                            if isinstance(meta_value, (str, int, float, bool)):
                                record[f"meta_{meta_key}"] = meta_value
                else:
                    # No arrays found, use the object itself as a record
                    records = [data]
            else:
                raise ValueError("JSON data is neither a list nor an object")
            
            # Extract headers
            if records:
                headers = list(set().union(*[set(record.keys()) for record in records]))
            else:
                headers = []
            
            return {
                'data': records,
                'headers': headers,
                'format': 'json',
                'shape': (len(records), len(headers)) if headers else (len(records), 0)
            }
            
        except Exception as e:
            raise ValueError(f"Error processing JSON file: {str(e)}")
    
    def process_xml(self, file_path: str) -> Dict:
        """Process an XML file and extract transaction data."""
        if not XML_SUPPORT:
            raise ImportError("XML support requires ElementTree. Please install it with 'pip install ElementTree'")
            
        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Look for repeating elements that might represent transactions
            transaction_elements = []
            
            # Helper function to find transaction elements
            def find_transaction_elements(element, path=""):
                children = list(element)
                
                # Skip if no children
                if not children:
                    return
                
                # Count each tag
                tag_counts = {}
                for child in children:
                    tag = child.tag
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Find tags with multiple occurrences
                for tag, count in tag_counts.items():
                    if count > 1:
                        # This might be a transaction element
                        new_path = f"{path}/{tag}" if path else tag
                        transaction_elements.append({
                            "path": new_path,
                            "tag": tag,
                            "count": count,
                            "parent": element
                        })
                
                # Recursively check children
                for child in children:
                    new_path = f"{path}/{child.tag}" if path else child.tag
                    find_transaction_elements(child, new_path)
            
            # Find potential transaction elements
            find_transaction_elements(root)
            
            # Sort by count (highest first)
            transaction_elements.sort(key=lambda x: x["count"], reverse=True)
            
            if not transaction_elements:
                # No repeating elements found, try to extract data from all elements
                all_data = []
                
                def extract_all_data(element, parent_path=""):
                    data = {}
                    path = f"{parent_path}/{element.tag}" if parent_path else element.tag
                    
                    # Add attributes
                    for key, value in element.attrib.items():
                        data[f"{path}@{key}"] = value
                    
                    # Add text content if any
                    if element.text and element.text.strip():
                        data[f"{path}#text"] = element.text.strip()
                    
                    # Add children
                    for child in element:
                        child_data = extract_all_data(child, path)
                        data.update(child_data)
                    
                    return data
                
                root_data = extract_all_data(root)
                if root_data:
                    all_data.append(root_data)
                
                return {
                    'data': all_data,
                    'headers': list(all_data[0].keys()) if all_data else [],
                    'format': 'xml-flat',
                    'warning': 'No repeating elements found. Extracted flat structure.'
                }
            
            # Process the most frequent repeating element
            best_candidate = transaction_elements[0]
            transaction_tag = best_candidate["tag"]
            parent = best_candidate["parent"]
            
            # Extract data from each transaction element
            transactions = []
            
            for tx_element in parent.findall(f"./{transaction_tag}"):
                transaction = {}
                
                # Add attributes from transaction element
                for key, value in tx_element.attrib.items():
                    transaction[f"{transaction_tag}@{key}"] = value
                
                # Add child elements
                for child in tx_element:
                    # Add attributes
                    for key, value in child.attrib.items():
                        transaction[f"{child.tag}@{key}"] = value
                    
                    # Add text content
                    if child.text and child.text.strip():
                        transaction[child.tag] = child.text.strip()
                    
                    # Add grandchild elements (if any)
                    for grandchild in child:
                        # Combine tag name
                        combined_tag = f"{child.tag}.{grandchild.tag}"
                        
                        # Add attributes
                        for key, value in grandchild.attrib.items():
                            transaction[f"{combined_tag}@{key}"] = value
                        
                        # Add text content
                        if grandchild.text and grandchild.text.strip():
                            transaction[combined_tag] = grandchild.text.strip()
                
                transactions.append(transaction)
            
            # Extract headers
            if transactions:
                headers = list(set().union(*[set(tx.keys()) for tx in transactions]))
            else:
                headers = []
            
            return {
                'data': transactions,
                'headers': headers,
                'format': 'xml',
                'element_path': best_candidate["path"],
                'shape': (len(transactions), len(headers))
            }
            
        except Exception as e:
            raise ValueError(f"Error processing XML file: {str(e)}")
    
    def process_html(self, file_path: str) -> Dict:
        """Process an HTML file and extract transaction data."""
        try:
            from bs4 import BeautifulSoup
            HTML_PARSER_SUPPORT = True
        except ImportError:
            HTML_PARSER_SUPPORT = False
            print("BeautifulSoup not found. Using basic HTML parsing.")
        
        try:
            # Read HTML file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            if HTML_PARSER_SUPPORT:
                # Use BeautifulSoup for parsing
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Look for tables
                tables = soup.find_all('table')
                
                if tables:
                    # Process the largest table first
                    largest_table = max(tables, key=lambda t: len(t.find_all('tr')))
                    
                    # Extract headers
                    headers = []
                    header_row = largest_table.find('thead')
                    if not header_row:
                        header_row = largest_table.find('tr')
                    
                    if header_row:
                        headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                    
                    # Extract data rows
                    data_rows = []
                    for row in largest_table.find_all('tr')[1:]:  # Skip header row
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            row_data = {}
                            for i, cell in enumerate(cells):
                                if i < len(headers):
                                    header = headers[i]
                                else:
                                    header = f"Column{i}"
                                
                                row_data[header] = cell.get_text().strip()
                            
                            data_rows.append(row_data)
                    
                    return {
                        'data': data_rows,
                        'headers': headers,
                        'format': 'html-table',
                        'shape': (len(data_rows), len(headers))
                    }
                else:
                    # No tables found, try to extract text
                    # Remove script and style elements
                    for script in soup(['script', 'style']):
                        script.extract()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Process as text
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    
                    # Try to find transaction-like data in text
                    transaction_data = []
                    
                    for line in lines:
                        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', line)
                        amount_match = re.search(r'([$€£]?\d{1,3}(,\d{3})*\.\d{2})', line)
                        
                        if date_match and amount_match:
                            date_end = date_match.end()
                            amount_start = amount_match.start()
                            
                            if date_end < amount_start:
                                description = line[date_end:amount_start].strip()
                            else:
                                description = line.replace(date_match.group(0), '').replace(amount_match.group(0), '').strip()
                            
                            transaction_data.append({
                                'date': date_match.group(0),
                                'description': description,
                                'amount': amount_match.group(0)
                            })
                    
                    if transaction_data:
                        return {
                            'data': transaction_data,
                            'headers': ['date', 'description', 'amount'],
                            'format': 'html-extracted',
                            'shape': (len(transaction_data), 3)
                        }
                    else:
                        # No structured data found, return text data
                        return {
                            'data': [{'line': line} for line in lines],
                            'headers': ['line'],
                            'format': 'html-text',
                            'raw_text': '\n'.join(lines[:30]) + ('...' if len(lines) > 30 else '')
                        }
            else:
                # Basic HTML parsing without BeautifulSoup
                # Look for tables
                table_match = re.search(r'<table[^>]*>(.*?)</table>', html_content, re.DOTALL)
                
                if table_match:
                    table_content = table_match.group(1)
                    
                    # Extract rows
                    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL)
                    
                    if rows:
                        # Extract headers
                        header_match = re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.DOTALL)
                        if not header_match:
                            header_match = re.findall(r'<td[^>]*>(.*?)</td>', rows[0], re.DOTALL)
                        
                        headers = [re.sub(r'<[^>]*>', '', header).strip() for header in header_match]
                        
                        # Extract data rows
                        data_rows = []
                        for row in rows[1:]:  # Skip header row
                            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                            if cells:
                                row_data = {}
                                for i, cell in enumerate(cells):
                                    cell_text = re.sub(r'<[^>]*>', '', cell).strip()
                                    if i < len(headers):
                                        header = headers[i]
                                    else:
                                        header = f"Column{i}"
                                    
                                    row_data[header] = cell_text
                                
                                data_rows.append(row_data)
                        
                        return {
                            'data': data_rows,
                            'headers': headers,
                            'format': 'html-table-basic',
                            'shape': (len(data_rows), len(headers))
                        }
                
                # If no table or failed to parse, extract text
                # Remove HTML tags
                text = re.sub(r'<[^>]*>', ' ', html_content)
                # Normalize whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Process as text
                return self.process_text(io.StringIO(text))
                
        except Exception as e:
            raise ValueError(f"Error processing HTML file: {str(e)}")
    
    def process_docx(self, file_path: str) -> Dict:
        """Process a Word document and extract transaction data."""
        if not DOCX_SUPPORT:
            raise ImportError("Word document support requires python-docx. Please install it with 'pip install python-docx'")
            
        try:
            # Read DOCX file
            doc = docx.Document(file_path)
            
            # Extract full text
            full_text = '\n'.join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
            
            # Check if tables exist
            if doc.tables:
                # Process the largest table first
                largest_table = max(doc.tables, key=lambda t: len(t.rows))
                
                # Extract headers (first row)
                headers = []
                if largest_table.rows:
                    headers = [cell.text.strip() for cell in largest_table.rows[0].cells if cell.text.strip()]
                
                # Extract data rows
                data_rows = []
                for row in largest_table.rows[1:]:  # Skip header row
                    row_data = {}
                    
                    for i, cell in enumerate(row.cells):
                        if i < len(headers):
                            header = headers[i]
                        else:
                            header = f"Column{i}"
                        
                        row_data[header] = cell.text.strip()
                    
                    data_rows.append(row_data)
                
                return {
                    'data': data_rows,
                    'headers': headers,
                    'format': 'docx-table',
                    'raw_text': full_text[:1000] + ('...' if len(full_text) > 1000 else ''),
                    'shape': (len(data_rows), len(headers))
                }
            else:
                # No tables, process as text
                # Try to find transaction-like data in text
                lines = full_text.split('\n')
                transaction_data = []
                
                for line in lines:
                    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', line)
                    amount_match = re.search(r'([$€£]?\d{1,3}(,\d{3})*\.\d{2})', line)
                    
                    if date_match and amount_match:
                        date_end = date_match.end()
                        amount_start = amount_match.start()
                        
                        if date_end < amount_start:
                            description = line[date_end:amount_start].strip()
                        else:
                            description = line.replace(date_match.group(0), '').replace(amount_match.group(0), '').strip()
                        
                        transaction_data.append({
                            'date': date_match.group(0),
                            'description': description,
                            'amount': amount_match.group(0)
                        })
                
                if transaction_data:
                    return {
                        'data': transaction_data,
                        'headers': ['date', 'description', 'amount'],
                        'format': 'docx-extracted',
                        'raw_text': full_text[:1000] + ('...' if len(full_text) > 1000 else '')
                    }
                else:
                    # No structured data found, return text data
                    return {
                        'data': [{'line': line} for line in lines if line.strip()],
                        'headers': ['line'],
                        'format': 'docx-text',
                        'raw_text': full_text[:1000] + ('...' if len(full_text) > 1000 else '')
                    }
            
        except Exception as e:
            raise ValueError(f"Error processing Word document: {str(e)}")
    
    def process_email(self, file_path: str) -> Dict:
        """Process an email file and extract transaction data."""
        try:
            import email
            EMAIL_SUPPORT = True
        except ImportError:
            EMAIL_SUPPORT = False
            print("Email parsing requires the email module.")
            
        try:
            if not EMAIL_SUPPORT:
                raise ImportError("Email parsing requires the email module.")
                
            # Read email file
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f)
            
            # Extract email metadata
            metadata = {
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'subject': msg.get('Subject', ''),
                'date': msg.get('Date', '')
            }
            
            # Extract email body
            body = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = part.get("Content-Disposition", "")
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Get text content
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            body += payload.decode(charset, errors='replace')
                        except:
                            body += payload.decode('utf-8', errors='replace')
                    
                    elif content_type == "text/html":
                        if not body:  # Only use HTML if no plain text
                            payload = part.get_payload(decode=True)
                            charset = part.get_content_charset() or 'utf-8'
                            try:
                                html = payload.decode(charset, errors='replace')
                                # Convert HTML to text
                                try:
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(html, 'html.parser')
                                    body += soup.get_text()
                                except ImportError:
                                    # Simple HTML to text conversion
                                    body += re.sub(r'<[^>]*>', ' ', html)
                                    body = re.sub(r'\s+', ' ', body).strip()
                            except:
                                pass
            else:
                # Not multipart
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    body = payload.decode(charset, errors='replace')
                except:
                    body = payload.decode('utf-8', errors='replace')
                
                # If HTML, convert to text
                if msg.get_content_type() == "text/html":
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(body, 'html.parser')
                        body = soup.get_text()
                    except ImportError:
                        # Simple HTML to text conversion
                        body = re.sub(r'<[^>]*>', ' ', body)
                        body = re.sub(r'\s+', ' ', body).strip()
            
            # Look for transaction data in the body
            lines = body.split('\n')
            transaction_data = []
            
            for line in lines:
                date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', line)
                amount_match = re.search(r'([$€£]?\d{1,3}(,\d{3})*\.\d{2})', line)
                
                if date_match and amount_match:
                    date_end = date_match.end()
                    amount_start = amount_match.start()
                    
                    if date_end < amount_start:
                        description = line[date_end:amount_start].strip()
                    else:
                        description = line.replace(date_match.group(0), '').replace(amount_match.group(0), '').strip()
                    
                    transaction_data.append({
                        'date': date_match.group(0),
                        'description': description,
                        'amount': amount_match.group(0)
                    })
            
            if transaction_data:
                # Add email metadata to each transaction
                for tx in transaction_data:
                    tx.update({f"email_{k}": v for k, v in metadata.items()})
                
                return {
                    'data': transaction_data,
                    'headers': list(transaction_data[0].keys()),
                    'format': 'email',
                    'metadata': metadata,
                    'raw_text': body[:1000] + ('...' if len(body) > 1000 else '')
                }
            else:
                # No transaction data found, try to extract structural elements
                # Look for potential statement summary
                balance_match = re.search(r'(balance|total)[\s:]*[$€£]?\s*(\d{1,3}(,\d{3})*\.\d{2})', body, re.IGNORECASE)
                account_match = re.search(r'(account|acct)[\s:#]*([A-Za-z0-9*_\-]+)', body, re.IGNORECASE)
                
                if balance_match or account_match:
                    summary_data = [metadata.copy()]
                    
                    if balance_match:
                        summary_data[0]['balance_text'] = balance_match.group(0)
                        summary_data[0]['balance_amount'] = balance_match.group(2)
                    
                    if account_match:
                        summary_data[0]['account_info'] = account_match.group(0)
                        summary_data[0]['account_number'] = account_match.group(2)
                    
                    # Add email body
                    summary_data[0]['email_body'] = body[:1000] + ('...' if len(body) > 1000 else '')
                    
                    return {
                        'data': summary_data,
                        'headers': list(summary_data[0].keys()),
                        'format': 'email-summary',
                        'metadata': metadata,
                        'raw_text': body[:1000] + ('...' if len(body) > 1000 else '')
                    }
                else:
                    # Just return metadata and body
                    data_item = metadata.copy()
                    data_item['body'] = body
                    
                    return {
                        'data': [data_item],
                        'headers': list(data_item.keys()),
                        'format': 'email-raw',
                        'raw_text': body[:1000] + ('...' if len(body) > 1000 else '')
                    }
                
        except Exception as e:
            raise ValueError(f"Error processing email file: {str(e)}")
    
    def process_archive(self, file_path: str) -> Dict:
        """Process an archive file containing documents."""
        try:
            # Create a temporary directory to extract files
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine archive type and extract
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.zip':
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                elif file_ext in ['.tar', '.gz', '.bz2']:
                    import tarfile
                    with tarfile.open(file_path) as tar_ref:
                        tar_ref.extractall(temp_dir)
                elif file_ext == '.7z':
                    # Try to use py7zr if available
                    try:
                        import py7zr
                        with py7zr.SevenZipFile(file_path, mode='r') as z:
                            z.extractall(temp_dir)
                    except ImportError:
                        raise ImportError("7z archive support requires py7zr. Please install it with 'pip install py7zr'")
                else:
                    raise ValueError(f"Unsupported archive format: {file_ext}")
                
                # Look for files in the extracted directory
                extracted_files = []
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        extracted_files.append(os.path.join(root, file))
                
                if not extracted_files:
                    raise ValueError("No files found in the archive.")
                
                # Sort files by extension preference
                # Preference order: CSV, Excel, PDF, then others
                def extension_priority(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.csv':
                        return 0
                    elif ext in ('.xlsx', '.xls'):
                        return 1
                    elif ext == '.pdf':
                        return 2
                    else:
                        return 3
                
                extracted_files.sort(key=extension_priority)
                
                # Try to process each file until one succeeds
                for file in extracted_files:
                    try:
                        result = self.process_document(file)
                        
                        # Add archive metadata
                        result['archive_source'] = os.path.basename(file_path)
                        result['extracted_file'] = os.path.basename(file)
                        result['all_extracted_files'] = [os.path.basename(f) for f in extracted_files]
                        
                        return result
                    except Exception as e:
                        print(f"Failed to process extracted file {file}: {str(e)}")
                
                # If all files fail, raise an error
                raise ValueError("Failed to process any files in the archive.")
                        
        except Exception as e:
            raise ValueError(f"Error processing archive file: {str(e)}")
    
    def guess_table_headers(self, first_row: List[str]) -> List[str]:
        """Try to determine what each column represents."""
        headers = []
        
        for cell in first_row:
            cell_lower = str(cell).lower()
            
            if re.search(r'date|time', cell_lower):
                headers.append('date')
            elif re.search(r'desc|detail|narr|memo', cell_lower):
                headers.append('description')
            elif re.search(r'amount|sum|payment|deposit|withdrawal', cell_lower):
                headers.append('amount')
            elif re.search(r'balance', cell_lower):
                headers.append('balance')
            elif re.search(r'ref|id|number', cell_lower):
                headers.append('reference')
            elif re.search(r'type|category', cell_lower):
                headers.append('category')
            else:
                headers.append(cell_lower or f"column{len(headers)}")
        
        return headers
    
    def identify_banking_categories(self, document_data: Dict) -> Dict:
        """Identify banking categories in data."""
        data = document_data['data']
        headers = document_data['headers']
        
        # Map of confident category matches
        category_mapping = {}
        
        # Score each header against our banking categories
        for header in headers:
            header_lower = str(header).lower()
            
            # Find the best category match
            best_category = None
            best_score = 0
            
            for category, keywords in self.banking_categories.items():
                # Calculate similarity score
                score = self.calculate_similarity_score(header_lower, keywords)
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_category = category
            
            if best_category:
                category_mapping[header] = {
                    'category': best_category,
                    'confidence': best_score,
                    'original': header
                }
        
        # For any missing essential categories, try to detect them from data values
        essential_categories = ['transaction_date', 'amount', 'description']
        missing_categories = [
            category for category in essential_categories 
            if not any(mapping.get('category') == category for mapping in category_mapping.values())
        ]
        
        if missing_categories and data:
            for category in missing_categories:
                # Find headers that weren't mapped yet
                unmapped_headers = [
                    header for header in headers 
                    if header not in category_mapping
                ]
                
                for header in unmapped_headers:
                    # Check if values match expected patterns for this category
                    if self.values_match_category(data, header, category):
                        category_mapping[header] = {
                            'category': category,
                            'confidence': 0.7,  # Medium confidence since it's based on values
                            'original': header
                        }
                        break  # Found a match for this category
        
        # Add category mapping to document data
        document_data['category_mapping'] = category_mapping
        document_data['ai_enhanced'] = True
        
        return document_data
    
    def calculate_similarity_score(self, header: str, keywords: List[str]) -> float:
        """Calculate similarity score between a header and category keywords."""
        # Exact match
        if header in keywords:
            return 1.0
        
        # Partial matches
        best_score = 0
        
        for keyword in keywords:
            # Check if header contains keyword
            if keyword in header:
                score = len(keyword) / len(header)
                best_score = max(best_score, score)
            # Check if keyword contains header
            elif header in keyword:
                score = len(header) / len(keyword)
                best_score = max(best_score, score)
            # Check for word similarity
            else:
                header_words = header.split()
                keyword_words = keyword.split()
                
                matches = sum(1 for header_word in header_words 
                             if any(kw in header_word or header_word in kw for kw in keyword_words))
                
                if matches > 0:
                    score = matches / max(len(header_words), len(keyword_words))
                    best_score = max(best_score, score)
        
        return best_score
    
    def values_match_category(self, data: List[Dict], header: str, category: str) -> bool:
        """Check if values in a column match expected patterns for a category."""
        # Sample a few rows
        sample_size = min(5, len(data))
        samples = [row.get(header) for row in data[:sample_size]]
        
        # Skip empty values
        valid_samples = [sample for sample in samples if sample]
        if not valid_samples:
            return False
        
        if category == 'transaction_date':
            # Check for date patterns
            return any(isinstance(sample, str) and 
                      re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}', sample)
                      for sample in valid_samples)
        
        elif category == 'amount':
            # Check for money patterns
            return any(
                (isinstance(sample, (int, float))) or
                (isinstance(sample, str) and 
                 re.search(r'^[$£€]?\d{1,3}(,\d{3})*(\.\d{2})?$', str(sample).strip()))
                for sample in valid_samples
            )
        
        elif category == 'description':
            # Check for text that's likely a description (longer text)
            return any(
                isinstance(sample, str) and 
                len(sample) > 10 and 
                len(sample.split()) >= 2
                for sample in valid_samples
            )
        
        return False


class EnhancedFraudDetectionSystem:
    """
    AI-enhanced fraud detection system that analyzes bank transactions
    for suspicious patterns and activities.
    """
    
    def __init__(self, rules=None):
        """Initialize with default or custom rules."""
        # Initialize with default rules if none provided
        self.rules = rules if rules else self.get_default_rules()
        
        # Add AI-enhanced rules
        self.add_ai_rules()
    
    def get_default_rules(self) -> List[Dict]:
        """Define default fraud detection rules."""
        return [
            {
                'id': 'large-amount',
                'name': 'Large Transaction Amount',
                'description': 'Flags transactions above a threshold amount',
                'evaluate': lambda tx, profile, history: 
                    float(tx.get('amount', 0)) > profile.get('large_amount_threshold', 3250),
                'risk_score': 6
            },
            {
                'id': 'unusual-hour',
                'name': 'Unusual Transaction Hour',
                'description': 'Flags transactions occurring during unusual hours (2AM-5AM)',
                'evaluate': lambda tx, profile, history:
                    self._get_hour(tx.get('timestamp')) in range(2, 5),
                'risk_score': 2
            },
            {
                'id': 'high-frequency',
                'name': 'High Transaction Frequency',
                'description': 'Flags when too many transactions occur in a short timeframe',
                'evaluate': lambda tx, profile, history:
                    len([
                        t for t in history 
                        if self._time_diff_hours(tx.get('timestamp'), t.get('timestamp')) <= 1
                    ]) >= 5 if history else False,
                'risk_score': 4
            },
            {
                'id': 'geo-velocity',
                'name': 'Geographical Velocity',
                'description': 'Flags transactions that occur in different locations in a short timeframe',
                'evaluate': lambda tx, profile, history:
                    self._check_geo_velocity(tx, history),
                'risk_score': 5
            },
            {
                'id': 'penny-amounts',
                'name': 'Unusual Penny Amounts',
                'description': 'Flags transactions with unusual cent values (often used in testing stolen cards)',
                'evaluate': lambda tx, profile, history:
                    self._parse_amount_cents(tx.get('amount', 0))
                    not in [0, 50, 95, 99],
                'risk_score': 1
            },
            {
                'id': 'new-merchant-category',
                'name': 'New Merchant Category',
                'description': 'Flags transactions in merchant categories the user hasn\'t used before',
                'evaluate': lambda tx, profile, history:
                    tx.get('merchantCategory') not in {
                        t.get('merchantCategory') for t in history
                    } if history else False,
                'risk_score': 2
            }
        ]
    
    def _parse_amount_cents(self, amount):
        """Parse amount to get cents value, handling different formats."""
        try:
            # Handle string amounts with currency symbols or commas
            if isinstance(amount, str):
                # Remove currency symbols and commas
                amount = re.sub(r'[$£€,]', '', amount)
                amount = float(amount)
            
            # Get cents part
            return int(round((float(amount) * 100) % 100))
        except:
            return 0
    
    def add_ai_rules(self):
        """Add AI-enhanced fraud detection rules."""
        ai_rules = [
            {
                'id': 'ai-amount-pattern',
                'name': 'AI: Unusual Amount Pattern',
                'description': 'Uses AI to detect unusual spending amounts compared to historical patterns',
                'evaluate': lambda tx, profile, history:
                    self._detect_unusual_amount(tx, history),
                'risk_score': 4,
                'is_ai_rule': True
            },
            {
                'id': 'ai-sequence-pattern',
                'name': 'AI: Suspicious Transaction Sequence',
                'description': 'Detects suspicious sequences of transactions that match known fraud patterns',
                'evaluate': lambda tx, profile, history:
                    self._detect_suspicious_sequence(tx, history),
                'risk_score': 5,
                'is_ai_rule': True
            },
            {
                'id': 'ai-description-anomaly',
                'name': 'AI: Description Anomaly',
                'description': 'Analyzes transaction descriptions for unusual patterns or known fraud indicators',
                'evaluate': lambda tx, profile, history:
                    self._detect_description_anomaly(tx),
                'risk_score': 3,
                'is_ai_rule': True
            }
        ]
        
        # Add AI rules to the existing rules
        self.rules.extend(ai_rules)
    
    def add_rule(self, rule: Dict):
        """Add a custom rule to the system."""
        required_keys = ['id', 'name', 'description', 'evaluate', 'risk_score']
        if not all(key in rule for key in required_keys):
            raise ValueError("Invalid rule format. Required keys: " + ", ".join(required_keys))
        
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a rule by ID."""
        self.rules = [rule for rule in self.rules if rule['id'] != rule_id]
    
    def analyze_transaction(self, transaction: Dict, user_profile: Dict = None, 
                           transaction_history: List[Dict] = None) -> Dict:
        """
        Analyze a transaction against all rules.
        
        Args:
            transaction: Transaction data
            user_profile: User profile data
            transaction_history: List of previous transactions
            
        Returns:
            dict: Analysis results with triggered rules and risk score
        """
        if user_profile is None:
            user_profile = {}
        
        if transaction_history is None:
            transaction_history = []
        
        triggered_rules = []
        total_risk_score = 0
        
        for rule in self.rules:
            try:
                is_triggered = rule['evaluate'](transaction, user_profile, transaction_history)
                
                if is_triggered:
                    triggered_rules.append({
                        'rule_id': rule['id'],
                        'rule_name': rule['name'],
                        'description': rule['description'],
                        'risk_score': rule['risk_score'],
                        'is_ai_rule': rule.get('is_ai_rule', False)
                    })
                    
                    total_risk_score += rule['risk_score']
            except Exception as e:
                print(f"Error evaluating rule {rule['id']}: {str(e)}")
        
        # Determine fraud likelihood based on risk score
        fraud_likelihood = 'Low'
        if total_risk_score >= 6:
            fraud_likelihood = 'High'
        elif total_risk_score >= 3:
            fraud_likelihood = 'Medium'
        
        return {
            'transaction_id': transaction.get('id', 'unknown'),
            'timestamp': transaction.get('timestamp'),
            'fraud_likelihood': fraud_likelihood,
            'risk_score': total_risk_score,
            'triggered_rules': triggered_rules,
            'requires_review': total_risk_score >= 5,
            'ai_enhanced': any(rule.get('is_ai_rule', False) for rule in triggered_rules)
        }
    
    def analyze_batch(self, transactions: List[Dict], user_profile: Dict = None) -> List[Dict]:
        """
        Analyze a batch of transactions.
        
        Args:
            transactions: List of transactions to analyze
            user_profile: User profile data
            
        Returns:
            list: List of analysis results
        """
        if user_profile is None:
            user_profile = {}
        
        results = []
        
        # Sort transactions by timestamp
        sorted_transactions = sorted(
            transactions,
            key=lambda tx: self._parse_timestamp(tx.get('timestamp', '1970-01-01'))
        )
        
        for i, transaction in enumerate(sorted_transactions):
            history = sorted_transactions[:i]
            result = self.analyze_transaction(transaction, user_profile, history)
            results.append(result)
        
        return results
    
    def generate_explanation(self, results: List[Dict], transactions: List[Dict]) -> str:
        """
        Generate a natural language explanation for the fraud analysis results.
        
        Args:
            results: Analysis results
            transactions: Original transactions
            
        Returns:
            str: Explanation text
        """
        # Count risk levels
        high_risk = sum(1 for r in results if r['fraud_likelihood'] == 'High')
        medium_risk = sum(1 for r in results if r['fraud_likelihood'] == 'Medium')
        low_risk = sum(1 for r in results if r['fraud_likelihood'] == 'Low')
        
        # Calculate percentages
        total = len(results)
        if total == 0:
            return "No transactions analyzed."
            
        high_risk_percent = (high_risk / total) * 100
        medium_risk_percent = (medium_risk / total) * 100
        
        # Get most commonly triggered rules
        rule_counts = {}
        for result in results:
            for rule in result['triggered_rules']:
                rule_name = rule['rule_name']
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate explanation
        explanation = f"## AI Fraud Analysis Summary\n\n"
        
        explanation += f"I analyzed {total} transactions and identified:\n\n"
        explanation += f"- **{high_risk}** high-risk transactions ({high_risk_percent:.1f}%)\n"
        explanation += f"- **{medium_risk}** medium-risk transactions ({medium_risk_percent:.1f}%)\n"
        explanation += f"- **{low_risk}** low-risk transactions\n\n"
        
        if top_rules:
            explanation += f"### Most Common Risk Factors\n\n"
            
            for rule, count in top_rules:
                percent = (count / total) * 100
                explanation += f"- **{rule}**: Triggered in {count} transactions ({percent:.1f}%)\n"
            
            explanation += f"\n"
        
        # Highlight specific high-risk transactions
        high_risk_transactions = [
            (result, transaction)
            for result, transaction in zip(results, transactions)
            if result['fraud_likelihood'] == 'High'
        ]
        
        high_risk_transactions.sort(key=lambda x: x[0]['risk_score'], reverse=True)
        top_high_risk = high_risk_transactions[:3]
        
        if top_high_risk:
            explanation += f"### Notable High-Risk Transactions\n\n"
            
            for result, transaction in top_high_risk:
                date = self._format_timestamp(transaction.get('timestamp', 'Unknown'))
                desc = transaction.get('description', 'Unknown')
                amount = transaction.get('amount', 0)
                
                if isinstance(amount, (int, float)):
                    amount_str = f"${amount:.2f}"
                else:
                    amount_str = amount
                
                explanation += f"- **{date}**: {desc} ({amount_str}) - Risk Score: {result['risk_score']}\n"
                explanation += f"  - Triggered rules: {', '.join(rule['rule_name'] for rule in result['triggered_rules'])}\n\n"
        
        explanation += f"### Risk Assessment\n\n"
        
        if high_risk > 0:
            explanation += f"The statement shows **{high_risk_percent:.1f}%** of transactions flagged as high-risk, which is "
            
            if high_risk / total > 0.15:
                explanation += "significantly higher than normal. This indicates a high probability of fraudulent activity that requires immediate investigation."
            elif high_risk / total > 0.05:
                explanation += "above average. A manual review of flagged transactions is recommended."
            else:
                explanation += "within normal parameters, but the flagged transactions should be reviewed."
        else:
            explanation += "No high-risk transactions were identified, which suggests normal account activity."
        
        return explanation
    
    # Helper methods
    def _get_hour(self, timestamp):
        """Extract hour from timestamp."""
        try:
            dt = self._parse_timestamp(timestamp)
            return dt.hour
        except:
            return -1  # Invalid hour
    
    def _parse_timestamp(self, timestamp):
        """Parse timestamp to datetime object."""
        if not timestamp:
            return datetime.min
            
        if isinstance(timestamp, datetime):
            return timestamp
            
        if isinstance(timestamp, str):
            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y",
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%d-%m-%Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
                    
            # If all formats fail, try dateutil
            try:
                from dateutil import parser
                return parser.parse(timestamp)
            except:
                return datetime.min
        
        return datetime.min
    
    def _format_timestamp(self, timestamp):
        """Format timestamp to readable date string."""
        dt = self._parse_timestamp(timestamp)
        if dt == datetime.min:
            return "Unknown"
        return dt.strftime("%m/%d/%Y")
    
    def _time_diff_hours(self, timestamp1, timestamp2):
        """Calculate time difference in hours between two timestamps."""
        dt1 = self._parse_timestamp(timestamp1)
        dt2 = self._parse_timestamp(timestamp2)
        
        if dt1 == datetime.min or dt2 == datetime.min:
            return float('inf')
            
        diff = abs(dt1 - dt2)
        return diff.total_seconds() / 3600
    
    def _check_geo_velocity(self, transaction, history):
        """Check for geographical velocity anomalies."""
        if not history:
            return False
            
        # Get recent transactions sorted by timestamp (newest first)
        recent_transactions = sorted(
            [tx for tx in history if tx.get('timestamp')],
            key=lambda tx: self._parse_timestamp(tx.get('timestamp')),
            reverse=True
        )
        
        if not recent_transactions:
            return False
            
        # Get the most recent transaction
        last_transaction = recent_transactions[0]
        
        # If locations are different
        if transaction.get('location') != last_transaction.get('location'):
            # Calculate time difference in hours
            time_diff = self._time_diff_hours(
                transaction.get('timestamp'), 
                last_transaction.get('timestamp')
            )
            
            # Flag if transactions in different locations are less than 3 hours apart
            return time_diff < 3
        
        return False
    
    def _detect_unusual_amount(self, transaction, history):
        """Detect unusual transaction amounts using statistical methods."""
        if not history or len(history) < 10:
            return False
            
        try:
            # Convert amount to float, handling string formats
            current_amount = self._parse_amount(transaction.get('amount', 0))
            
            # Get amounts from history
            amounts = [self._parse_amount(tx.get('amount', 0)) for tx in history]
            
            # Calculate mean and standard deviation
            mean = sum(amounts) / len(amounts)
            variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return False
                
            # Calculate z-score (number of standard deviations from the mean)
            z_score = abs(current_amount - mean) / std_dev
            
            # Flag if amount is more than 3 standard deviations from the mean
            return z_score > 3
        except Exception as e:
            print(f"Error in unusual amount detection: {str(e)}")
            return False
    
    def _parse_amount(self, amount):
        """Parse amount to float, handling different formats."""
        try:
            if isinstance(amount, (int, float)):
                return float(amount)
            elif isinstance(amount, str):
                # Remove currency symbols and commas
                amount = re.sub(r'[$£€,]', '', amount)
                return float(amount)
            else:
                return 0.0
        except:
            return 0.0
    
    def _detect_suspicious_sequence(self, transaction, history):
        """Detect suspicious sequences of transactions."""
        if not history or len(history) < 3:
            return False
            
        try:
            # Get recent transactions from the last 24 hours
            current_time = self._parse_timestamp(transaction.get('timestamp'))
            
            recent_transactions = [
                tx for tx in history
                if self._time_diff_hours(transaction.get('timestamp'), tx.get('timestamp')) <= 24
            ]
            
            # Add current transaction
            all_transactions = recent_transactions + [transaction]
            
            # Sort by timestamp
            sorted_transactions = sorted(
                all_transactions,
                key=lambda tx: self._parse_timestamp(tx.get('timestamp'))
            )
            
            if len(sorted_transactions) < 3:
                return False
                
            # Pattern 1: Test transaction followed by larger ones
            for i in range(len(sorted_transactions) - 2):
                tx1 = sorted_transactions[i]
                tx2 = sorted_transactions[i + 1]
                tx3 = sorted_transactions[i + 2]
                
                amount1 = self._parse_amount(tx1.get('amount', 0))
                amount2 = self._parse_amount(tx2.get('amount', 0))
                amount3 = self._parse_amount(tx3.get('amount', 0))
                
                if amount1 < 5 and amount2 > amount1 and amount3 > amount2:
                    return True
                    
            # Pattern 2: Multiple transactions at same merchant in short time
            merchant_counts = {}
            for tx in sorted_transactions:
                merchant = tx.get('description', '') or tx.get('merchantCategory', '')
                merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1
                
                if merchant_counts[merchant] >= 3:
                    return True
                    
            return False
        except Exception as e:
            print(f"Error in suspicious sequence detection: {str(e)}")
            return False
    
    def _detect_description_anomaly(self, transaction):
        """Analyze description text for suspicious patterns."""
        description = transaction.get('description', '')
        if not description:
            return False
            
        description = str(description).lower()
        
        try:
            # Check for suspicious keywords
            suspicious_keywords = [
                'verify', 'verification', 'confirm', 'validate', 'security',
                'update', 'account', 'suspicious', 'urgent', 'bitcoin', 'crypto',
                'wire transfer', 'western union', 'moneygram', 'gift card',
                'lottery', 'prize', 'won', 'inheritance', 'donation', 'charity'
            ]
            
            for keyword in suspicious_keywords:
                if keyword in description:
                    return True
                    
            # Check for unusual character patterns
            unusual_patterns = [
                r'[^\x00-\x7F]+',  # Non-ASCII characters
                r'(.)\1{4,}',       # Repeated characters (e.g., "aaaaa")
                r'[A-Z]{10,}'       # Excessive uppercase
            ]
            
            for pattern in unusual_patterns:
                if re.search(pattern, description):
                    return True
                    
            return False
        except Exception as e:
            print(f"Error in description anomaly detection: {str(e)}")
            return False


class FraudDetectionApp:
    """
    Main application for bank statement fraud detection.
    Handles document processing, analysis, and reporting.
    """
    
    def __init__(self):
        """Initialize the application components."""
        self.document_processor = AIDocumentProcessor()
        self.fraud_detector = EnhancedFraudDetectionSystem()
        self.document_data = None
        self.transactions = []
        self.results = []
        self.user_profile = {
            'large_amount_threshold': 2000
        }
    
    def process_document(self, file_path: str) -> Dict:
        """Process a document and extract transaction data."""
        print(f"\n=== AI-Enhanced Bank Statement Fraud Detection ===\n")
        print(f"Processing document: {file_path}")
        
        # Process the document
        self.document_data = self.document_processor.process_document(file_path)
        
        # Generate insights
        insights = self.document_processor.generate_insights(
            self.document_data['data'], 
            self.document_data.get('category_mapping', {})
        ) if hasattr(self.document_processor, 'generate_insights') else []
        
        # Display processing information
        print(f"\n=== Document Processing Results ===")
        print(f"Format detected: {self.document_data.get('format', 'unknown')}")
        print(f"Processing method: {self.document_data.get('processing_method', 'standard')}")
        print(f"Records found: {len(self.document_data['data'])}")
        print(f"Fields detected: {len(self.document_data['headers'])}")
        
        # Display any metadata if available
        if 'metadata' in self.document_data:
            print("\n=== Document Metadata ===")
            for key, value in self.document_data['metadata'].items():
                print(f"  {key}: {value}")
        
        # Display insights if available
        if insights:
            print("\n=== AI Insights ===")
            for insight in insights:
                print(f"  {insight}")
        
        return self.document_data
        
    def generate_insights(self, data, category_mapping):
        """
        Generate AI-enhanced insights from transaction data based on category mapping.
        
        Args:
            data: List of transaction data dictionaries
            category_mapping: Dictionary mapping fields to banking categories
            
        Returns:
            list: List of insight strings
        """
        if not data:
            return ["No data available for insights."]
            
        insights = []
        
        try:
            # Get relevant fields based on category mapping
            date_field = None
            amount_field = None
            description_field = None
            
            for field, mapping in category_mapping.items():
                if mapping['category'] == 'transaction_date':
                    date_field = field
                elif mapping['category'] == 'amount':
                    amount_field = field
                elif mapping['category'] == 'description':
                    description_field = field
            
            # Analyze transaction amounts
            if amount_field:
                # Convert amounts to floats, handling string formats
                amounts = []
                for record in data:
                    amount = record.get(amount_field)
                    if amount:
                        if isinstance(amount, str):
                            # Remove currency symbols and commas
                            clean_amount = re.sub(r'[$£€,]', '', amount)
                            try:
                                amounts.append(float(clean_amount))
                            except:
                                pass
                        elif isinstance(amount, (int, float)):
                            amounts.append(float(amount))
                
                if amounts:
                    total = sum(amounts)
                    avg = total / len(amounts)
                    max_amount = max(amounts)
                    min_amount = min(amounts)
                    
                    insights.append(f"Total transaction amount: ${total:.2f}")
                    insights.append(f"Average transaction amount: ${avg:.2f}")
                    insights.append(f"Largest transaction: ${max_amount:.2f}")
                    insights.append(f"Smallest transaction: ${min_amount:.2f}")
            
            # Analyze transaction dates
            if date_field:
                dates = []
                for record in data:
                    date_str = record.get(date_field)
                    if date_str:
                        try:
                            # Try common date formats
                            date_formats = [
                                "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
                                "%m-%d-%Y", "%d-%m-%Y", "%m/%d/%y"
                            ]
                            
                            for fmt in date_formats:
                                try:
                                    date = datetime.strptime(str(date_str), fmt)
                                    dates.append(date)
                                    break
                                except:
                                    continue
                        except:
                            pass
                
                if dates:
                    earliest = min(dates)
                    latest = max(dates)
                    
                    insights.append(f"Date range: {earliest.strftime('%m/%d/%Y')} to {latest.strftime('%m/%d/%Y')}")
                    
                    # If more than 5 transactions, analyze frequency
                    if len(dates) >= 5:
                        date_diff = (latest - earliest).days
                        if date_diff > 0:
                            transactions_per_day = len(dates) / date_diff
                            insights.append(f"Average transactions per day: {transactions_per_day:.2f}")
            
            # Analyze merchants/categories
            if description_field:
                descriptions = {}
                for record in data:
                    desc = record.get(description_field)
                    if desc:
                        descriptions[desc] = descriptions.get(desc, 0) + 1
                
                if descriptions:
                    # Find most common merchants
                    sorted_merchants = sorted(descriptions.items(), key=lambda x: x[1], reverse=True)
                    top_merchants = sorted_merchants[:3]
                    
                    if top_merchants:
                        merchants_str = ", ".join(f"{m[0]} ({m[1]} transactions)" for m in top_merchants)
                        insights.append(f"Top merchants: {merchants_str}")
            
            return insights
                    
        except Exception as e:
            return [f"Error generating insights: {str(e)}"]

    def map_fields(self, field_mapping: Dict = None) -> None:
        """
        Map document fields to transaction fields.
        
        Args:
            field_mapping: Dictionary mapping transaction fields to document headers
        """
        if not self.document_data:
            raise ValueError("No document data. Process a document first.")
        
        if field_mapping is None:
            # Try to create automatic mapping using AI-detected categories
            field_mapping = self._create_automatic_mapping()
            
            if not field_mapping:
                print("\nWarning: Could not create automatic field mapping. Using best guess...")
                field_mapping = self._create_best_guess_mapping()
        
        # Validate mapping
        required_fields = ['id', 'timestamp', 'amount', 'description']
        missing_fields = [field for field in required_fields if field not in field_mapping]
        
        if missing_fields:
            print("\nWarning: The following fields are missing from mapping:")
            for field in missing_fields:
                print(f"  - {field}")
                
            print("\nWill use default values for missing fields.")
        
        self.field_mapping = field_mapping
        
        # Display mapping
        print("\n=== Field Mapping ===")
        for field, header in self.field_mapping.items():
            confidence = ""
            if self.document_data.get('category_mapping') and header in self.document_data['category_mapping']:
                conf = self.document_data['category_mapping'][header]['confidence']
                confidence = f" (AI confidence: {conf:.0%})"
            
            print(f"  {field} -> {header}{confidence}")
    
    def _create_automatic_mapping(self) -> Dict:
        """Create automatic field mapping using AI-detected categories."""
        if not self.document_data.get('category_mapping'):
            return {}
        
        # Define field to category mapping
        field_to_category = {
            'id': 'transaction_id',
            'timestamp': 'transaction_date',
            'amount': 'amount',
            'description': 'description',
            'merchantCategory': 'category',
            'location': 'location',
            'balance': 'balance'
        }
        
        # Create mapping dictionary
        mapping = {}
        
        for field, category in field_to_category.items():
            # Find header with matching category
            for header, info in self.document_data['category_mapping'].items():
                if info['category'] == category:
                    mapping[field] = header
                    break
        
        return mapping
    
    def _create_best_guess_mapping(self) -> Dict:
        """Create best guess mapping when AI detection fails."""
        headers = self.document_data['headers']
        mapping = {}
        
        # Common field name patterns
        patterns = {
            'id': ['id', 'transaction id', 'reference', 'ref', 'confirmation', 'number'],
            'timestamp': ['date', 'time', 'transaction date', 'posting date', 'posted', 'datetime'],
            'amount': ['amount', 'sum', 'transaction amount', 'debit', 'credit', 'payment', 'value'],
            'description': ['description', 'details', 'memo', 'narrative', 'merchant', 'payee', 'note'],
            'merchantCategory': ['category', 'type', 'transaction type', 'classification'],
            'location': ['location', 'place', 'address', 'merchant location', 'city', 'state']
        }
        
        # For each field, find the best matching header
        for field, keywords in patterns.items():
            best_match = None
            best_score = 0
            
            for header in headers:
                header_lower = str(header).lower()
                
                # Calculate match score
                score = 0
                for keyword in keywords:
                    if keyword == header_lower:
                        score = 1.0
                        break
                    elif keyword in header_lower:
                        score = max(score, len(keyword) / len(header_lower))
                
                if score > best_score:
                    best_score = score
                    best_match = header
            
            if best_match and best_score > 0.3:
                mapping[field] = best_match
        
        # If still no mapping for essential fields, make educated guesses
        if 'timestamp' not in mapping and headers:
            # Try to find any header that might contain dates by examining values
            for header in headers:
                if any(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}', 
                               str(row.get(header, ''))) 
                       for row in self.document_data['data'][:5]):
                    mapping['timestamp'] = header
                    break
        
        if 'amount' not in mapping and headers:
            # Try to find any header that might contain money values
            for header in headers:
                if any(re.search(r'[$€£]?\d+\.\d{2}', str(row.get(header, ''))) 
                       for row in self.document_data['data'][:5]):
                    mapping['amount'] = header
                    break
        
        if 'description' not in mapping and headers:
            # Try to find a field with longer text values
            text_lengths = {}
            for header in headers:
                if header not in mapping.values():
                    avg_length = sum(len(str(row.get(header, ''))) 
                                    for row in self.document_data['data'][:5]) / 5
                    text_lengths[header] = avg_length
            
            if text_lengths:
                # Use the header with the longest average text
                mapping['description'] = max(text_lengths.items(), key=lambda x: x[1])[0]
        
        # Generate a unique ID field if not found
        if 'id' not in mapping:
            mapping['id'] = '_generated_id'
        
        return mapping
    
    def process_transactions(self) -> List[Dict]:
        """Process document data into transactions."""
        if not self.document_data or not hasattr(self, 'field_mapping'):
            raise ValueError("Document data or field mapping not available.")
        
        data = self.document_data['data']
        mapping = self.field_mapping
        
        transactions = []
        
        for i, row in enumerate(data):
            # Create basic transaction object
            transaction = {
                'userId': 'user1',  # Default user ID
            }
            
            # Handle ID field
            if 'id' in mapping:
                if mapping['id'] == '_generated_id':
                    # Generate a unique ID
                    transaction['id'] = f"tx{i+1:04d}"
                else:
                    transaction['id'] = row.get(mapping['id'], f"tx{i+1:04d}")
            else:
                transaction['id'] = f"tx{i+1:04d}"
            
            # Process timestamp
            if 'timestamp' in mapping:
                raw_date = row.get(mapping['timestamp'])
                try:
                    # Try to parse the date
                    parsed_date = self.fraud_detector._parse_timestamp(raw_date)
                    transaction['timestamp'] = parsed_date.isoformat()
                except:
                    # If parsing fails, use current date
                    transaction['timestamp'] = datetime.now().isoformat()
                    print(f"Warning: Could not parse date: {raw_date}")
            else:
                transaction['timestamp'] = datetime.now().isoformat()
            
            # Process amount
            if 'amount' in mapping:
                amount = row.get(mapping['amount'], 0)
                
                # If amount is a string, try to clean and parse it
                if isinstance(amount, str):
                    # Remove currency symbols and commas
                    amount = re.sub(r'[$£€,]', '', amount)
                    # Parse to float
                    try:
                        amount = float(amount)
                    except:
                        amount = 0
                
                transaction['amount'] = abs(float(amount))  # Use absolute value
            else:
                transaction['amount'] = 0
            
            # Process location
            transaction['location'] = row.get(mapping.get('location', ''), 'Unknown')
            
            # Process merchant category
            transaction['merchantCategory'] = row.get(mapping.get('merchantCategory', ''), 'Other')
            
            # Process description (optional)
            if 'description' in mapping:
                transaction['description'] = row.get(mapping['description'], '')
            else:
                transaction['description'] = ''
            
            # Process balance (optional)
            if 'balance' in mapping:
                balance = row.get(mapping['balance'], '')
                if isinstance(balance, str):
                    # Clean balance string
                    balance = re.sub(r'[$£€,]', '', balance)
                    try:
                        balance = float(balance)
                    except:
                        balance = ''
                transaction['balance'] = balance
            
            # Add the original row data for reference
            transaction['originalData'] = row
            
            transactions.append(transaction)
        
        self.transactions = transactions
        return transactions
    
    def analyze_transactions(self) -> List[Dict]:
        """Analyze transactions for fraud indicators."""
        if not self.transactions:
            raise ValueError("No transactions to analyze. Process transactions first.")
        
        # Analyze the transactions
        self.results = self.fraud_detector.analyze_batch(
            self.transactions, 
            self.user_profile
        )
        
        return self.results
    
    def display_results(self) -> None:
        """Display analysis results."""
        if not self.results:
            raise ValueError("No analysis results available.")
        
        # Count risk levels
        total = len(self.results)
        high_risk = sum(1 for r in self.results if r['fraud_likelihood'] == 'High')
        medium_risk = sum(1 for r in self.results if r['fraud_likelihood'] == 'Medium')
        low_risk = sum(1 for r in self.results if r['fraud_likelihood'] == 'Low')
        
        # Calculate percentages
        high_risk_pct = (high_risk / total) * 100 if total > 0 else 0
        medium_risk_pct = (medium_risk / total) * 100 if total > 0 else 0
        low_risk_pct = (low_risk / total) * 100 if total > 0 else 0
        
        # Calculate amount statistics
        total_amount = sum(float(tx.get('amount', 0)) for tx in self.transactions)
        high_risk_amount = sum(
            float(tx.get('amount', 0)) 
            for tx, result in zip(self.transactions, self.results) 
            if result['fraud_likelihood'] == 'High'
        )
        high_risk_amount_pct = (high_risk_amount / total_amount) * 100 if total_amount > 0 else 0
        
        # Display summary
        print("\n=== Analysis Results ===")
        print(f"Total transactions: {total}")
        print(f"Risk levels:")
        print(f"  - High Risk: {high_risk} ({high_risk_pct:.1f}%)")
        print(f"  - Medium Risk: {medium_risk} ({medium_risk_pct:.1f}%)")
        print(f"  - Low Risk: {low_risk} ({low_risk_pct:.1f}%)")
        print(f"\nTransaction volume:")
        print(f"  - Total: ${total_amount:.2f}")
        print(f"  - High Risk Amount: ${high_risk_amount:.2f} ({high_risk_amount_pct:.1f}%)")
        
        # Display top high-risk transactions
        if high_risk > 0:
            print("\nTop High-Risk Transactions:")
            high_risk_transactions = [
                (result, tx) 
                for result, tx in zip(self.results, self.transactions) 
                if result['fraud_likelihood'] == 'High'
            ]
            
            # Sort by risk score
            high_risk_transactions.sort(key=lambda x: x[0]['risk_score'], reverse=True)
            
            # Display top 5 (or all if less than 5)
            for i, (result, tx) in enumerate(high_risk_transactions[:5]):
                date = self.fraud_detector._format_timestamp(tx.get('timestamp'))
                desc = tx.get('description', '') or tx.get('merchantCategory', 'Unknown')
                amount = float(tx.get('amount', 0))
                
                print(f"  {i+1}. {date} - {desc} - ${amount:.2f} (Risk Score: {result['risk_score']})")
                
                # Show triggered rules
                rules = ", ".join(rule['rule_name'] for rule in result['triggered_rules'])
                print(f"     Triggered rules: {rules}")
    
    def generate_ai_explanation(self) -> str:
        """Generate an AI explanation of fraud analysis results."""
        if not self.results or not self.transactions:
            raise ValueError("No analysis results available.")
        
        explanation = self.fraud_detector.generate_explanation(
            self.results, 
            self.transactions
        )
        
        return explanation
    
    def export_analysis_to_csv(self, output_file: str) -> None:
        """
        Export fraud analysis results to CSV.
        
        Args:
            output_file: Path to the output CSV file
        """
        if not self.results or not self.transactions:
            raise ValueError("No analysis results available.")
        
        # Create DataFrame for export
        data = []
        
        for transaction, result in zip(self.transactions, self.results):
            # Get original data
            row = transaction.get('originalData', {}).copy()
            
            # Add fraud analysis results
            row['Risk_Level'] = result['fraud_likelihood']
            row['Risk_Score'] = result['risk_score']
            row['Triggered_Rules'] = "; ".join(
                rule['rule_name'] for rule in result['triggered_rules']
            )
            row['AI_Enhanced'] = 'Yes' if result.get('ai_enhanced') else 'No'
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        print(f"\nExported analysis to: {output_file}")
    
    def visualize_results(self) -> None:
        """Generate visualizations of fraud analysis results."""
        if not self.results:
            raise ValueError("No analysis results available.")
        
        # Setup the plot aesthetics
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Risk distribution pie chart
        risk_counts = {
            'High': sum(1 for r in self.results if r['fraud_likelihood'] == 'High'),
            'Medium': sum(1 for r in self.results if r['fraud_likelihood'] == 'Medium'),
            'Low': sum(1 for r in self.results if r['fraud_likelihood'] == 'Low')
        }
        
        colors = ['#ff6b6b', '#feca57', '#1dd1a1']
        wedges, texts, autotexts = ax1.pie(
            risk_counts.values(),
            labels=risk_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        # Make the percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('Transaction Risk Distribution', fontweight='bold', pad=20)
        
        # 2. Top triggered rules bar chart
        rule_counts = {}
        for result in self.results:
            for rule in result['triggered_rules']:
                rule_name = rule['rule_name']
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        # Sort and get top rules
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Plot bar chart
        rule_names = [r[0] for r in top_rules]
        rule_values = [r[1] for r in top_rules]
        
        bars = ax2.barh(rule_names, rule_values, color='#3498db')
        
        # Add count labels to the right of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(
                width + 0.1,
                bar.get_y() + bar.get_height()/2,
                f'{width:.0f}',
                va='center',
                fontweight='bold'
            )
        
        ax2.set_title('Top Triggered Rules', fontweight='bold', pad=20)
        ax2.set_xlabel('Number of Occurrences')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('fraud_analysis_visualization.png', dpi=300, bbox_inches='tight')
        print("\nSaved visualization to: fraud_analysis_visualization.png")
        
        # Show the plot if running in interactive environment
        plt.show()

        # Create a time series of transactions by risk
        if any('timestamp' in tx for tx in self.transactions):
            plt.figure(figsize=(12, 6))
            
            # Convert timestamps to datetime
            dates = [self.fraud_detector._parse_timestamp(tx.get('timestamp')) for tx in self.transactions]
            amounts = [float(tx.get('amount', 0)) for tx in self.transactions]
            risk_levels = [result['fraud_likelihood'] for result in self.results]
            
            # Use different colors for risk levels
            colors = {'High': '#ff6b6b', 'Medium': '#feca57', 'Low': '#1dd1a1'}
            risk_colors = [colors[level] for level in risk_levels]
            
            # Plot scatter with size proportional to amount
            plt.scatter(dates, amounts, c=risk_colors, alpha=0.7, s=[a*5 for a in amounts])
            
            plt.title('Transactions Over Time by Risk Level', fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Amount ($)')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=level,
                      markerfacecolor=color, markersize=10)
                for level, color in colors.items()
            ]
            plt.legend(handles=legend_elements, title='Risk Level')
            
            plt.tight_layout()
            plt.savefig('transaction_time_series.png', dpi=300, bbox_inches='tight')
            print("Saved time series visualization to: transaction_time_series.png")
            plt.show()


def main():
    """Main function for the fraud detection application."""
    parser = argparse.ArgumentParser(description='AI-Enhanced Bank Statement Fraud Detection')
    parser.add_argument('file', help='Path to the bank statement file (any format)')
    parser.add_argument('--threshold', type=float, default=2000, 
                       help='Threshold for large transaction amount (default: 2000)')
    parser.add_argument('--output', default='fraud_analysis_report.csv',
                       help='Output file for analysis report (default: fraud_analysis_report.csv)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed debug output')
    
    args = parser.parse_args()
    
    # Initialize the app
    app = FraudDetectionApp()
    
    # Set user profile
    app.user_profile['large_amount_threshold'] = args.threshold
    
    try:
        # Process document
        app.process_document(args.file)
        
        # Map fields
        app.map_fields()
        
        # Process and analyze transactions
        app.process_transactions()
        app.analyze_transactions()
        
        # Display results
        app.display_results()
        
        # Export results
        app.export_analysis_to_csv(args.output)
        
        # Generate visualizations
        if not args.no_viz:
            app.visualize_results()
        
        # Display AI explanation
        print("\n=== AI Explanation ===")
        explanation = app.generate_ai_explanation()
        print(explanation)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
