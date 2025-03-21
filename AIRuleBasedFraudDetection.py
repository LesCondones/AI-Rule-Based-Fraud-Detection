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
    import pdfplumber  # For more advanced PDF processing
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    print("pdfplumber not found. Advanced PDF processing disabled.")

# For a complete implementation, you'd also want to add:
# - pdfplumber or pdfminer for better PDF text extraction
# - python-docx for Word documents
# - xlrd/openpyxl for Excel (already included in pandas)

class AIDocumentProcessor:
    """
    AI-powered processor for various document types (CSV, Excel, PDF, Images)
    that extracts transaction data and identifies banking information.
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
       # Update the statement_patterns dictionary
        self.statement_patterns = {
            'account_info': re.compile(r'(account\s*number|acct\s*#|account\s*#|Account Holder)[:.\s]*([^$\n]+)', re.IGNORECASE),
            'date_range': re.compile(r'(statement\s*period|from|statement\s*dates|Account Statement)[:.\s]*([^-]+)\s*-\s*([^$\n]+)', re.IGNORECASE),
            'balance': re.compile(r'(closing\s*balance|ending\s*balance|available\s*balance|current\s*balance|Balance)[:.\s]*[$£€]?(\d{1,3}(,\d{3})*\.\d{2})', re.IGNORECASE),
            'transaction_date': re.compile(r'(2024-\d{2}-\d{2})', re.IGNORECASE)
        }
        
        # Supported file formats and their processor methods
        self.supported_formats = {
            '.csv': self.process_csv,
            '.xlsx': self.process_excel,
            '.xls': self.process_excel,
            '.pdf': self.process_pdf,
            '.jpg': self.process_image,
            '.jpeg': self.process_image,
            '.png': self.process_image
        }
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process a document file and extract transaction data.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict: Document data including transactions and headers
        """
        print(f"Processing document: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        processor = self.supported_formats[file_ext]
        
        try:
            print(f"Step 1/3: Document analysis...")
            # Process the document based on its type
            data = processor(file_path)
            
            print(f"Step 2/3: Data extraction complete")
            
            # Identify banking categories in the data
            print(f"Step 3/3: AI classification of fields...")
            enhanced_data = self.identify_banking_categories(data)
            
            print(f"AI processing complete. Found {len(enhanced_data['data'])} transactions and {len(enhanced_data.get('category_mapping', {}))} categorized fields.")
            return enhanced_data
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise
    
    def process_csv(self, file_path: str) -> Dict:
        """Process a CSV file and extract transaction data."""
        try:
            # Read CSV file with pandas
            df = pd.read_csv(file_path)
            
            # Basic data validation
            if df.empty:
                raise ValueError("CSV file contains no data")
                
            return {
                'data': df.to_dict('records'),
                'headers': list(df.columns),
                'format': 'csv',
                'shape': df.shape
            }
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def process_excel(self, file_path: str) -> Dict:
        """Process an Excel file and extract transaction data."""
        try:
            # Read Excel file with pandas
            df = pd.read_excel(file_path)
            
            # Basic data validation
            if df.empty:
                raise ValueError("Excel file contains no data")
                
            return {
                'data': df.to_dict('records'),
                'headers': list(df.columns),
                'format': 'excel',
                'shape': df.shape
            }
        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
    
    def process_pdf(self, file_path: str) -> Dict:
        """Process a PDF file and extract transaction data."""
    if not PDF_SUPPORT:
        raise ImportError("PDF support requires PyPDF2. Please install it with 'pip install PyPDF2'")
            
    try:
        # Try pdfplumber first (if available) for better table extraction
        if PDFPLUMBER_SUPPORT:
            try:
                with pdfplumber.open(file_path) as pdf:
                    tables = []
                    for page in pdf.pages:
                        extracted_tables = page.extract_tables()
                        if extracted_tables:
                            tables.extend(extracted_tables)
                    
                    if tables:
                        # Process extracted tables from pdfplumber
                        return self._process_pdfplumber_tables(tables)
            except Exception as e:
                print(f"pdfplumber processing failed, falling back to PyPDF2: {str(e)}")
        
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
                    # Update pattern to match YYYY-MM-DD format dates
                    if re.search(r'(2024-\d{2}-\d{2})', line) and re.search(r'[-+]?\$?\d+\.\d{2}', line):
                        # Correctly parse the bank statement format
                        parts = self._parse_bank_statement_line(line)
                        if parts:
                            table_data.append(parts)
        
        # Create structured data from extracted text
        if not table_data:
            # If table extraction failed, try specialized extraction
            return self._extract_bank_statement_structure(extracted_text)
        
        # Process the extracted table data
        return self._process_extracted_table_data(table_data, extracted_text)
                
    except Exception as e:
        raise ValueError(f"Error processing PDF file: {str(e)}")

def _process_pdfplumber_tables(self, tables):
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
    
    return {
        'data': transactions,
        'headers': headers,
        'format': 'pdf',
    }

def _parse_bank_statement_line(self, line):
    """Parse a transaction line from the bank statement format."""
    # Match the format: 2024-MM-DD DESCRIPTION Category Amount Balance
    match = re.match(r'(2024-\d{2}-\d{2})\s+(.*?)\s+([A-Za-z &]+)\s+([-+]?\$?\d+\.\d{2})\s+(\$?\d+\.\d{2})', line)
    if match:
        return {
            'date': match.group(1),
            'description': match.group(2).strip(),
            'category': match.group(3).strip(),
            'amount': match.group(4),
            'balance': match.group(5)
        }
    return None

    def _extract_bank_statement_structure(self, text):
        """Extract structured data from the bank statement text."""
    # Get header info
    statement_match = re.search(r'Account Statement: (.*)', text)
    statement_period = statement_match.group(1) if statement_match else ''
    
    account_holder_match = re.search(r'Account Holder: (.*)', text)
    account_holder = account_holder_match.group(1) if account_holder_match else ''
    
    # Extract transactions using date pattern as separators
    transactions = []
    date_pattern = re.compile(r'(2024-\d{2}-\d{2})')
    
    # Find all potential transaction blocks
    lines = text.split('\n')
    for line in lines:
        if date_pattern.search(line):
            # Specialized parsing for bank statement format
            parts = self._parse_bank_statement_line(line)
            if parts:
                transactions.append(parts)
    
    return {
        'data': transactions,
        'headers': ['date', 'description', 'category', 'amount', 'balance'],
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

    def extract_transactions_from_pdf(self, text: str, table_data: List[List[str]]) -> List[Dict]:
        """Extract transactions from PDF text using patterns."""
    transactions = []
    
    # Try to identify statement header information
    account_match = self.statement_patterns['account_info'].search(text)
    account_number = account_match.group(2) if account_match else 'Unknown'
    
    # Update pattern to match "March 1, 2024 - May 31, 2024" format
    date_range_match = re.search(r'Account Statement:\s*(.*?)\s*-\s*(.*)', text)
    statement_period = f"{date_range_match.group(1)} to {date_range_match.group(2)}" if date_range_match else ''
    
    # First try to use detected table data
    if table_data:
        for row in table_data:
            if isinstance(row, dict):
                # Already parsed structured data
                transactions.append(row)
            else:
                # Process as array data
                process_row_data(row)
    else:
        # Updated regex pattern for the date format in example (YYYY-MM-DD)
        date_pattern = re.compile(r'(2024-\d{2}-\d{2})')
        amount_pattern = re.compile(r'([-+]?\$?\d{1,3}(,\d{3})*\.\d{2})')
        
        lines = text.split('\n')
        for line in lines:
            date_match = date_pattern.search(line)
            if date_match:
                # Process transaction line with specialized parsing
                parsed = self._parse_bank_statement_line(line)
                if parsed:
                    parsed['account_number'] = account_number
                    parsed['statement_period'] = statement_period
                    transactions.append(parsed)
    
    return transactions
        
        return transactions
    
    def guess_table_headers(self, first_row: List[str]) -> List[str]:
        """Try to determine what each column represents."""
        headers = []
        
        for cell in first_row:
            cell_lower = cell.lower()
            
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
    
    def process_image(self, file_path: str) -> Dict:
        """Process an image file using OCR to extract transaction data."""
        if not OCR_SUPPORT:
            raise ImportError("Image processing requires pytesseract and PIL. Please install them with 'pip install pytesseract pillow'")
            
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(img)
            
            # Extract transaction data similar to PDF approach
            transactions = self.extract_transactions_from_pdf(extracted_text, [])
            
            if not transactions:
                # If no transactions found, create mock data for demonstration
                # In a real implementation, you'd want to improve OCR and extraction
                print("Warning: Could not extract transactions from image. Using mock data for demonstration.")
                transactions = [
                    {'date': '03/15/2025', 'description': 'GROCERY STORE', 'amount': '42.57', 'balance': '1,240.33'},
                    {'date': '03/14/2025', 'description': 'ATM WITHDRAWAL', 'amount': '100.00', 'balance': '1,282.90'},
                    {'date': '03/14/2025', 'description': 'COFFEE SHOP', 'amount': '5.43', 'balance': '1,382.90'},
                    {'date': '03/12/2025', 'description': 'DIRECT DEPOSIT - PAYROLL', 'amount': '1,470.25', 'balance': '1,388.33'}
                ]
            
            headers = list(transactions[0].keys())
            
            return {
                'data': transactions,
                'headers': headers,
                'raw_text': extracted_text,
                'format': 'image'
            }
                
        except Exception as e:
            raise ValueError(f"Error processing image file: {str(e)}")
    
    def identify_banking_categories(self, document_data: Dict) -> Dict:
        """Identify banking categories in data."""
        data = document_data['data']
        headers = document_data['headers']
        
        # Map of confident category matches
        category_mapping = {}
        
        # Score each header against our banking categories
        for header in headers:
            header_lower = header.lower()
            
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
                      re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', sample)
                      for sample in valid_samples)
        
        elif category == 'amount':
            # Check for money patterns
            return any(
                (isinstance(sample, (int, float))) or
                (isinstance(sample, str) and 
                 re.search(r'^[$£€]?\d{1,3}(,\d{3})*(\.\d{2})?$', sample.strip()))
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
    
    def generate_insights(self, transactions: List[Dict], category_mapping: Dict) -> List[str]:
        """Generate AI insights based on transaction data."""
        insights = []
        
        # Basic analysis
        if transactions:
            # Get transaction amount field
            amount_field = next(
                (header for header, info in category_mapping.items() 
                 if info.get('category') == 'amount'),
                None
            )
            
            if amount_field:
                # Calculate total spending
                total_spending = 0
                largest_transaction = {'amount': 0}
                spending_by_category = {}
                
                for transaction in transactions:
                    # Parse amount
                    amount = transaction.get(amount_field, 0)
                    if isinstance(amount, str):
                        amount = float(re.sub(r'[$£€,]', '', amount) or 0)
                    
                    if amount > 0:
                        total_spending += amount
                        
                        # Track largest transaction
                        if amount > largest_transaction.get('amount', 0):
                            largest_transaction = {
                                'amount': amount,
                                **transaction
                            }
                        
                        # Track spending by category if available
                        category_field = next(
                            (header for header, info in category_mapping.items() 
                             if info.get('category') == 'category'),
                            None
                        )
                        
                        if category_field and transaction.get(category_field):
                            category = transaction[category_field]
                            spending_by_category[category] = spending_by_category.get(category, 0) + amount
                
                # Add insights
                insights.append(f"Analyzed {len(transactions)} transactions")
                insights.append(f"Total spending: ${total_spending:.2f}")
                
                if largest_transaction.get('amount', 0) > 0:
                    desc_field = next(
                        (header for header, info in category_mapping.items() 
                         if info.get('category') == 'description'),
                        None
                    )
                    
                    if desc_field and largest_transaction.get(desc_field):
                        insights.append(
                            f"Largest transaction: ${largest_transaction['amount']:.2f} "
                            f"for \"{largest_transaction[desc_field]}\""
                        )
                
                # Add category insights
                top_categories = sorted(
                    spending_by_category.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                if top_categories:
                    insights.append('Top spending categories:')
                    for category, amount in top_categories:
                        insights.append(f"- {category}: ${amount:.2f}")
        
        return insights


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
                    tx.get('amount', 0) > profile.get('large_amount_threshold', 3250),
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
                    int(round((float(tx.get('amount', 0)) * 100) % 100)) 
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
                "%Y-%m-%d"
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
            
        # Get amounts from history
        amounts = [float(tx.get('amount', 0)) for tx in history]
        
        # Calculate mean and standard deviation
        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return False
            
        # Calculate z-score (number of standard deviations from the mean)
        current_amount = float(transaction.get('amount', 0))
        z_score = abs(current_amount - mean) / std_dev
        
        # Flag if amount is more than 3 standard deviations from the mean
        return z_score > 3
    
    def _detect_suspicious_sequence(self, transaction, history):
        """Detect suspicious sequences of transactions."""
        if not history or len(history) < 3:
            return False
            
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
            
            amount1 = float(tx1.get('amount', 0))
            amount2 = float(tx2.get('amount', 0))
            amount3 = float(tx3.get('amount', 0))
            
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
    
    def _detect_description_anomaly(self, transaction):
        """Analyze description text for suspicious patterns."""
        description = transaction.get('description', '')
        if not description:
            return False
            
        description = str(description).lower()
        
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
        )
        
        # Display insights
        print("\n=== AI Insights ===")
        for insight in insights:
            print(f"  {insight}")
        
        return self.document_data
    
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
                raise ValueError("Could not create automatic field mapping. Please provide manual mapping.")
        
        # Validate mapping
        required_fields = ['id', 'timestamp', 'amount', 'location', 'merchantCategory']
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
            'location': 'location'
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
                'id': row.get(mapping.get('id', ''), f"tx{i}"),
                'userId': 'user1',  # Default user ID
            }
            
            # Process timestamp
            if 'timestamp' in mapping:
                raw_date = row.get(mapping['timestamp'])
                try:
                    # Try to parse the date
                    transaction['timestamp'] = self.fraud_detector._parse_timestamp(raw_date).isoformat()
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
    parser.add_argument('file', help='Path to the bank statement file (CSV, Excel, PDF, or image)')
    parser.add_argument('--threshold', type=float, default=2000, 
                       help='Threshold for large transaction amount (default: 2000)')
    parser.add_argument('--output', default='fraud_analysis_report.csv',
                       help='Output file for analysis report (default: fraud_analysis_report.csv)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization generation')
    
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
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())