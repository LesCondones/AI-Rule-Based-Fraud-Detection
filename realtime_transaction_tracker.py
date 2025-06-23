"""
Real-time Transaction Tracking System
Integrates with payment processors to capture transactions instantly
Author: Lester L. Artis Jr.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
import threading
import queue
import logging
from collections import defaultdict
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RealTimeTransaction:
    """Real-time transaction data structure"""
    transaction_id: str
    account_id: str
    amount: float
    merchant: str
    timestamp: str
    location: Optional[str] = None
    card_last_four: Optional[str] = None
    transaction_type: str = "purchase"
    status: str = "pending"
    raw_data: Optional[Dict] = None

class TransactionDatabase:
    """SQLite database for storing real-time transactions"""
    
    def __init__(self, db_path: str = "realtime_transactions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                account_id TEXT,
                amount REAL,
                merchant TEXT,
                timestamp TEXT,
                location TEXT,
                card_last_four TEXT,
                transaction_type TEXT,
                status TEXT,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_account_timestamp 
            ON transactions(account_id, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_transaction(self, transaction: RealTimeTransaction):
        """Store a real-time transaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (transaction_id, account_id, amount, merchant, timestamp, 
                 location, card_last_four, transaction_type, status, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.transaction_id,
                transaction.account_id,
                transaction.amount,
                transaction.merchant,
                transaction.timestamp,
                transaction.location,
                transaction.card_last_four,
                transaction.transaction_type,
                transaction.status,
                json.dumps(transaction.raw_data) if transaction.raw_data else None
            ))
            conn.commit()
            logger.info(f"Stored transaction {transaction.transaction_id}")
        except Exception as e:
            logger.error(f"Error storing transaction: {e}")
        finally:
            conn.close()
    
    def get_recent_transactions(self, account_id: str, minutes: int = 60) -> List[Dict]:
        """Get recent transactions for an account"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate timestamp threshold
        threshold = datetime.now(timezone.utc).timestamp() - (minutes * 60)
        threshold_str = datetime.fromtimestamp(threshold, timezone.utc).isoformat()
        
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE account_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (account_id, threshold_str))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class RealTimeTransactionProcessor:
    """Processes transactions in real-time with fraud detection"""
    
    def __init__(self, fraud_detector=None):
        self.transaction_queue = queue.Queue()
        self.db = TransactionDatabase()
        self.fraud_detector = fraud_detector
        self.processing_thread = None
        self.running = False
        self.callbacks = []
        
        # Transaction velocity tracking
        self.velocity_tracker = defaultdict(list)
    
    def add_callback(self, callback: Callable[[RealTimeTransaction], None]):
        """Add callback function to be called when transaction is processed"""
        self.callbacks.append(callback)
    
    def start_processing(self):
        """Start the background transaction processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_transactions)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Transaction processing started")
    
    def stop_processing(self):
        """Stop the background processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Transaction processing stopped")
    
    def queue_transaction(self, transaction_data: Dict):
        """Queue a transaction for processing"""
        try:
            # Parse transaction data
            transaction = self._parse_transaction_data(transaction_data)
            self.transaction_queue.put(transaction)
            logger.info(f"Queued transaction {transaction.transaction_id}")
        except Exception as e:
            logger.error(f"Error queuing transaction: {e}")
    
    def _parse_transaction_data(self, data: Dict) -> RealTimeTransaction:
        """Parse raw transaction data into RealTimeTransaction object"""
        return RealTimeTransaction(
            transaction_id=data.get('id', f"tx_{int(time.time() * 1000)}"),
            account_id=data.get('account_id', data.get('account')),
            amount=float(data.get('amount', 0)),
            merchant=data.get('merchant', data.get('description', 'Unknown')),
            timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            location=data.get('location'),
            card_last_four=data.get('card_last_four'),
            transaction_type=data.get('type', 'purchase'),
            status=data.get('status', 'pending'),
            raw_data=data
        )
    
    def _process_transactions(self):
        """Background thread to process queued transactions"""
        while self.running:
            try:
                # Get transaction from queue with timeout
                transaction = self.transaction_queue.get(timeout=1)
                
                # Store in database immediately
                self.db.store_transaction(transaction)
                
                # Update velocity tracking
                self._update_velocity_tracking(transaction)
                
                # Run fraud detection if available
                if self.fraud_detector:
                    self._run_fraud_detection(transaction)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(transaction)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Mark task as done
                self.transaction_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
    
    def _update_velocity_tracking(self, transaction: RealTimeTransaction):
        """Update velocity tracking for the account"""
        account_id = transaction.account_id
        current_time = time.time()
        
        # Add current transaction
        self.velocity_tracker[account_id].append(current_time)
        
        # Remove transactions older than 1 hour
        self.velocity_tracker[account_id] = [
            t for t in self.velocity_tracker[account_id] 
            if current_time - t <= 3600
        ]
    
    def _run_fraud_detection(self, transaction: RealTimeTransaction):
        """Run fraud detection on the transaction"""
        try:
            # Get recent transaction history
            recent_transactions = self.db.get_recent_transactions(
                transaction.account_id, minutes=60
            )
            
            # Convert to format expected by fraud detector
            tx_data = {
                'amount': transaction.amount,
                'timestamp': transaction.timestamp,
                'merchant': transaction.merchant,
                'location': transaction.location,
                'account_id': transaction.account_id
            }
            
            # Create user profile (simplified)
            user_profile = {
                'large_amount_threshold': 1000,
                'account_id': transaction.account_id
            }
            
            # Run fraud detection
            if hasattr(self.fraud_detector, 'detect_fraud'):
                result = self.fraud_detector.detect_fraud(
                    tx_data, user_profile, recent_transactions
                )
                
                if result.get('is_fraud', False):
                    logger.warning(f"FRAUD DETECTED: {transaction.transaction_id}")
                    logger.warning(f"Risk Score: {result.get('risk_score', 0)}")
                    logger.warning(f"Triggered Rules: {result.get('triggered_rules', [])}")
        
        except Exception as e:
            logger.error(f"Fraud detection error: {e}")
    
    def get_account_velocity(self, account_id: str) -> int:
        """Get current transaction velocity for an account"""
        return len(self.velocity_tracker.get(account_id, []))

class WebhookServer:
    """Flask server to receive webhook notifications from payment processors"""
    
    def __init__(self, processor: RealTimeTransactionProcessor, port: int = 5001):
        self.app = Flask(__name__)
        self.processor = processor
        self.port = port
        self.setup_routes()
    
    def setup_routes(self):
        """Setup webhook endpoints"""
        
        @self.app.route('/webhook/stripe', methods=['POST'])
        def stripe_webhook():
            """Handle Stripe webhook"""
            try:
                data = request.get_json()
                if data.get('type') == 'payment_intent.succeeded':
                    payment_intent = data.get('data', {}).get('object', {})
                    
                    transaction_data = {
                        'id': payment_intent.get('id'),
                        'account_id': payment_intent.get('customer'),
                        'amount': payment_intent.get('amount', 0) / 100,  # Stripe uses cents
                        'merchant': payment_intent.get('description', 'Stripe Payment'),
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'type': 'purchase',
                        'status': 'completed'
                    }
                    
                    self.processor.queue_transaction(transaction_data)
                    
                return jsonify({'status': 'success'}), 200
            except Exception as e:
                logger.error(f"Stripe webhook error: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/webhook/plaid', methods=['POST'])
        def plaid_webhook():
            """Handle Plaid webhook"""
            try:
                data = request.get_json()
                
                if data.get('webhook_type') == 'TRANSACTIONS':
                    if data.get('webhook_code') == 'DEFAULT_UPDATE':
                        # Process new transactions
                        for transaction in data.get('new_transactions', []):
                            transaction_data = {
                                'id': transaction.get('transaction_id'),
                                'account_id': transaction.get('account_id'),
                                'amount': abs(float(transaction.get('amount', 0))),
                                'merchant': transaction.get('merchant_name', 'Unknown'),
                                'timestamp': transaction.get('date'),
                                'location': transaction.get('location', {}).get('address'),
                                'type': 'purchase' if transaction.get('amount', 0) > 0 else 'deposit'
                            }
                            
                            self.processor.queue_transaction(transaction_data)
                
                return jsonify({'status': 'success'}), 200
            except Exception as e:
                logger.error(f"Plaid webhook error: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/webhook/generic', methods=['POST'])
        def generic_webhook():
            """Handle generic transaction webhook"""
            try:
                data = request.get_json()
                self.processor.queue_transaction(data)
                return jsonify({'status': 'success'}), 200
            except Exception as e:
                logger.error(f"Generic webhook error: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/api/transactions/<account_id>', methods=['GET'])
        def get_transactions(account_id):
            """Get recent transactions for an account"""
            try:
                minutes = request.args.get('minutes', 60, type=int)
                transactions = self.processor.db.get_recent_transactions(account_id, minutes)
                return jsonify({'transactions': transactions}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/api/velocity/<account_id>', methods=['GET'])
        def get_velocity(account_id):
            """Get current transaction velocity for an account"""
            try:
                velocity = self.processor.get_account_velocity(account_id)
                return jsonify({'account_id': account_id, 'velocity': velocity}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200
    
    def run(self, debug=False):
        """Run the webhook server"""
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

# Example usage and integration functions
def integrate_with_fraud_detection():
    """Example integration with existing fraud detection system"""
    try:
        # Import the existing fraud detection system
        from AIRuleBasedFraudDetection import EnhancedFraudDetectionSystem
        fraud_detector = EnhancedFraudDetectionSystem()
        
        # Create real-time processor
        processor = RealTimeTransactionProcessor(fraud_detector)
        
        # Add custom callback for high-risk transactions
        def fraud_alert_callback(transaction: RealTimeTransaction):
            if transaction.amount > 5000:
                logger.warning(f"HIGH VALUE TRANSACTION: {transaction.transaction_id} - ${transaction.amount}")
        
        processor.add_callback(fraud_alert_callback)
        
        return processor
    
    except ImportError:
        logger.warning("Could not import fraud detection system, using processor without fraud detection")
        return RealTimeTransactionProcessor()

if __name__ == "__main__":
    # Initialize the real-time system
    processor = integrate_with_fraud_detection()
    processor.start_processing()
    
    # Start webhook server
    webhook_server = WebhookServer(processor)
    
    try:
        print("Starting Real-time Transaction Tracking System...")
        print(f"Webhook server running on port {webhook_server.port}")
        print("Available endpoints:")
        print("  POST /webhook/stripe - Stripe webhooks")
        print("  POST /webhook/plaid - Plaid webhooks") 
        print("  POST /webhook/generic - Generic transaction webhooks")
        print("  GET /api/transactions/<account_id> - Get recent transactions")
        print("  GET /api/velocity/<account_id> - Get transaction velocity")
        print("  GET /health - Health check")
        
        webhook_server.run()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        processor.stop_processing()