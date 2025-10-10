import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import pickle
from io import BytesIO

class ChatPersistence:
    """Handles SQLite persistence for chat history and sessions."""
    
    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                dataset_name TEXT,
                browser_id TEXT
            )
        """)
        
        # Migration: Add browser_id column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN browser_id TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                query_type TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Session state table (for storing dataframes and other state)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                original_df BLOB,
                current_df BLOB,
                approved_df BLOB,
                phase TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_name: Optional[str] = None, dataset_name: Optional[str] = None, browser_id: Optional[str] = None) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        if not session_name:
            session_name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, session_name, created_at, last_updated, dataset_name, browser_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, session_name, datetime.now(), datetime.now(), dataset_name, browser_id))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def get_all_sessions(self, browser_id: Optional[str] = None) -> List[Dict]:
        """Get all sessions for a specific browser, ordered by last updated."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if browser_id:
            cursor.execute("""
                SELECT session_id, session_name, created_at, last_updated, dataset_name
                FROM sessions
                WHERE browser_id = ?
                ORDER BY last_updated DESC
            """, (browser_id,))
        else:
            cursor.execute("""
                SELECT session_id, session_name, created_at, last_updated, dataset_name
                FROM sessions
                ORDER BY last_updated DESC
            """)
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'session_name': row[1],
                'created_at': row[2],
                'last_updated': row[3],
                'dataset_name': row[4]
            })
        
        conn.close()
        return sessions
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, session_name, created_at, last_updated, dataset_name, browser_id
            FROM sessions
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'session_id': row[0],
                'session_name': row[1],
                'created_at': row[2],
                'last_updated': row[3],
                'dataset_name': row[4],
                'browser_id': row[5]
            }
        return None
    
    def update_session_name(self, session_id: str, new_name: str):
        """Update a session's name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET session_name = ?, last_updated = ?
            WHERE session_id = ?
        """, (new_name, datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    def save_chat_message(self, session_id: str, query: str, query_type: str):
        """Save a chat message to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chat_history (session_id, query, query_type, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, query, query_type, datetime.now()))
        
        # Update session's last_updated timestamp
        cursor.execute("""
            UPDATE sessions
            SET last_updated = ?
            WHERE session_id = ?
        """, (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get chat history for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute("""
                SELECT query, query_type, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
                SELECT query, query_type, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'query': row[0],
                'type': row[1],
                'timestamp': row[2]
            })
        
        conn.close()
        return history
    
    def save_session_state(self, session_id: str, original_df: Optional[pd.DataFrame] = None, 
                          current_df: Optional[pd.DataFrame] = None, approved_df: Optional[pd.DataFrame] = None,
                          phase: Optional[str] = None):
        """Save dataframes and phase to session state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert dataframes to pickle bytes for storage
        original_blob = pickle.dumps(original_df) if original_df is not None else None
        current_blob = pickle.dumps(current_df) if current_df is not None else None
        approved_blob = pickle.dumps(approved_df) if approved_df is not None else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO session_state 
            (session_id, original_df, current_df, approved_df, phase)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, original_blob, current_blob, approved_blob, phase))
        
        conn.commit()
        conn.close()
    
    def load_session_state(self, session_id: str) -> Optional[Dict]:
        """Load dataframes and phase from session state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT original_df, current_df, approved_df, phase
            FROM session_state
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'original_df': pickle.loads(row[0]) if row[0] else None,
                'current_df': pickle.loads(row[1]) if row[1] else None,
                'approved_df': pickle.loads(row[2]) if row[2] else None,
                'phase': row[3]
            }
        return None
    
    def delete_session(self, session_id: str):
        """Delete a session and all its data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM session_state WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_latest_session_id(self) -> Optional[str]:
        """Get the most recently updated session ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id
            FROM sessions
            ORDER BY last_updated DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None
