"""
Database Module for Sentiment Analysis System.
Manages SQLite storage for predictions history, statistics, and audit logs.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "sentiment.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create table structure if not present."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_type TEXT NOT NULL,
                input_content TEXT,
                emotion TEXT,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print("[Database] Initialized SQLite database successfully.")

def save_prediction(input_type, input_content, emotion, sentiment, confidence):
    """Inserts a new prediction record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (input_type, input_content, emotion, sentiment, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            input_type,
            input_content or "-",
            emotion or "-",
            sentiment,
            round(confidence, 1),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        return cursor.lastrowid

def get_history(limit=100, filter_type=None, filter_sentiment=None, search=None):
    """Fetch prediction logs with optional filtering."""
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if filter_type and filter_type.upper() != "ALL":
        query += " AND input_type = ?"
        params.append(filter_type.upper())
        
    if filter_sentiment and filter_sentiment.capitalize() != "All":
        query += " AND sentiment = ?"
        params.append(filter_sentiment.capitalize())
        
    if search:
        query += " AND (input_content LIKE ? OR emotion LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term])
        
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def delete_prediction(pred_id):
    """Delete single record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        conn.commit()
        return cursor.rowcount > 0

def clear_history():
    """Truncate prediction table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()

def get_statistics():
    """Compute comprehensive metrics for dashboard analytics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_count = cursor.fetchone()[0]
        
        # By sentiment
        cursor.execute("""
            SELECT sentiment, COUNT(*) as cnt 
            FROM predictions 
            GROUP BY sentiment
        """)
        sentiment_counts = {row["sentiment"]: row["cnt"] for row in cursor.fetchall()}
        
        # By input type
        cursor.execute("""
            SELECT input_type, COUNT(*) as cnt 
            FROM predictions 
            GROUP BY input_type
        """)
        type_counts = {row["input_type"]: row["cnt"] for row in cursor.fetchall()}
        
        # By emotion (for face & multimodal)
        cursor.execute("""
            SELECT emotion, COUNT(*) as cnt 
            FROM predictions 
            WHERE emotion != '-' AND emotion != 'No Face Detected'
            GROUP BY emotion
        """)
        emotion_counts = {row["emotion"]: row["cnt"] for row in cursor.fetchall()}
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_conf = cursor.fetchone()[0] or 0.0
        
        return {
            "total_predictions": total_count,
            "sentiment_distribution": {
                "Positive": sentiment_counts.get("Positive", 0),
                "Negative": sentiment_counts.get("Negative", 0),
                "Neutral": sentiment_counts.get("Neutral", 0)
            },
            "type_distribution": type_counts,
            "emotion_distribution": emotion_counts,
            "average_confidence": round(avg_conf, 1)
        }

if __name__ == "__main__":
    init_db()
    pid = save_prediction("TEXT", "Awesome movie!", "-", "Positive", 98.2)
    print("Inserted test record ID:", pid)
    print("History:", get_history(5))
    print("Stats:", get_statistics())
