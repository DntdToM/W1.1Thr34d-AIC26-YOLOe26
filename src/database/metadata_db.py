"""SQLite database wrapper for managing frame-level metadata."""

import sqlite3
from typing import Dict, Any, List, Optional


class MetadataDB:
    """SQLite Database wrapper for video and frame metadata persistence."""

    def __init__(self, db_path: str = "processed_data/3_metadata/metadata.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema and set WAL journal mode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frame_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    shot_id INTEGER,
                    frame_type TEXT,
                    frame_idx INTEGER NOT NULL,
                    timestamp_ms INTEGER,
                    frame_path TEXT UNIQUE,
                    ocr_text TEXT,
                    asr_text TEXT,
                    detected_objects TEXT,
                    context_summary TEXT
                )
            """)
            cursor.execute("PRAGMA table_info(frame_metadata)")
            columns = [col[1] for col in cursor.fetchall()]
            if "shot_id" not in columns:
                cursor.execute("ALTER TABLE frame_metadata ADD COLUMN shot_id INTEGER")
            if "frame_type" not in columns:
                cursor.execute("ALTER TABLE frame_metadata ADD COLUMN frame_type TEXT")
            if "context_summary" not in columns:
                cursor.execute("ALTER TABLE frame_metadata ADD COLUMN context_summary TEXT")

            conn.commit()

    def insert_frame_metadata(self, data: Dict[str, Any]):
        """Insert or replace frame metadata entry into the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO frame_metadata 
                (video_id, shot_id, frame_type, frame_idx, timestamp_ms, frame_path, ocr_text, asr_text, detected_objects, context_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("video_id"),
                data.get("shot_id", 0),
                data.get("frame_type", ""),
                data.get("frame_idx", 0),
                data.get("timestamp_ms", 0),
                data.get("frame_path", ""),
                data.get("ocr_text", ""),
                data.get("asr_text", ""),
                data.get("detected_objects", ""),
                data.get("context_summary", "")
            ))
            conn.commit()

    def get_by_frame_path(self, frame_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata record by frame path."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM frame_metadata WHERE frame_path = ?", (frame_path,))
            row = cursor.fetchone()
            if row:
                return self._parse_row(row)
        return None

    def get_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve metadata record by primary ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM frame_metadata WHERE id = ?", (frame_id,))
            row = cursor.fetchone()
            if row:
                return self._parse_row(row)
        return None

    def _parse_row(self, row: tuple) -> Dict[str, Any]:
        """Parse database row tuple into a dictionary mapping."""
        return {
            "id": row[0],
            "video_id": row[1],
            "shot_id": row[2] if len(row) > 2 else 0,
            "frame_type": row[3] if len(row) > 3 else "",
            "frame_idx": row[4] if len(row) > 4 else 0,
            "timestamp_ms": row[5] if len(row) > 5 else 0,
            "frame_path": row[6] if len(row) > 6 else "",
            "ocr_text": row[7] if len(row) > 7 else "",
            "asr_text": row[8] if len(row) > 8 else "",
            "detected_objects": row[9] if len(row) > 9 else "",
            "context_summary": row[10] if len(row) > 10 else ""
        }
