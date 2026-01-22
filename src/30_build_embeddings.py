"""
Embeddings Builder Module

Builds vector embeddings for F1 race control messages using:
1. sentence-transformers (preferred, local model)
2. Deterministic hash-based fallback if model unavailable

Loads embeddings into f1_messages table with VECTOR column.
Loads connection settings from .env file if present.
"""

import os
import sys
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import oracledb

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Builds embeddings for F1 race control messages."""

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, user: str, password: str, dsn: str):
        """Initialize with Oracle connection parameters."""
        self.user = user
        self.password = password
        self.dsn = dsn
        self.connection = None
        self.model = None
        self.use_fallback = False
        self.project_root = Path(__file__).parent.parent

    def connect(self):
        """Establish Oracle connection."""
        logger.info(f"Connecting to Oracle as {self.user}@{self.dsn}...")
        self.connection = oracledb.connect(
            user=self.user,
            password=self.password,
            dsn=self.dsn
        )
        logger.info("Connected successfully")

    def close(self):
        """Close Oracle connection."""
        if self.connection:
            self.connection.close()
            logger.info("Connection closed")

    def _load_model(self):
        """Load sentence-transformers model or fall back to hash-based."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.MODEL_NAME}...")
            self.model = SentenceTransformer(self.MODEL_NAME)
            logger.info("Model loaded successfully")
            self.use_fallback = False
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback embeddings")
            self.use_fallback = True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using fallback embeddings")
            self.use_fallback = True

    def _hash_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic embedding from text using hash.
        This is a fallback when sentence-transformers is unavailable.
        """
        # Create deterministic seed from text
        text_hash = hashlib.sha256(text.encode("utf-8")).digest()

        # Use hash bytes as seed for reproducible random numbers
        seed = int.from_bytes(text_hash[:4], "big")
        rng = np.random.default_rng(seed)

        # Generate normalized random vector
        embedding = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts to embeddings."""
        if self.use_fallback:
            return [self._hash_embedding(t) for t in texts]
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

    def _get_session_id(self, year: int, gp: str, session: str) -> str:
        """Build session_id matching the exporter's slug logic."""
        gp_slug = re.sub(r"\s+", "_", gp.strip())
        gp_slug = re.sub(r"[^A-Za-z0-9_\-]", "", gp_slug)
        session_slug = re.sub(r"\s+", "_", session.strip())
        session_slug = re.sub(r"[^A-Za-z0-9_\-]", "", session_slug)
        return f"{year}_{gp_slug}_{session_slug}"

    def _get_bronze_path(self) -> Path:
        """Get path to bronze layer."""
        return self.project_root / "lakehouse" / "01_bronze"

    def _read_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Read a CSV file from bronze layer."""
        filepath = self._get_bronze_path() / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        logger.info(f"Reading {filename}...")
        return pd.read_csv(filepath)

    def _get_event_id(self, session_id: str) -> Optional[int]:
        """Get event_id for a session."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT event_id FROM dim_event WHERE session_id = :1",
            [session_id]
        )
        row = cursor.fetchone()
        cursor.close()

        if row:
            return row[0]
        return None

    def _clear_existing_messages(self, event_id: int):
        """Clear existing messages for this event."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM f1_messages WHERE event_id = :1", [event_id])
        deleted = cursor.rowcount
        self.connection.commit()
        cursor.close()
        if deleted > 0:
            logger.info(f"Cleared {deleted} existing messages for event_id={event_id}")

    def _vector_to_string(self, embedding: List[float]) -> str:
        """Convert embedding list to Oracle VECTOR string format."""
        return "[" + ",".join(str(x) for x in embedding) + "]"

    def build_and_load(self, year: int, gp: str, session: str, batch_size: int = 100):
        """Build embeddings and load into Oracle."""
        session_id = self._get_session_id(year, gp, session)

        logger.info("=" * 60)
        logger.info(f"Building embeddings for session: {session_id}")
        logger.info("=" * 60)

        # Get event_id
        event_id = self._get_event_id(session_id)
        if not event_id:
            logger.error(f"Event not found: {session_id}")
            logger.error("Run 20_load_oracle.py first to load the base data")
            return False

        # Load model
        self._load_model()
        if self.use_fallback:
            logger.warning("Using deterministic hash-based fallback embeddings")
        else:
            logger.info(f"Using sentence-transformers model: {self.MODEL_NAME}")

        # Read race control messages
        messages_file = f"race_control_messages_{session_id}.csv"
        df = self._read_csv(messages_file)
        if df is None:
            logger.error(f"Messages file not found: {messages_file}")
            return False

        logger.info(f"Processing {len(df)} messages...")

        # Clear existing messages
        self._clear_existing_messages(event_id)

        # Prepare texts for embedding
        texts = []
        records = []

        for _, row in df.iterrows():
            message_text = str(row.get("Message", "")).strip()
            category = str(row.get("Category", "")).strip()

            # Combine message and category for richer embedding
            combined_text = f"{category}: {message_text}" if category else message_text
            texts.append(combined_text)

            # Parse timestamp
            msg_ts = None
            msg_time_sec = None
            time_str = row.get("Time")
            if pd.notna(time_str):
                try:
                    msg_ts = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
                except ValueError:
                    pass
                # Also try to get session time if available
                if pd.notna(row.get("SessionTime")):
                    msg_time_sec = float(row["SessionTime"])

            records.append({
                "event_id": event_id,
                "msg_ts": msg_ts,
                "msg_time_sec": msg_time_sec,
                "category": category if category else None,
                "flag": row.get("Flag") if pd.notna(row.get("Flag")) else None,
                "scope": row.get("Scope") if pd.notna(row.get("Scope")) else None,
                "message_text": message_text if message_text else None,
            })

        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
                logger.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} messages")

        # Insert into Oracle
        logger.info("Loading messages with embeddings into Oracle...")
        cursor = self.connection.cursor()
        loaded = 0

        for i, (record, embedding) in enumerate(zip(records, all_embeddings)):
            try:
                # Convert embedding to Oracle VECTOR string
                embedding_str = self._vector_to_string(embedding)

                cursor.execute(f"""
                    INSERT INTO f1_messages (
                        event_id, msg_ts, msg_time_sec, category, flag, scope,
                        message_text, embedding
                    ) VALUES (
                        :1, :2, :3, :4, :5, :6, :7,
                        TO_VECTOR(:8, {self.EMBEDDING_DIM}, FLOAT32)
                    )
                """, [
                    record["event_id"],
                    record["msg_ts"],
                    record["msg_time_sec"],
                    record["category"],
                    record["flag"],
                    record["scope"],
                    record["message_text"],
                    embedding_str,
                ])
                loaded += 1
            except oracledb.DatabaseError as e:
                error, = e.args
                # If VECTOR type not supported, try without embedding
                if error.code in (902, 904, 12704):  # Invalid datatype
                    try:
                        cursor.execute("""
                            INSERT INTO f1_messages (
                                event_id, msg_ts, msg_time_sec, category, flag, scope,
                                message_text
                            ) VALUES (:1, :2, :3, :4, :5, :6, :7)
                        """, [
                            record["event_id"],
                            record["msg_ts"],
                            record["msg_time_sec"],
                            record["category"],
                            record["flag"],
                            record["scope"],
                            record["message_text"],
                        ])
                        loaded += 1
                        if loaded == 1:
                            logger.warning("VECTOR type not supported, loading messages without embeddings")
                    except oracledb.DatabaseError as e2:
                        logger.warning(f"Failed to insert message {i}: {e2}")
                else:
                    logger.warning(f"Failed to insert message {i}: {e}")

            if loaded % 100 == 0:
                self.connection.commit()

        self.connection.commit()
        cursor.close()

        logger.info("=" * 60)
        logger.info(f"Loaded {loaded} messages with embeddings")
        logger.info(f"Embedding method: {'sentence-transformers' if not self.use_fallback else 'hash-based fallback'}")
        logger.info("=" * 60)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and load embeddings for F1 race control messages"
    )
    parser.add_argument(
        "--year", type=int, default=2024,
        help="Season year (default: 2024)"
    )
    parser.add_argument(
        "--gp", type=str, default="Singapore",
        help="Grand Prix name (default: Singapore)"
    )
    parser.add_argument(
        "--session", type=str, default="R",
        help="Session type: R=Race, Q=Qualifying, FP1/FP2/FP3 (default: R)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size for embedding generation (default: 100)"
    )

    args = parser.parse_args()

    # Get Oracle connection params
    user = os.environ.get("ORA_USER")
    password = os.environ.get("ORA_PASSWORD")
    dsn = os.environ.get("ORA_DSN")

    if not all([user, password, dsn]):
        print("ERROR: Missing Oracle connection environment variables")
        print("Required: ORA_USER, ORA_PASSWORD, ORA_DSN")
        sys.exit(1)

    builder = EmbeddingBuilder(user, password, dsn)

    try:
        builder.connect()
        success = builder.build_and_load(
            year=args.year,
            gp=args.gp,
            session=args.session,
            batch_size=args.batch_size
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Embedding build failed: {e}")
        sys.exit(1)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
