"""
Use Cases Runner Module

Runs a curated subset of SQL use cases from 20_use_cases.sql and displays results.
Demonstrates Oracle 23ai Converged Database capabilities.
Loads connection settings from .env file if present.
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
import oracledb

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UseCaseRunner:
    """Runs F1 demonstration use cases against Oracle."""

    def __init__(self, user: str, password: str, dsn: str):
        """Initialize with Oracle connection parameters."""
        self.user = user
        self.password = password
        self.dsn = dsn
        self.connection = None
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

    def _print_results(self, cursor, title: str, max_rows: int = 20):
        """Print query results in a formatted table."""
        print()
        print("=" * 80)
        print(f"  {title}")
        print("=" * 80)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(max_rows)

        if not rows:
            print("  (No results)")
            return

        # Calculate column widths
        widths = [len(col) for col in columns]
        for row in rows:
            for i, val in enumerate(row):
                val_str = str(val) if val is not None else "NULL"
                widths[i] = max(widths[i], min(len(val_str), 30))

        # Print header
        header = " | ".join(col[:widths[i]].ljust(widths[i]) for i, col in enumerate(columns))
        print(f"  {header}")
        print("  " + "-" * len(header))

        # Print rows
        for row in rows:
            row_str = " | ".join(
                (str(val) if val is not None else "NULL")[:widths[i]].ljust(widths[i])
                for i, val in enumerate(row)
            )
            print(f"  {row_str}")

        remaining = cursor.fetchall()
        total = len(rows) + len(remaining)
        if remaining:
            print(f"  ... ({total} total rows, showing first {len(rows)})")
        print()

    def run_relational_analytics(self):
        """Run relational analytics use cases."""
        cursor = self.connection.cursor()

        # 1.1 Pace Degradation Analysis
        try:
            cursor.execute("""
                SELECT
                    d.driver_code,
                    l.stint_number,
                    c.compound_name AS compound,
                    l.lap_number,
                    ROUND(l.lap_time_sec, 3) AS lap_time_sec,
                    ROUND(l.lap_time_sec - LAG(l.lap_time_sec) OVER (
                        PARTITION BY l.event_id, l.driver_id, l.stint_number
                        ORDER BY l.lap_number
                    ), 3) AS delta_to_prev
                FROM fact_lap l
                JOIN dim_driver d ON l.driver_id = d.driver_id
                LEFT JOIN dim_compound c ON l.compound_id = c.compound_id
                WHERE l.is_accurate = 1 AND l.lap_time_sec IS NOT NULL
                ORDER BY d.driver_code, l.stint_number, l.lap_number
                FETCH FIRST 15 ROWS ONLY
            """)
            self._print_results(cursor, "1.1 RELATIONAL: Pace Degradation per Stint (LAG window function)")
        except oracledb.DatabaseError as e:
            logger.warning(f"Pace degradation query failed: {e}")

        # 1.2 Driver Consistency Ranking
        try:
            cursor.execute("""
                SELECT
                    d.driver_code,
                    d.full_name,
                    COUNT(*) AS valid_laps,
                    ROUND(AVG(l.lap_time_sec), 3) AS avg_lap_time,
                    ROUND(STDDEV(l.lap_time_sec), 4) AS lap_time_stddev,
                    RANK() OVER (ORDER BY STDDEV(l.lap_time_sec)) AS consistency_rank
                FROM fact_lap l
                JOIN dim_driver d ON l.driver_id = d.driver_id
                WHERE l.is_accurate = 1
                  AND l.lap_time_sec IS NOT NULL
                  AND l.lap_time_sec < 120
                GROUP BY d.driver_id, d.driver_code, d.full_name
                ORDER BY consistency_rank
                FETCH FIRST 10 ROWS ONLY
            """)
            self._print_results(cursor, "1.2 RELATIONAL: Driver Consistency Ranking (STDDEV, RANK)")
        except oracledb.DatabaseError as e:
            logger.warning(f"Consistency ranking query failed: {e}")

        cursor.close()

    def run_json_queries(self):
        """Run JSON use cases."""
        cursor = self.connection.cursor()

        # 2.1 Extract Race Control Messages from JSON
        try:
            cursor.execute("""
                SELECT
                    d.doc_id,
                    JSON_VALUE(d.payload, '$.Category') AS category,
                    JSON_VALUE(d.payload, '$.Flag') AS flag,
                    SUBSTR(JSON_VALUE(d.payload, '$.Message'), 1, 50) AS message_text,
                    JSON_VALUE(d.payload, '$.Lap' RETURNING NUMBER) AS lap
                FROM f1_raw_documents d
                WHERE d.doc_type = 'race_control_message'
                ORDER BY d.doc_id
                FETCH FIRST 15 ROWS ONLY
            """)
            self._print_results(cursor, "2.1 JSON: Extract Race Control Messages (JSON_VALUE)")
        except oracledb.DatabaseError as e:
            logger.warning(f"JSON extraction query failed: {e}")

        # 2.2 Session Info JSON
        try:
            cursor.execute("""
                SELECT
                    JSON_VALUE(d.payload, '$.session_id') AS session_id,
                    JSON_VALUE(d.payload, '$.gp') AS gp_name,
                    JSON_VALUE(d.payload, '$.session_name') AS session_name,
                    JSON_VALUE(d.payload, '$.data.Meeting.Location') AS location
                FROM f1_raw_documents d
                WHERE d.doc_type = 'session_info'
            """)
            self._print_results(cursor, "2.2 JSON: Session Info Extraction (Nested JSON Navigation)")
        except oracledb.DatabaseError as e:
            logger.warning(f"Session info JSON query failed: {e}")

        cursor.close()

    def run_time_series_queries(self):
        """Run time series use cases."""
        cursor = self.connection.cursor()

        # 3.1 Heavy Braking Events
        try:
            cursor.execute("""
                SELECT
                    d.driver_code,
                    COUNT(*) AS braking_events,
                    ROUND(AVG(t.speed), 1) AS avg_speed_at_brake,
                    ROUND(MAX(t.speed), 1) AS max_speed_at_brake
                FROM fact_telemetry t
                JOIN dim_driver d ON t.driver_id = d.driver_id
                WHERE t.brake = 1 AND t.speed > 150
                GROUP BY d.driver_id, d.driver_code
                ORDER BY braking_events DESC
                FETCH FIRST 10 ROWS ONLY
            """)
            self._print_results(cursor, "3.1 TIME SERIES: Heavy Braking Events per Driver")
        except oracledb.DatabaseError as e:
            logger.warning(f"Braking events query failed: {e}")

        # 3.2 Weather Changes
        try:
            cursor.execute("""
                SELECT
                    ROUND(w.time_sec / 60, 0) AS race_minute,
                    ROUND(w.air_temp, 1) AS air_temp_c,
                    ROUND(w.track_temp, 1) AS track_temp_c,
                    ROUND(w.humidity, 0) AS humidity_pct,
                    w.rainfall
                FROM fact_weather w
                WHERE w.time_sec > 3400
                ORDER BY w.time_sec
                FETCH FIRST 10 ROWS ONLY
            """)
            self._print_results(cursor, "3.2 TIME SERIES: Weather Timeline During Race")
        except oracledb.DatabaseError as e:
            logger.warning(f"Weather timeline query failed: {e}")

        # 3.3 Session Status Timeline
        try:
            cursor.execute("""
                SELECT
                    ROUND(time_sec / 60, 1) AS session_minute,
                    status
                FROM fact_session_status
                ORDER BY time_sec
            """)
            self._print_results(cursor, "3.3 TIME SERIES: Session Status Timeline")
        except oracledb.DatabaseError as e:
            logger.warning(f"Session status query failed: {e}")

        # 3.4 Time Series Gap Filling with MODEL clause
        try:
            cursor.execute("""
                WITH weather_minutes AS (
                    SELECT
                        event_id,
                        TRUNC(time_sec / 60) AS minute_bucket,
                        ROUND(AVG(air_temp), 1) AS air_temp,
                        ROUND(AVG(track_temp), 1) AS track_temp
                    FROM fact_weather
                    GROUP BY event_id, TRUNC(time_sec / 60)
                )
                SELECT minute_bucket, air_temp, track_temp, is_interpolated
                FROM weather_minutes
                MODEL
                    PARTITION BY (event_id)
                    DIMENSION BY (minute_bucket)
                    MEASURES (
                        air_temp,
                        track_temp,
                        0 AS is_interpolated
                    )
                    RULES AUTOMATIC ORDER (
                        air_temp[FOR minute_bucket FROM 57 TO 62 INCREMENT 1] =
                            CASE WHEN air_temp[CV()] IS NULL
                                 THEN ROUND((air_temp[CV()-1] + air_temp[CV()+1]) / 2, 1)
                                 ELSE air_temp[CV()]
                            END,
                        track_temp[FOR minute_bucket FROM 57 TO 62 INCREMENT 1] =
                            CASE WHEN track_temp[CV()] IS NULL
                                 THEN ROUND((track_temp[CV()-1] + track_temp[CV()+1]) / 2, 1)
                                 ELSE track_temp[CV()]
                            END,
                        is_interpolated[FOR minute_bucket FROM 57 TO 62 INCREMENT 1] =
                            CASE WHEN air_temp[CV()] IS NULL THEN 1 ELSE 0 END
                    )
                WHERE minute_bucket BETWEEN 57 AND 62
                ORDER BY minute_bucket
            """)
            self._print_results(cursor, "3.4 TIME SERIES: Gap Filling with MODEL Clause (interpolation)")
        except oracledb.DatabaseError as e:
            logger.warning(f"Gap filling query failed: {e}")

        # 3.5 Time Series Continuous Timeline Generation
        try:
            cursor.execute("""
                WITH time_range AS (
                    SELECT MIN(TRUNC(time_sec)) AS min_sec, MAX(TRUNC(time_sec)) AS max_sec
                    FROM fact_weather
                ),
                continuous_timeline AS (
                    SELECT min_sec + (LEVEL - 1) * 60 AS time_sec
                    FROM time_range
                    CONNECT BY LEVEL <= (max_sec - min_sec) / 60 + 1
                    AND LEVEL <= 20
                ),
                weather_sampled AS (
                    SELECT
                        TRUNC(time_sec / 60) * 60 AS time_bucket,
                        ROUND(AVG(air_temp), 1) AS air_temp,
                        ROUND(AVG(track_temp), 1) AS track_temp
                    FROM fact_weather
                    GROUP BY TRUNC(time_sec / 60) * 60
                )
                SELECT
                    ROUND(ct.time_sec / 60, 0) AS race_minute,
                    COALESCE(ws.air_temp, LAG(ws.air_temp IGNORE NULLS) OVER (ORDER BY ct.time_sec)) AS air_temp_filled,
                    COALESCE(ws.track_temp, LAG(ws.track_temp IGNORE NULLS) OVER (ORDER BY ct.time_sec)) AS track_temp_filled,
                    CASE WHEN ws.time_bucket IS NULL THEN 'FILLED' ELSE 'ACTUAL' END AS data_source
                FROM continuous_timeline ct
                LEFT JOIN weather_sampled ws ON ct.time_sec = ws.time_bucket
                ORDER BY ct.time_sec
                FETCH FIRST 15 ROWS ONLY
            """)
            self._print_results(cursor, "3.5 TIME SERIES: Continuous Timeline with Forward Fill")
        except oracledb.DatabaseError as e:
            logger.warning(f"Continuous timeline query failed: {e}")

        cursor.close()

    def run_spatial_queries(self):
        """Run spatial use cases."""
        cursor = self.connection.cursor()

        # 4.1 Find points near a reference location
        try:
            cursor.execute("""
                SELECT
                    d.driver_code,
                    ts.session_time_sec,
                    ts.speed,
                    ts.brake,
                    ROUND(ts.geom.SDO_POINT.X, 1) AS x,
                    ROUND(ts.geom.SDO_POINT.Y, 1) AS y
                FROM telemetry_spatial ts
                JOIN dim_driver d ON ts.driver_id = d.driver_id
                WHERE SDO_WITHIN_DISTANCE(
                    ts.geom,
                    SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(850, 1200, NULL), NULL, NULL),
                    'distance=50 unit=M'
                ) = 'TRUE'
                ORDER BY ts.session_time_sec
                FETCH FIRST 10 ROWS ONLY
            """)
            self._print_results(cursor, "4.1 SPATIAL: Telemetry Points Near Reference Location (SDO_WITHIN_DISTANCE)")
        except oracledb.DatabaseError as e:
            logger.warning(f"Spatial proximity query failed: {e}")

        # 4.2 Track Sector Analysis by Spatial Region
        try:
            cursor.execute("""
                SELECT
                    d.driver_code,
                    CASE
                        WHEN ts.geom.SDO_POINT.X > 800 AND ts.geom.SDO_POINT.Y > 1000 THEN 'SECTOR_1'
                        WHEN ts.geom.SDO_POINT.X < 200 AND ts.geom.SDO_POINT.Y > 500 THEN 'SECTOR_2'
                        ELSE 'SECTOR_3'
                    END AS track_sector,
                    COUNT(*) AS sample_count,
                    ROUND(AVG(ts.speed), 1) AS avg_speed,
                    ROUND(MAX(ts.speed), 1) AS max_speed
                FROM telemetry_spatial ts
                JOIN dim_driver d ON ts.driver_id = d.driver_id
                GROUP BY d.driver_id, d.driver_code,
                    CASE
                        WHEN ts.geom.SDO_POINT.X > 800 AND ts.geom.SDO_POINT.Y > 1000 THEN 'SECTOR_1'
                        WHEN ts.geom.SDO_POINT.X < 200 AND ts.geom.SDO_POINT.Y > 500 THEN 'SECTOR_2'
                        ELSE 'SECTOR_3'
                    END
                ORDER BY d.driver_code, track_sector
                FETCH FIRST 15 ROWS ONLY
            """)
            self._print_results(cursor, "4.2 SPATIAL: Track Sector Analysis by Coordinates")
        except oracledb.DatabaseError as e:
            logger.warning(f"Track sector spatial query failed: {e}")

        cursor.close()

    def run_graph_queries(self):
        """Run graph use cases."""
        cursor = self.connection.cursor()

        # 5.1 Find Teammates using Graph
        try:
            cursor.execute("""
                SELECT
                    driver1_code,
                    driver1_name,
                    team_name,
                    driver2_code,
                    driver2_name
                FROM GRAPH_TABLE (f1_graph
                    MATCH (d1 IS driver) -[dt1 IS drives_for]-> (t IS team) <-[dt2 IS drives_for]- (d2 IS driver)
                    WHERE d1.driver_id < d2.driver_id
                    COLUMNS (
                        d1.driver_code AS driver1_code,
                        d1.full_name AS driver1_name,
                        t.team_name AS team_name,
                        d2.driver_code AS driver2_code,
                        d2.full_name AS driver2_name
                    )
                )
                ORDER BY team_name
            """)
            self._print_results(cursor, "5.1 GRAPH: Find Teammates (GRAPH_TABLE pattern matching)")
        except oracledb.DatabaseError as e:
            logger.warning(f"Graph teammates query failed: {e}")
            # Fallback to relational query
            try:
                cursor.execute("""
                    SELECT
                        d1.driver_code AS driver1_code,
                        d1.full_name AS driver1_name,
                        t.team_name,
                        d2.driver_code AS driver2_code,
                        d2.full_name AS driver2_name
                    FROM bridge_driver_team bdt1
                    JOIN bridge_driver_team bdt2
                        ON bdt1.event_id = bdt2.event_id
                        AND bdt1.team_id = bdt2.team_id
                        AND bdt1.driver_id < bdt2.driver_id
                    JOIN dim_team t ON bdt1.team_id = t.team_id
                    JOIN dim_driver d1 ON bdt1.driver_id = d1.driver_id
                    JOIN dim_driver d2 ON bdt2.driver_id = d2.driver_id
                    ORDER BY t.team_name
                """)
                self._print_results(cursor, "5.1 GRAPH (fallback): Find Teammates (relational equivalent)")
            except oracledb.DatabaseError:
                pass

        # 5.2 Driver-Team Network
        try:
            cursor.execute("""
                SELECT
                    driver_code,
                    driver_name,
                    team_name,
                    final_position,
                    points
                FROM GRAPH_TABLE (f1_graph
                    MATCH (d IS driver) -[dt IS drives_for]-> (t IS team),
                          (d) -[p IS participates_in]-> (e IS event)
                    COLUMNS (
                        d.driver_code AS driver_code,
                        d.full_name AS driver_name,
                        t.team_name AS team_name,
                        p.final_position AS final_position,
                        p.points AS points
                    )
                )
                ORDER BY final_position NULLS LAST
            """)
            self._print_results(cursor, "5.2 GRAPH: Driver-Team-Event Network")
        except oracledb.DatabaseError as e:
            logger.warning(f"Graph network query failed: {e}")

        cursor.close()

    def run_vector_queries(self):
        """
        Run vector similarity search use cases using Oracle VECTOR_DISTANCE.

        Demonstrates TRUE semantic search using Oracle's native vector capabilities:
        - VECTOR_DISTANCE function with COSINE distance
        - Query embedding generated via sentence-transformers
        - Hybrid queries combining vector + SQL filters
        """
        cursor = self.connection.cursor()

        # Check if f1_messages table has embeddings
        try:
            cursor.execute("SELECT COUNT(*) FROM f1_messages WHERE embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
        except oracledb.DatabaseError as e:
            logger.warning(f"Could not check embeddings: {e}")
            cursor.close()
            return

        if embedding_count == 0:
            logger.warning("No embeddings found in f1_messages - run 30_build_embeddings.py first")
            cursor.close()
            return

        print()
        print("=" * 80)
        print("  6.1 VECTOR: Message Categories (baseline)")
        print("=" * 80)

        # Show category distribution from f1_messages
        cursor.execute("""
            SELECT category, COUNT(*) as cnt
            FROM f1_messages
            WHERE embedding IS NOT NULL
            GROUP BY category
            ORDER BY cnt DESC
        """)
        rows = cursor.fetchall()

        print(f"  {'CATEGORY':<20} | {'COUNT':>6}")
        print("  " + "-" * 30)
        for category, count in rows:
            cat = category or 'Unknown'
            print(f"  {cat:<20} | {count:>6}")
        print()
        print(f"  Total messages with embeddings: {embedding_count}")
        print()

        # Load sentence-transformers model for query embedding
        logger.info("Loading embedding model for query encoding...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"sentence-transformers not available: {e}")
            cursor.close()
            return

        def embedding_to_oracle_str(embedding) -> str:
            """Convert numpy embedding to Oracle VECTOR string format."""
            return "[" + ",".join(str(float(x)) for x in embedding) + "]"

        def oracle_vector_search(query: str, top_k: int = 8, category_filter: str = None):
            """
            Perform semantic search using Oracle VECTOR_DISTANCE.
            Returns list of (distance, category, message_text) tuples.
            """
            # Generate query embedding
            query_emb = model.encode([query])[0]
            query_str = embedding_to_oracle_str(query_emb)

            # Build SQL with optional category filter
            filter_clause = ""
            params = [query_str, query_str, top_k]
            if category_filter:
                filter_clause = "AND m.category = :4"
                params.append(category_filter)

            sql = f"""
                SELECT
                    ROUND(VECTOR_DISTANCE(m.embedding, TO_VECTOR(:1, 384, FLOAT32), COSINE), 4) AS distance,
                    m.category,
                    m.message_text
                FROM f1_messages m
                WHERE m.embedding IS NOT NULL
                {filter_clause}
                ORDER BY VECTOR_DISTANCE(m.embedding, TO_VECTOR(:2, 384, FLOAT32), COSINE)
                FETCH FIRST :3 ROWS ONLY
            """
            cursor.execute(sql, params)
            return cursor.fetchall()

        # 6.2 Semantic Search: "dangerous driving" using Oracle VECTOR_DISTANCE
        print("=" * 80)
        print("  6.2 VECTOR: Semantic Search using Oracle VECTOR_DISTANCE")
        print("      Query: 'dangerous driving or collision between cars'")
        print("      (finds INCIDENT messages WITHOUT keyword matching)")
        print("=" * 80)

        query = "dangerous driving or collision between cars"
        results = oracle_vector_search(query, top_k=8)

        print(f"  Query: \"{query}\"")
        print()
        print(f"  {'DISTANCE':<10} | {'CATEGORY':<12} | {'MESSAGE':<50}")
        print("  " + "-" * 78)
        for distance, category, message in results:
            cat = category or 'Unknown'
            msg_short = message[:47] + "..." if len(message) > 47 else message
            print(f"  {distance:<10.4f} | {cat:<12} | {msg_short:<50}")
        print()

        # 6.3 Semantic Search: "race neutralization"
        print("=" * 80)
        print("  6.3 VECTOR: Semantic Search for 'race neutralization or caution'")
        print("      (should find Safety Car, VSC, Yellow Flag messages)")
        print("=" * 80)

        query = "race neutralization or caution period"
        results = oracle_vector_search(query, top_k=8)

        print(f"  Query: \"{query}\"")
        print()
        print(f"  {'DISTANCE':<10} | {'CATEGORY':<12} | {'MESSAGE':<50}")
        print("  " + "-" * 78)
        for distance, category, message in results:
            cat = category or 'Unknown'
            msg_short = message[:47] + "..." if len(message) > 47 else message
            print(f"  {distance:<10.4f} | {cat:<12} | {msg_short:<50}")
        print()

        # 6.4 Oracle Hybrid Query: Vector + SQL filters
        print("=" * 80)
        print("  6.4 VECTOR: Hybrid Query - Vector Search + SQL Filters")
        print("      Combines VECTOR_DISTANCE with relational filtering")
        print("=" * 80)

        query = "overtaking assistance system"
        query_emb = model.encode([query])[0]
        query_str = embedding_to_oracle_str(query_emb)

        # Hybrid query: vector similarity + time range + category filter
        cursor.execute("""
            SELECT
                ROUND(VECTOR_DISTANCE(m.embedding, TO_VECTOR(:1, 384, FLOAT32), COSINE), 4) AS distance,
                m.category,
                m.flag,
                ROUND(m.msg_time_sec / 60, 1) AS race_minute,
                m.message_text
            FROM f1_messages m
            WHERE m.embedding IS NOT NULL
              AND m.category = 'Drs'
            ORDER BY VECTOR_DISTANCE(m.embedding, TO_VECTOR(:2, 384, FLOAT32), COSINE)
            FETCH FIRST 8 ROWS ONLY
        """, [query_str, query_str])

        results = cursor.fetchall()
        print(f"  Query: \"{query}\" + Filter: category='Drs'")
        print()
        print(f"  {'DISTANCE':<10} | {'MINUTE':<8} | {'MESSAGE':<55}")
        print("  " + "-" * 78)
        for distance, category, flag, race_min, message in results:
            msg_short = message[:52] + "..." if len(message) > 52 else message
            print(f"  {distance:<10.4f} | {race_min or 0:<8.1f} | {msg_short:<55}")
        print()

        # 6.5 Find Similar Messages by Example (seed-based search)
        print("=" * 80)
        print("  6.5 VECTOR: Find Similar Messages by Example")
        print("      Uses existing message embedding as query vector")
        print("=" * 80)

        cursor.execute("""
            SELECT
                m2.msg_id,
                m2.category,
                ROUND(VECTOR_DISTANCE(m1.embedding, m2.embedding, COSINE), 4) AS distance,
                m2.message_text
            FROM f1_messages m1
            CROSS JOIN f1_messages m2
            WHERE m1.msg_id = (SELECT MIN(msg_id) FROM f1_messages WHERE category = 'SafetyCar')
              AND m2.msg_id != m1.msg_id
              AND m2.embedding IS NOT NULL
            ORDER BY VECTOR_DISTANCE(m1.embedding, m2.embedding, COSINE)
            FETCH FIRST 8 ROWS ONLY
        """)

        results = cursor.fetchall()
        print("  Seed message: First SafetyCar message")
        print()
        print(f"  {'DISTANCE':<10} | {'CATEGORY':<12} | {'MESSAGE':<50}")
        print("  " + "-" * 78)
        for msg_id, category, distance, message in results:
            cat = category or 'Unknown'
            msg_short = message[:47] + "..." if len(message) > 47 else message
            print(f"  {distance:<10.4f} | {cat:<12} | {msg_short:<50}")
        print()

        print("  Note: Oracle VECTOR_DISTANCE returns distance (lower = more similar)")
        print("        Cosine distance 0.0 = identical, 2.0 = opposite")
        print()

        cursor.close()

    def run_all(self, features: Optional[List[str]] = None):
        """Run all or selected use cases."""
        all_features = ["relational", "json", "timeseries", "spatial", "graph", "vector"]

        if features is None:
            features = all_features
        else:
            features = [f.lower() for f in features]
            invalid = set(features) - set(all_features)
            if invalid:
                logger.warning(f"Unknown features: {invalid}")

        print("\n" + "=" * 80)
        print("  Oracle Converged Database - F1 Demo Use Cases")
        print("=" * 80)

        if "relational" in features:
            print("\n>>> RELATIONAL ANALYTICS (Window Functions, Aggregations)")
            self.run_relational_analytics()

        if "json" in features:
            print("\n>>> JSON QUERIES (JSON_VALUE, Document Extraction)")
            self.run_json_queries()

        if "timeseries" in features:
            print("\n>>> TIME SERIES QUERIES (Telemetry Analysis)")
            self.run_time_series_queries()

        if "spatial" in features:
            print("\n>>> SPATIAL QUERIES (SDO_GEOMETRY, Proximity Search)")
            self.run_spatial_queries()

        if "graph" in features:
            print("\n>>> GRAPH QUERIES (SQL Property Graph)")
            self.run_graph_queries()

        if "vector" in features:
            print("\n>>> VECTOR QUERIES (Similarity Search)")
            self.run_vector_queries()

        print("\n" + "=" * 80)
        print("  Demo Complete!")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run F1 demonstration use cases against Oracle 23ai"
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
        help="Session type (default: R)"
    )
    parser.add_argument(
        "--features", type=str, nargs="+",
        choices=["relational", "json", "timeseries", "spatial", "graph", "vector"],
        help="Specific features to demonstrate (default: all)"
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

    runner = UseCaseRunner(user, password, dsn)

    try:
        runner.connect()
        runner.run_all(features=args.features)
    except Exception as e:
        logger.error(f"Use case execution failed: {e}")
        sys.exit(1)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
