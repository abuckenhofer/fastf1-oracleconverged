"""
Oracle Data Loader Module

Loads F1 bronze data into Oracle 23ai, demonstrating converged database capabilities:
- Relational dimension/fact tables
- JSON document storage
- Time series data
- Spatial data (SDO_GEOMETRY)
- Graph vertex/edge tables

Loads connection settings from .env file if present.
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
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


class OracleF1Loader:
    """Loads F1 data from bronze layer into Oracle 23ai."""

    BATCH_SIZE = 1000

    def __init__(self, user: str, password: str, dsn: str):
        """Initialize loader with Oracle connection parameters."""
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

    def _read_json(self, filename: str) -> Optional[dict]:
        """Read a JSON file from bronze layer."""
        filepath = self._get_bronze_path() / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        logger.info(f"Reading {filename}...")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _check_oracle_version(self) -> tuple:
        """Check Oracle version and available features."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT version_full FROM v$instance")
        version_str = cursor.fetchone()[0]
        major_version = int(version_str.split('.')[0])

        # Check for VECTOR support (Oracle 23ai+)
        has_vector = False
        if major_version >= 23:
            try:
                cursor.execute("SELECT 1 FROM dual WHERE VECTOR('[1,2,3]', 3, FLOAT32) IS NOT NULL")
                has_vector = True
            except oracledb.DatabaseError:
                pass

        cursor.close()
        logger.info(f"Oracle version: {version_str} (major: {major_version})")
        logger.info(f"VECTOR support: {'Yes' if has_vector else 'No'}")
        return major_version, has_vector

    def _exec_ddl(self, cursor, sql: str, ignore_errors: list = None):
        """Execute DDL with optional error suppression."""
        ignore_errors = ignore_errors or []
        try:
            cursor.execute(sql)
        except oracledb.DatabaseError as e:
            error, = e.args
            if error.code not in ignore_errors:
                logger.warning(f"DDL Error ({error.code}): {error.message[:80]}")

    def run_schema_sql(self):
        """Create schema programmatically for Oracle version compatibility."""
        logger.info("Creating schema...")
        major_version, has_vector = self._check_oracle_version()
        cursor = self.connection.cursor()

        # Drop existing objects (ignore errors)
        drop_ignore = [942, 2289, 4043, 1418, 31626]  # table/sequence/index not exists, graph not exists
        for obj in ['f1_graph']:
            self._exec_ddl(cursor, f"DROP PROPERTY GRAPH {obj}", drop_ignore)
        for tbl in ['telemetry_spatial', 'f1_messages', 'f1_raw_documents',
                    'fact_session_status', 'fact_track_status', 'fact_weather',
                    'fact_telemetry', 'fact_lap', 'fact_result', 'bridge_driver_team',
                    'dim_team', 'dim_driver', 'dim_event']:
            self._exec_ddl(cursor, f"DROP TABLE {tbl} CASCADE CONSTRAINTS", drop_ignore)
        for seq in ['seq_event_id', 'seq_driver_id', 'seq_team_id', 'seq_doc_id', 'seq_msg_id']:
            self._exec_ddl(cursor, f"DROP SEQUENCE {seq}", drop_ignore)

        # Create sequences
        for seq in ['seq_event_id', 'seq_driver_id', 'seq_team_id', 'seq_doc_id', 'seq_msg_id']:
            self._exec_ddl(cursor, f"CREATE SEQUENCE {seq} START WITH 1 INCREMENT BY 1 NOCACHE", [955])

        # Create dimension tables
        self._exec_ddl(cursor, """
            CREATE TABLE dim_event (
                event_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                session_id VARCHAR2(100) NOT NULL UNIQUE,
                season NUMBER(4) NOT NULL,
                gp_name VARCHAR2(200) NOT NULL,
                gp_slug VARCHAR2(100) NOT NULL,
                session_code VARCHAR2(10) NOT NULL,
                session_name VARCHAR2(100),
                circuit_name VARCHAR2(200),
                country_code VARCHAR2(10),
                country_name VARCHAR2(100),
                session_start_ts TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        self._exec_ddl(cursor, "CREATE INDEX idx_dim_event_season ON dim_event(season)", [955])

        self._exec_ddl(cursor, """
            CREATE TABLE dim_driver (
                driver_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                driver_code VARCHAR2(10) NOT NULL UNIQUE,
                driver_number NUMBER(3),
                full_name VARCHAR2(200),
                first_name VARCHAR2(100),
                last_name VARCHAR2(100),
                country_code VARCHAR2(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

        self._exec_ddl(cursor, """
            CREATE TABLE dim_team (
                team_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                team_name VARCHAR2(200) NOT NULL,
                team_code VARCHAR2(50) UNIQUE,
                team_color VARCHAR2(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

        self._exec_ddl(cursor, """
            CREATE TABLE bridge_driver_team (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                driver_id NUMBER NOT NULL REFERENCES dim_driver(driver_id),
                team_id NUMBER NOT NULL REFERENCES dim_team(team_id),
                CONSTRAINT pk_bridge_driver_team PRIMARY KEY (event_id, driver_id)
            )""")

        # Tyre compound dimension
        self._exec_ddl(cursor, """
            CREATE TABLE dim_compound (
                compound_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                compound_name VARCHAR2(20) NOT NULL UNIQUE,
                compound_type VARCHAR2(20)
            )""")

        # Pre-populate known compounds
        for compound, ctype in [('SOFT', 'DRY'), ('MEDIUM', 'DRY'), ('HARD', 'DRY'),
                                ('INTERMEDIATE', 'WET'), ('WET', 'WET'),
                                ('HYPERSOFT', 'DRY'), ('ULTRASOFT', 'DRY'),
                                ('SUPERSOFT', 'DRY'), ('UNKNOWN', 'OTHER')]:
            self._exec_ddl(cursor, f"""
                INSERT INTO dim_compound (compound_name, compound_type)
                VALUES ('{compound}', '{ctype}')
            """, [1])  # Ignore unique constraint errors

        # Result status dimension
        self._exec_ddl(cursor, """
            CREATE TABLE dim_result_status (
                status_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                status_name VARCHAR2(100) NOT NULL UNIQUE,
                is_classified NUMBER(1) DEFAULT 1
            )""")

        # Pre-populate common statuses
        for status, classified in [('Finished', 1), ('+1 Lap', 1), ('+2 Laps', 1),
                                   ('Collision', 0), ('Accident', 0), ('Retired', 0),
                                   ('Engine', 0), ('Gearbox', 0), ('Hydraulics', 0),
                                   ('Brakes', 0), ('Suspension', 0), ('Electrical', 0),
                                   ('Spun off', 0), ('Collision damage', 0), ('Power Unit', 0),
                                   ('Disqualified', 0), ('Excluded', 0), ('Not classified', 0)]:
            self._exec_ddl(cursor, f"""
                INSERT INTO dim_result_status (status_name, is_classified)
                VALUES ('{status}', {classified})
            """, [1])

        # Create fact tables
        self._exec_ddl(cursor, """
            CREATE TABLE fact_result (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                driver_id NUMBER NOT NULL REFERENCES dim_driver(driver_id),
                grid_position NUMBER(3),
                final_position NUMBER(3),
                classified_position NUMBER(3),
                points NUMBER(5,2),
                status_id NUMBER REFERENCES dim_result_status(status_id),
                laps_completed NUMBER(4),
                total_time_sec NUMBER(12,3),
                q1_time_sec NUMBER(12,6),
                q2_time_sec NUMBER(12,6),
                q3_time_sec NUMBER(12,6),
                CONSTRAINT pk_fact_result PRIMARY KEY (event_id, driver_id)
            )""")

        self._exec_ddl(cursor, """
            CREATE TABLE fact_lap (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                driver_id NUMBER NOT NULL REFERENCES dim_driver(driver_id),
                lap_number NUMBER(4) NOT NULL,
                lap_time_sec NUMBER(12,6),
                sector1_sec NUMBER(12,6),
                sector2_sec NUMBER(12,6),
                sector3_sec NUMBER(12,6),
                stint_number NUMBER(3),
                compound_id NUMBER REFERENCES dim_compound(compound_id),
                tyre_life NUMBER(4),
                fresh_tyre NUMBER(1),
                is_personal_best NUMBER(1),
                is_accurate NUMBER(1),
                position NUMBER(3),
                pit_in_time_sec NUMBER(12,6),
                pit_out_time_sec NUMBER(12,6),
                track_status_code NUMBER(2),
                speed_i1 NUMBER(6,1),
                speed_i2 NUMBER(6,1),
                speed_fl NUMBER(6,1),
                speed_st NUMBER(6,1),
                lap_start_time_sec NUMBER(14,6),
                CONSTRAINT pk_fact_lap PRIMARY KEY (event_id, driver_id, lap_number)
            )""")
        self._exec_ddl(cursor, "CREATE INDEX idx_fact_lap_compound ON fact_lap(compound_id)", [955])

        # JSON landing table
        self._exec_ddl(cursor, """
            CREATE TABLE f1_raw_documents (
                doc_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                doc_type VARCHAR2(50) NOT NULL,
                event_id NUMBER REFERENCES dim_event(event_id),
                driver_id NUMBER REFERENCES dim_driver(driver_id),
                payload JSON CONSTRAINT chk_payload_json CHECK (payload IS JSON),
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

        # Time series tables
        self._exec_ddl(cursor, """
            CREATE TABLE fact_telemetry (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                driver_id NUMBER NOT NULL REFERENCES dim_driver(driver_id),
                sample_ts TIMESTAMP,
                session_time_sec NUMBER(14,6) NOT NULL,
                speed NUMBER(6,2),
                rpm NUMBER(8,2),
                n_gear NUMBER(2),
                throttle NUMBER(5,2),
                brake NUMBER(1),
                drs NUMBER(2),
                x NUMBER(12,4),
                y NUMBER(12,4),
                z NUMBER(12,4),
                distance NUMBER(14,6),
                CONSTRAINT pk_fact_telemetry PRIMARY KEY (event_id, driver_id, session_time_sec)
            )""")

        self._exec_ddl(cursor, """
            CREATE TABLE fact_weather (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                time_sec NUMBER(14,6) NOT NULL,
                air_temp NUMBER(5,2),
                track_temp NUMBER(5,2),
                humidity NUMBER(5,2),
                pressure NUMBER(8,2),
                wind_speed NUMBER(5,2),
                wind_direction NUMBER(5,1),
                rainfall NUMBER(1),
                CONSTRAINT pk_fact_weather PRIMARY KEY (event_id, time_sec)
            )""")

        self._exec_ddl(cursor, """
            CREATE TABLE fact_track_status (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                time_sec NUMBER(14,6) NOT NULL,
                status_code NUMBER(2),
                status_message VARCHAR2(100),
                CONSTRAINT pk_fact_track_status PRIMARY KEY (event_id, time_sec)
            )""")

        self._exec_ddl(cursor, """
            CREATE TABLE fact_session_status (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                time_sec NUMBER(14,6) NOT NULL,
                status VARCHAR2(50),
                CONSTRAINT pk_fact_session_status PRIMARY KEY (event_id, time_sec)
            )""")

        # Spatial table
        self._exec_ddl(cursor, """
            CREATE TABLE telemetry_spatial (
                event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                driver_id NUMBER NOT NULL REFERENCES dim_driver(driver_id),
                session_time_sec NUMBER(14,6) NOT NULL,
                geom MDSYS.SDO_GEOMETRY,
                speed NUMBER(6,2),
                throttle NUMBER(5,2),
                brake NUMBER(1),
                CONSTRAINT pk_telemetry_spatial PRIMARY KEY (event_id, driver_id, session_time_sec)
            )""")

        # Spatial metadata and index
        try:
            cursor.execute("DELETE FROM USER_SDO_GEOM_METADATA WHERE TABLE_NAME = 'TELEMETRY_SPATIAL'")
            cursor.execute("""
                INSERT INTO USER_SDO_GEOM_METADATA (TABLE_NAME, COLUMN_NAME, DIMINFO, SRID)
                VALUES ('TELEMETRY_SPATIAL', 'GEOM',
                    MDSYS.SDO_DIM_ARRAY(
                        MDSYS.SDO_DIM_ELEMENT('X', -10000, 10000, 0.5),
                        MDSYS.SDO_DIM_ELEMENT('Y', -10000, 10000, 0.5)
                    ), NULL)
            """)
        except oracledb.DatabaseError as e:
            logger.warning(f"Spatial metadata setup: {e}")

        # Messages table (without VECTOR for Oracle < 23)
        if has_vector:
            self._exec_ddl(cursor, """
                CREATE TABLE f1_messages (
                    event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                    msg_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    msg_ts TIMESTAMP,
                    msg_time_sec NUMBER(14,6),
                    category VARCHAR2(50),
                    flag VARCHAR2(20),
                    scope VARCHAR2(50),
                    message_text VARCHAR2(4000),
                    embedding VECTOR(384, FLOAT32),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")
        else:
            self._exec_ddl(cursor, """
                CREATE TABLE f1_messages (
                    event_id NUMBER NOT NULL REFERENCES dim_event(event_id),
                    msg_id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    msg_ts TIMESTAMP,
                    msg_time_sec NUMBER(14,6),
                    category VARCHAR2(50),
                    flag VARCHAR2(20),
                    scope VARCHAR2(50),
                    message_text VARCHAR2(4000),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")
            logger.info("Created f1_messages without VECTOR column (Oracle < 23ai)")

        self._exec_ddl(cursor, "CREATE INDEX idx_messages_event ON f1_messages(event_id)", [955])
        self._exec_ddl(cursor, "CREATE INDEX idx_messages_category ON f1_messages(category)", [955])

        # Property Graph (Oracle 23ai only)
        if major_version >= 23:
            try:
                cursor.execute("""
                    CREATE PROPERTY GRAPH f1_graph
                    VERTEX TABLES (
                        dim_driver KEY (driver_id) LABEL driver
                            PROPERTIES (driver_id, driver_code, full_name, driver_number),
                        dim_team KEY (team_id) LABEL team
                            PROPERTIES (team_id, team_name, team_code),
                        dim_event KEY (event_id) LABEL event
                            PROPERTIES (event_id, session_id, season, gp_name, session_name)
                    )
                    EDGE TABLES (
                        bridge_driver_team KEY (event_id, driver_id)
                            SOURCE KEY (driver_id) REFERENCES dim_driver (driver_id)
                            DESTINATION KEY (team_id) REFERENCES dim_team (team_id)
                            LABEL drives_for PROPERTIES (event_id),
                        fact_result AS driver_participates KEY (event_id, driver_id)
                            SOURCE KEY (driver_id) REFERENCES dim_driver (driver_id)
                            DESTINATION KEY (event_id) REFERENCES dim_event (event_id)
                            LABEL participates_in PROPERTIES (final_position, points, status_id)
                    )
                """)
                logger.info("Created f1_graph property graph")
            except oracledb.DatabaseError as e:
                logger.warning(f"Property graph creation skipped: {e}")
        else:
            logger.info("Skipping property graph (requires Oracle 23ai)")

        self.connection.commit()
        cursor.close()
        logger.info("Schema creation completed")

    def load_event(self, session_id: str, year: int, gp: str, session: str) -> int:
        """Load or get the event dimension record."""
        cursor = self.connection.cursor()

        # Check if event exists
        cursor.execute(
            "SELECT event_id FROM dim_event WHERE session_id = :1",
            [session_id]
        )
        row = cursor.fetchone()
        if row:
            logger.info(f"Event already exists: {session_id} (event_id={row[0]})")
            cursor.close()
            return row[0]

        # Read session info JSON for additional metadata
        json_file = f"session_info_{session_id}.json"
        session_info = self._read_json(json_file)

        gp_slug = re.sub(r"\s+", "_", gp.strip())
        gp_slug = re.sub(r"[^A-Za-z0-9_\-]", "", gp_slug)

        # Extract metadata from JSON
        session_name = None
        circuit_name = None
        country_code = None
        country_name = None
        session_start = None

        if session_info:
            data = session_info.get("data", {})
            session_name = data.get("session_name")
            nested = data.get("data", {})
            meeting = nested.get("Meeting", {})
            circuit_name = meeting.get("Location")
            country = meeting.get("Country", {})
            country_code = country.get("Code")
            country_name = country.get("Name")
            start_date_str = nested.get("StartDate")
            if start_date_str:
                try:
                    session_start = datetime.fromisoformat(start_date_str)
                except ValueError:
                    pass

        # Insert event
        event_id_var = cursor.var(oracledb.NUMBER)
        cursor.execute("""
            INSERT INTO dim_event (
                session_id, season, gp_name, gp_slug, session_code,
                session_name, circuit_name, country_code, country_name, session_start_ts
            ) VALUES (
                :1, :2, :3, :4, :5, :6, :7, :8, :9, :10
            ) RETURNING event_id INTO :11
        """, [
            session_id, year, gp, gp_slug, session,
            session_name, circuit_name, country_code, country_name, session_start,
            event_id_var
        ])

        event_id = event_id_var.getvalue()
        if event_id:
            event_id = int(event_id[0])
        else:
            # Fallback: query the inserted ID
            cursor.execute("SELECT event_id FROM dim_event WHERE session_id = :1", [session_id])
            event_id = cursor.fetchone()[0]

        self.connection.commit()
        cursor.close()
        logger.info(f"Created event: {session_id} (event_id={event_id})")
        return event_id

    def load_drivers_and_teams(self, session_id: str, event_id: int) -> tuple:
        """Load drivers, teams, and their relationships. Returns (driver_code_map, driver_number_map)."""
        results_file = f"results_{session_id}.csv"
        df = self._read_csv(results_file)
        if df is None:
            return {}, {}

        cursor = self.connection.cursor()
        driver_map = {}       # driver_code -> driver_id
        driver_num_map = {}   # driver_number (str) -> driver_id
        team_map = {}         # team_name -> team_id

        for _, row in df.iterrows():
            driver_code = str(row.get("Abbreviation", "")).strip()
            if not driver_code:
                continue

            driver_number = row.get("DriverNumber")

            # Get or create driver
            cursor.execute(
                "SELECT driver_id FROM dim_driver WHERE driver_code = :1",
                [driver_code]
            )
            driver_row = cursor.fetchone()

            if driver_row:
                driver_id = driver_row[0]
            else:
                cursor.execute("""
                    INSERT INTO dim_driver (
                        driver_code, driver_number, full_name, first_name, last_name, country_code
                    ) VALUES (:1, :2, :3, :4, :5, :6)
                """, [
                    driver_code,
                    int(driver_number) if pd.notna(driver_number) else None,
                    row.get("FullName"),
                    row.get("FirstName"),
                    row.get("LastName"),
                    row.get("CountryCode")
                ])
                cursor.execute(
                    "SELECT driver_id FROM dim_driver WHERE driver_code = :1",
                    [driver_code]
                )
                driver_id = cursor.fetchone()[0]

            driver_map[driver_code] = driver_id
            # Also map by driver number (as string, since telemetry uses it)
            if pd.notna(driver_number):
                driver_num_map[str(int(driver_number))] = driver_id

            # Get or create team
            team_name = str(row.get("TeamName", "")).strip()
            if team_name and team_name not in team_map:
                cursor.execute(
                    "SELECT team_id FROM dim_team WHERE team_name = :1",
                    [team_name]
                )
                team_row = cursor.fetchone()

                if team_row:
                    team_id = team_row[0]
                else:
                    cursor.execute("""
                        INSERT INTO dim_team (team_name, team_code, team_color)
                        VALUES (:1, :2, :3)
                    """, [
                        team_name,
                        row.get("TeamId"),
                        row.get("TeamColor")
                    ])
                    cursor.execute(
                        "SELECT team_id FROM dim_team WHERE team_name = :1",
                        [team_name]
                    )
                    team_id = cursor.fetchone()[0]

                team_map[team_name] = team_id

            # Create bridge record (driver-team for this event)
            if team_name in team_map:
                try:
                    cursor.execute("""
                        INSERT INTO bridge_driver_team (event_id, driver_id, team_id)
                        VALUES (:1, :2, :3)
                    """, [event_id, driver_id, team_map[team_name]])
                except oracledb.IntegrityError:
                    pass  # Already exists

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {len(driver_map)} drivers, {len(team_map)} teams")
        return driver_map, driver_num_map

    def _get_or_create_status(self, cursor, status: str) -> Optional[int]:
        """Get or create a result status, returning status_id."""
        if not status or pd.isna(status):
            return None
        status = str(status).strip()
        if not status:
            return None

        cursor.execute("SELECT status_id FROM dim_result_status WHERE status_name = :1", [status])
        row = cursor.fetchone()
        if row:
            return row[0]

        # Insert new status
        try:
            cursor.execute("""
                INSERT INTO dim_result_status (status_name, is_classified)
                VALUES (:1, CASE WHEN :2 LIKE '%Lap%' OR :3 = 'Finished' THEN 1 ELSE 0 END)
            """, [status, status, status])
            cursor.execute("SELECT status_id FROM dim_result_status WHERE status_name = :1", [status])
            return cursor.fetchone()[0]
        except oracledb.IntegrityError:
            cursor.execute("SELECT status_id FROM dim_result_status WHERE status_name = :1", [status])
            return cursor.fetchone()[0]

    def load_results(self, session_id: str, event_id: int, driver_map: dict):
        """Load race results."""
        results_file = f"results_{session_id}.csv"
        df = self._read_csv(results_file)
        if df is None:
            return

        cursor = self.connection.cursor()
        loaded = 0

        for _, row in df.iterrows():
            driver_code = str(row.get("Abbreviation", "")).strip()
            driver_id = driver_map.get(driver_code)
            if not driver_id:
                continue

            # Lookup status_id from dimension
            status_id = self._get_or_create_status(cursor, row.get("Status"))

            # Parse classified position as numeric
            classified_pos = None
            if pd.notna(row.get("ClassifiedPosition")):
                try:
                    classified_pos = int(float(row["ClassifiedPosition"]))
                except (ValueError, TypeError):
                    pass

            try:
                cursor.execute("""
                    INSERT INTO fact_result (
                        event_id, driver_id, grid_position, final_position,
                        classified_position, points, status_id, laps_completed,
                        total_time_sec, q1_time_sec, q2_time_sec, q3_time_sec
                    ) VALUES (
                        :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12
                    )
                """, [
                    event_id, driver_id,
                    int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else None,
                    int(row["Position"]) if pd.notna(row.get("Position")) else None,
                    classified_pos,
                    float(row["Points"]) if pd.notna(row.get("Points")) else None,
                    status_id,
                    int(row["Laps"]) if pd.notna(row.get("Laps")) else None,
                    float(row["Time"]) if pd.notna(row.get("Time")) else None,
                    float(row["Q1"]) if pd.notna(row.get("Q1")) else None,
                    float(row["Q2"]) if pd.notna(row.get("Q2")) else None,
                    float(row["Q3"]) if pd.notna(row.get("Q3")) else None,
                ])
                loaded += 1
            except oracledb.IntegrityError:
                pass  # Already exists

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} result records")

    def _get_compound_id(self, cursor, compound: str) -> Optional[int]:
        """Get compound_id from dim_compound, creating if necessary."""
        if not compound or pd.isna(compound):
            return None
        compound = str(compound).strip().upper()
        if not compound:
            return None

        cursor.execute("SELECT compound_id FROM dim_compound WHERE compound_name = :1", [compound])
        row = cursor.fetchone()
        if row:
            return row[0]

        # Insert new compound
        try:
            ctype = 'DRY' if compound in ('SOFT', 'MEDIUM', 'HARD') else 'OTHER'
            cursor.execute("""
                INSERT INTO dim_compound (compound_name, compound_type)
                VALUES (:1, :2)
            """, [compound, ctype])
            cursor.execute("SELECT compound_id FROM dim_compound WHERE compound_name = :1", [compound])
            return cursor.fetchone()[0]
        except oracledb.IntegrityError:
            cursor.execute("SELECT compound_id FROM dim_compound WHERE compound_name = :1", [compound])
            return cursor.fetchone()[0]

    def load_laps(self, session_id: str, event_id: int, driver_map: dict):
        """Load lap data."""
        laps_file = f"laps_{session_id}.csv"
        df = self._read_csv(laps_file)
        if df is None:
            return

        cursor = self.connection.cursor()

        # Build compound lookup cache
        compound_cache = {}
        for _, row in df.iterrows():
            compound = row.get("Compound")
            if compound and pd.notna(compound):
                compound = str(compound).strip().upper()
                if compound and compound not in compound_cache:
                    compound_cache[compound] = self._get_compound_id(cursor, compound)

        batch = []
        loaded = 0

        for _, row in df.iterrows():
            driver_code = str(row.get("Driver", "")).strip()
            driver_id = driver_map.get(driver_code)
            if not driver_id:
                continue

            lap_number = row.get("LapNumber")
            if pd.isna(lap_number):
                continue

            # Get compound_id from cache
            compound = row.get("Compound")
            compound_id = None
            if compound and pd.notna(compound):
                compound_id = compound_cache.get(str(compound).strip().upper())

            # Parse track status as numeric code
            track_status_code = None
            ts = row.get("TrackStatus")
            if pd.notna(ts):
                try:
                    track_status_code = int(float(ts))
                except (ValueError, TypeError):
                    pass

            batch.append([
                event_id, driver_id, int(lap_number),
                float(row["LapTime"]) if pd.notna(row.get("LapTime")) else None,
                float(row["Sector1Time"]) if pd.notna(row.get("Sector1Time")) else None,
                float(row["Sector2Time"]) if pd.notna(row.get("Sector2Time")) else None,
                float(row["Sector3Time"]) if pd.notna(row.get("Sector3Time")) else None,
                int(row["Stint"]) if pd.notna(row.get("Stint")) else None,
                compound_id,
                int(row["TyreLife"]) if pd.notna(row.get("TyreLife")) else None,
                1 if row.get("FreshTyre") == True else 0,
                1 if row.get("IsPersonalBest") == True else 0,
                1 if row.get("IsAccurate") == True else 0,
                int(row["Position"]) if pd.notna(row.get("Position")) else None,
                float(row["PitInTime"]) if pd.notna(row.get("PitInTime")) else None,
                float(row["PitOutTime"]) if pd.notna(row.get("PitOutTime")) else None,
                track_status_code,
                float(row["SpeedI1"]) if pd.notna(row.get("SpeedI1")) else None,
                float(row["SpeedI2"]) if pd.notna(row.get("SpeedI2")) else None,
                float(row["SpeedFL"]) if pd.notna(row.get("SpeedFL")) else None,
                float(row["SpeedST"]) if pd.notna(row.get("SpeedST")) else None,
                float(row["Time"]) if pd.notna(row.get("Time")) else None,
            ])

            if len(batch) >= self.BATCH_SIZE:
                cursor.executemany("""
                    INSERT INTO fact_lap (
                        event_id, driver_id, lap_number,
                        lap_time_sec, sector1_sec, sector2_sec, sector3_sec,
                        stint_number, compound_id, tyre_life, fresh_tyre,
                        is_personal_best, is_accurate, position,
                        pit_in_time_sec, pit_out_time_sec, track_status_code,
                        speed_i1, speed_i2, speed_fl, speed_st, lap_start_time_sec
                    ) VALUES (
                        :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20, :21, :22
                    )
                """, batch)
                loaded += len(batch)
                batch = []

        if batch:
            cursor.executemany("""
                INSERT INTO fact_lap (
                    event_id, driver_id, lap_number,
                    lap_time_sec, sector1_sec, sector2_sec, sector3_sec,
                    stint_number, compound_id, tyre_life, fresh_tyre,
                    is_personal_best, is_accurate, position,
                    pit_in_time_sec, pit_out_time_sec, track_status_code,
                    speed_i1, speed_i2, speed_fl, speed_st, lap_start_time_sec
                ) VALUES (
                    :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20, :21, :22
                )
            """, batch)
            loaded += len(batch)

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} lap records")

    def load_telemetry(self, session_id: str, event_id: int, driver_num_map: dict, sample_rate: int = 10):
        """Load telemetry data (sampled for performance)."""
        telemetry_file = f"telemetry_{session_id}.csv"
        df = self._read_csv(telemetry_file)
        if df is None:
            return

        # Sample data to reduce volume
        df = df.iloc[::sample_rate].copy()
        logger.info(f"Sampling telemetry at 1:{sample_rate} ratio ({len(df)} rows)")

        cursor = self.connection.cursor()
        batch = []
        loaded = 0

        for _, row in df.iterrows():
            # Driver column contains driver number (e.g., "4"), not code (e.g., "NOR")
            driver_key = str(row.get("Driver", "")).strip()
            driver_id = driver_num_map.get(driver_key)
            if not driver_id:
                continue

            session_time = row.get("SessionTime")
            if pd.isna(session_time):
                continue

            # Parse timestamp if available
            sample_ts = None
            date_str = row.get("Date")
            if pd.notna(date_str):
                try:
                    sample_ts = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
                except ValueError:
                    pass

            batch.append([
                event_id, driver_id, sample_ts, float(session_time),
                float(row["Speed"]) if pd.notna(row.get("Speed")) else None,
                float(row["RPM"]) if pd.notna(row.get("RPM")) else None,
                int(row["nGear"]) if pd.notna(row.get("nGear")) else None,
                float(row["Throttle"]) if pd.notna(row.get("Throttle")) else None,
                1 if row.get("Brake") == True else 0,
                int(row["DRS"]) if pd.notna(row.get("DRS")) else None,
                float(row["X"]) if pd.notna(row.get("X")) else None,
                float(row["Y"]) if pd.notna(row.get("Y")) else None,
                float(row["Z"]) if pd.notna(row.get("Z")) else None,
                float(row["Distance"]) if pd.notna(row.get("Distance")) else None,
            ])

            if len(batch) >= self.BATCH_SIZE:
                try:
                    cursor.executemany("""
                        INSERT INTO fact_telemetry (
                            event_id, driver_id, sample_ts, session_time_sec,
                            speed, rpm, n_gear, throttle, brake, drs, x, y, z, distance
                        ) VALUES (
                            :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14
                        )
                    """, batch)
                    loaded += len(batch)
                except oracledb.DatabaseError as e:
                    logger.warning(f"Telemetry batch error: {e}")
                batch = []

        if batch:
            try:
                cursor.executemany("""
                    INSERT INTO fact_telemetry (
                        event_id, driver_id, sample_ts, session_time_sec,
                        speed, rpm, n_gear, throttle, brake, drs, x, y, z, distance
                    ) VALUES (
                        :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14
                    )
                """, batch)
                loaded += len(batch)
            except oracledb.DatabaseError as e:
                logger.warning(f"Telemetry batch error: {e}")

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} telemetry records")

    def load_spatial_data(self, session_id: str, event_id: int, driver_num_map: dict, sample_rate: int = 50):
        """Load spatial telemetry data with SDO_GEOMETRY points."""
        telemetry_file = f"telemetry_{session_id}.csv"
        df = self._read_csv(telemetry_file)
        if df is None:
            return

        # Sample more aggressively for spatial data
        df = df.iloc[::sample_rate].copy()
        logger.info(f"Sampling spatial data at 1:{sample_rate} ratio ({len(df)} rows)")

        cursor = self.connection.cursor()
        loaded = 0

        for _, row in df.iterrows():
            # Driver column contains driver number (e.g., "4"), not code (e.g., "NOR")
            driver_key = str(row.get("Driver", "")).strip()
            driver_id = driver_num_map.get(driver_key)
            if not driver_id:
                continue

            session_time = row.get("SessionTime")
            x = row.get("X")
            y = row.get("Y")

            if pd.isna(session_time) or pd.isna(x) or pd.isna(y):
                continue

            try:
                cursor.execute("""
                    INSERT INTO telemetry_spatial (
                        event_id, driver_id, session_time_sec, geom, speed, throttle, brake
                    ) VALUES (
                        :1, :2, :3,
                        SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(:4, :5, NULL), NULL, NULL),
                        :6, :7, :8
                    )
                """, [
                    event_id, driver_id, float(session_time),
                    float(x), float(y),
                    float(row["Speed"]) if pd.notna(row.get("Speed")) else None,
                    float(row["Throttle"]) if pd.notna(row.get("Throttle")) else None,
                    1 if row.get("Brake") == True else 0,
                ])
                loaded += 1
            except oracledb.DatabaseError:
                pass  # Skip duplicates or errors

            if loaded % 1000 == 0:
                self.connection.commit()

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} spatial telemetry records")

    def load_weather(self, session_id: str, event_id: int):
        """Load weather data."""
        weather_file = f"weather_{session_id}.csv"
        df = self._read_csv(weather_file)
        if df is None:
            return

        cursor = self.connection.cursor()
        batch = []
        loaded = 0

        for _, row in df.iterrows():
            time_val = row.get("Time")
            if pd.isna(time_val):
                continue

            batch.append([
                event_id, float(time_val),
                float(row["AirTemp"]) if pd.notna(row.get("AirTemp")) else None,
                float(row["TrackTemp"]) if pd.notna(row.get("TrackTemp")) else None,
                float(row["Humidity"]) if pd.notna(row.get("Humidity")) else None,
                float(row["Pressure"]) if pd.notna(row.get("Pressure")) else None,
                float(row["WindSpeed"]) if pd.notna(row.get("WindSpeed")) else None,
                float(row["WindDirection"]) if pd.notna(row.get("WindDirection")) else None,
                1 if row.get("Rainfall") == True else 0,
            ])

            if len(batch) >= self.BATCH_SIZE:
                cursor.executemany("""
                    INSERT INTO fact_weather (
                        event_id, time_sec, air_temp, track_temp, humidity,
                        pressure, wind_speed, wind_direction, rainfall
                    ) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9)
                """, batch)
                loaded += len(batch)
                batch = []

        if batch:
            cursor.executemany("""
                INSERT INTO fact_weather (
                    event_id, time_sec, air_temp, track_temp, humidity,
                    pressure, wind_speed, wind_direction, rainfall
                ) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9)
            """, batch)
            loaded += len(batch)

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} weather records")

    def load_track_status(self, session_id: str, event_id: int):
        """Load track status timeline."""
        status_file = f"track_status_{session_id}.csv"
        df = self._read_csv(status_file)
        if df is None:
            return

        cursor = self.connection.cursor()
        loaded = 0

        for _, row in df.iterrows():
            time_val = row.get("Time")
            if pd.isna(time_val):
                continue

            try:
                cursor.execute("""
                    INSERT INTO fact_track_status (event_id, time_sec, status_code, status_message)
                    VALUES (:1, :2, :3, :4)
                """, [
                    event_id, float(time_val),
                    int(row["Status"]) if pd.notna(row.get("Status")) else None,
                    row.get("Message"),
                ])
                loaded += 1
            except oracledb.IntegrityError:
                pass

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} track status records")

    def load_session_status(self, session_id: str, event_id: int):
        """Load session status timeline."""
        status_file = f"session_status_{session_id}.csv"
        df = self._read_csv(status_file)
        if df is None:
            return

        cursor = self.connection.cursor()
        loaded = 0

        for _, row in df.iterrows():
            time_val = row.get("Time")
            if pd.isna(time_val):
                continue

            try:
                cursor.execute("""
                    INSERT INTO fact_session_status (event_id, time_sec, status)
                    VALUES (:1, :2, :3)
                """, [event_id, float(time_val), row.get("Status")])
                loaded += 1
            except oracledb.IntegrityError:
                pass

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} session status records")

    def load_json_documents(self, session_id: str, event_id: int):
        """Load raw JSON documents into f1_raw_documents."""
        cursor = self.connection.cursor()
        loaded = 0

        # Load session_info JSON
        json_file = f"session_info_{session_id}.json"
        session_info = self._read_json(json_file)
        if session_info:
            try:
                cursor.execute("""
                    INSERT INTO f1_raw_documents (doc_type, event_id, payload)
                    VALUES ('session_info', :1, :2)
                """, [event_id, json.dumps(session_info.get("data", session_info))])
                loaded += 1
            except oracledb.DatabaseError as e:
                logger.warning(f"Failed to insert session_info JSON: {e}")

        # Load race control messages as JSON documents
        messages_file = f"race_control_messages_{session_id}.csv"
        df = self._read_csv(messages_file)
        if df is not None:
            batch = []
            for _, row in df.iterrows():
                msg_dict = row.to_dict()
                # Convert NaN to None for JSON serialization
                msg_dict = {k: (None if pd.isna(v) else v) for k, v in msg_dict.items()}
                batch.append([
                    "race_control_message",
                    event_id,
                    None,  # driver_id
                    json.dumps(msg_dict)
                ])

                if len(batch) >= self.BATCH_SIZE:
                    cursor.executemany("""
                        INSERT INTO f1_raw_documents (doc_type, event_id, driver_id, payload)
                        VALUES (:1, :2, :3, :4)
                    """, batch)
                    loaded += len(batch)
                    batch = []

            if batch:
                cursor.executemany("""
                    INSERT INTO f1_raw_documents (doc_type, event_id, driver_id, payload)
                    VALUES (:1, :2, :3, :4)
                """, batch)
                loaded += len(batch)

        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded {loaded} JSON documents")

    def run_views_sql(self):
        """Execute the views SQL script."""
        views_path = self.project_root / "sql" / "10_views.sql"
        if not views_path.exists():
            logger.warning(f"Views file not found: {views_path}")
            return

        logger.info("Creating views...")
        with open(views_path, "r", encoding="utf-8") as f:
            sql_content = f.read()

        cursor = self.connection.cursor()

        # Split by semicolon and execute each statement
        statements = sql_content.split(";")
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue
            try:
                cursor.execute(stmt)
            except oracledb.DatabaseError as e:
                error, = e.args
                if error.code != 942:  # Ignore table not found
                    logger.warning(f"View error: {error.message[:80]}")

        self.connection.commit()
        cursor.close()
        logger.info("Views created")

    def load_all(self, year: int, gp: str, session: str,
                 telemetry_sample_rate: int = 10,
                 spatial_sample_rate: int = 50):
        """Load all data for a session."""
        session_id = self._get_session_id(year, gp, session)

        logger.info("=" * 60)
        logger.info(f"Loading data for session: {session_id}")
        logger.info("=" * 60)

        # 1. Create schema
        self.run_schema_sql()

        # 2. Load dimension tables
        event_id = self.load_event(session_id, year, gp, session)
        driver_map, driver_num_map = self.load_drivers_and_teams(session_id, event_id)

        # 3. Load fact tables (use driver_code map)
        self.load_results(session_id, event_id, driver_map)
        self.load_laps(session_id, event_id, driver_map)

        # 4. Load time series (telemetry uses driver_number map)
        self.load_weather(session_id, event_id)
        self.load_track_status(session_id, event_id)
        self.load_session_status(session_id, event_id)
        self.load_telemetry(session_id, event_id, driver_num_map, sample_rate=telemetry_sample_rate)

        # 5. Load spatial data (uses driver_number map)
        self.load_spatial_data(session_id, event_id, driver_num_map, sample_rate=spatial_sample_rate)

        # 6. Load JSON documents
        self.load_json_documents(session_id, event_id)

        # 7. Create views
        self.run_views_sql()

        logger.info("=" * 60)
        logger.info("Data loading complete!")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load F1 bronze data into Oracle 23ai"
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
        "--telemetry-sample-rate", type=int, default=10,
        help="Sample rate for telemetry (1:N, default: 10)"
    )
    parser.add_argument(
        "--spatial-sample-rate", type=int, default=50,
        help="Sample rate for spatial data (1:N, default: 50)"
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

    loader = OracleF1Loader(user, password, dsn)

    try:
        loader.connect()
        loader.load_all(
            year=args.year,
            gp=args.gp,
            session=args.session,
            telemetry_sample_rate=args.telemetry_sample_rate,
            spatial_sample_rate=args.spatial_sample_rate
        )
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        sys.exit(1)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
