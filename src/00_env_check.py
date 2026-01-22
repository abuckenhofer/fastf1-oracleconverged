"""
Oracle Environment Check Module

Verifies Oracle connectivity and prints database information.
Uses environment variables: ORA_USER, ORA_PASSWORD, ORA_DSN
Loads from .env file if present.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import oracledb

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def get_connection_params():
    """Get Oracle connection parameters from environment variables."""
    user = os.environ.get("ORA_USER")
    password = os.environ.get("ORA_PASSWORD")
    dsn = os.environ.get("ORA_DSN")

    missing = []
    if not user:
        missing.append("ORA_USER")
    if not password:
        missing.append("ORA_PASSWORD")
    if not dsn:
        missing.append("ORA_DSN")

    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("\nRequired environment variables:")
        print("  ORA_USER     - Oracle username")
        print("  ORA_PASSWORD - Oracle password")
        print("  ORA_DSN      - Oracle connection string (e.g., localhost:1521/XEPDB1)")
        return None

    return {"user": user, "password": password, "dsn": dsn}


def check_oracle_connection():
    """Connect to Oracle and print database information."""
    params = get_connection_params()
    if not params:
        return False

    print(f"Connecting to Oracle as {params['user']}@{params['dsn']}...")
    print("-" * 60)

    try:
        # Use thin mode (no Oracle Client required)
        connection = oracledb.connect(
            user=params["user"],
            password=params["password"],
            dsn=params["dsn"]
        )

        cursor = connection.cursor()

        # Get Oracle version
        cursor.execute("SELECT banner FROM v$version WHERE ROWNUM = 1")
        version_row = cursor.fetchone()
        version = version_row[0] if version_row else "Unknown"
        print(f"Oracle Version: {version}")

        # Get database name
        cursor.execute("SELECT name FROM v$database")
        db_row = cursor.fetchone()
        db_name = db_row[0] if db_row else "Unknown"
        print(f"Database Name:  {db_name}")

        # Check if CDB or PDB
        try:
            cursor.execute("SELECT CDB FROM v$database")
            cdb_row = cursor.fetchone()
            is_cdb = cdb_row[0] if cdb_row else "N"
            print(f"Is CDB:         {is_cdb}")

            if is_cdb == "YES":
                cursor.execute("SELECT name FROM v$pdbs WHERE con_id = SYS_CONTEXT('USERENV', 'CON_ID')")
                pdb_row = cursor.fetchone()
                pdb_name = pdb_row[0] if pdb_row else "N/A"
                print(f"Current PDB:    {pdb_name}")
        except oracledb.DatabaseError:
            print("CDB/PDB Info:   Not available (possibly non-CDB)")

        # Get current schema
        cursor.execute("SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM dual")
        schema_row = cursor.fetchone()
        schema = schema_row[0] if schema_row else "Unknown"
        print(f"Current Schema: {schema}")

        # Get session info
        cursor.execute("SELECT SYS_CONTEXT('USERENV', 'SESSION_USER') FROM dual")
        session_row = cursor.fetchone()
        session_user = session_row[0] if session_row else "Unknown"
        print(f"Session User:   {session_user}")

        # Check for key Oracle features
        print("\n" + "-" * 60)
        print("Checking Oracle Features:")

        # Check for JSON support
        try:
            cursor.execute("SELECT JSON_OBJECT('test' VALUE 'ok') FROM dual")
            print("  JSON Support:       Available")
        except oracledb.DatabaseError:
            print("  JSON Support:       Not available")

        # Check for VECTOR type (Oracle 23ai)
        try:
            cursor.execute("SELECT 1 FROM dual WHERE 1=0 AND VECTOR('[1,2,3]', 3, FLOAT32) IS NOT NULL")
            print("  VECTOR Type:        Available")
        except oracledb.DatabaseError:
            print("  VECTOR Type:        Not available (requires Oracle 23ai)")

        # Check for Property Graph (Oracle 23ai)
        try:
            cursor.execute("SELECT 1 FROM all_property_graphs WHERE ROWNUM = 1")
            print("  Property Graphs:    Available")
        except oracledb.DatabaseError:
            print("  Property Graphs:    Not available or no permissions")

        # Check for Spatial
        try:
            cursor.execute("SELECT SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(0, 0, NULL), NULL, NULL) FROM dual")
            print("  Spatial (SDO):      Available")
        except oracledb.DatabaseError:
            print("  Spatial (SDO):      Not available")

        # Check for In-Memory Column Store
        try:
            cursor.execute("SELECT value FROM v$parameter WHERE name = 'inmemory_size'")
            row = cursor.fetchone()
            inmemory_size = int(row[0]) if row else 0
            if inmemory_size > 0:
                size_gb = inmemory_size / (1024 * 1024 * 1024)
                print(f"  In-Memory Store:    Available ({size_gb:.1f} GB)")
            else:
                print("  In-Memory Store:    Not enabled (inmemory_size=0)")
        except oracledb.DatabaseError:
            print("  In-Memory Store:    Not available or no permissions")

        print("\n" + "-" * 60)
        print("Connection successful!")

        cursor.close()
        connection.close()
        return True

    except oracledb.DatabaseError as e:
        error, = e.args
        print(f"\nERROR: Failed to connect to Oracle")
        print(f"  Code:    {error.code}")
        print(f"  Message: {error.message}")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("Oracle Converged Database - Environment Check")
    print("=" * 60)
    print()

    success = check_oracle_connection()

    print()
    if success:
        print("Environment check PASSED")
        sys.exit(0)
    else:
        print("Environment check FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
