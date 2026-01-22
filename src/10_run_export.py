"""
F1 Data Export Wrapper Module

Calls the existing 01_export_f1_data.py exporter to produce bronze files.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_export(year: int, gp: str, session: str, limit_drivers: int = -1):
    """
    Run the F1 data exporter.

    Args:
        year: Season year (e.g., 2024)
        gp: Grand Prix name (e.g., "Singapore")
        session: Session type (e.g., "R" for Race, "Q" for Qualifying)
        limit_drivers: Limit telemetry to N drivers (-1 for all)
    """
    # Find the exporter script
    project_root = Path(__file__).parent.parent
    exporter_script = project_root / "01_export_f1_data.py"

    if not exporter_script.exists():
        print(f"ERROR: Exporter script not found: {exporter_script}")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(exporter_script),
        "--year", str(year),
        "--gp", gp,
        "--session", session,
        "--limit-drivers", str(limit_drivers)
    ]

    print("=" * 60)
    print("F1 Data Export")
    print("=" * 60)
    print(f"Year:     {year}")
    print(f"GP:       {gp}")
    print(f"Session:  {session}")
    print(f"Drivers:  {'All' if limit_drivers < 0 else limit_drivers}")
    print("-" * 60)
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print("Export completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Export failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Python interpreter not found: {sys.executable}")
        return False


def check_bronze_files(year: int, gp: str, session: str):
    """Check if bronze files already exist for this session."""
    project_root = Path(__file__).parent.parent
    bronze_path = project_root / "lakehouse" / "01_bronze"

    # Build session_id (matching the exporter's slug logic)
    import re
    gp_slug = re.sub(r"\s+", "_", gp.strip())
    gp_slug = re.sub(r"[^A-Za-z0-9_\-]", "", gp_slug)
    session_slug = re.sub(r"\s+", "_", session.strip())
    session_slug = re.sub(r"[^A-Za-z0-9_\-]", "", session_slug)
    session_id = f"{year}_{gp_slug}_{session_slug}"

    expected_files = [
        f"laps_{session_id}.csv",
        f"results_{session_id}.csv",
        f"weather_{session_id}.csv",
        f"telemetry_{session_id}.csv",
        f"race_control_messages_{session_id}.csv",
        f"track_status_{session_id}.csv",
        f"session_status_{session_id}.csv",
        f"session_info_{session_id}.json",
    ]

    existing = []
    missing = []

    for filename in expected_files:
        filepath = bronze_path / filename
        if filepath.exists():
            existing.append(filename)
        else:
            missing.append(filename)

    return session_id, existing, missing


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run F1 data export to bronze layer"
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
        "--limit-drivers", type=int, default=-1,
        help="Limit telemetry to N drivers (-1 for all, default: -1)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-export even if files exist"
    )

    args = parser.parse_args()

    # Check existing files
    session_id, existing, missing = check_bronze_files(args.year, args.gp, args.session)

    print("=" * 60)
    print(f"Session ID: {session_id}")
    print("=" * 60)

    if existing and not args.force:
        print(f"\nExisting bronze files found: {len(existing)}/{len(existing) + len(missing)}")
        for f in existing:
            print(f"  [OK] {f}")
        for f in missing:
            print(f"  [MISSING] {f}")

        if not missing:
            print("\nAll bronze files already exist. Use --force to re-export.")
            return
        else:
            print("\nSome files are missing. Running export...")

    success = run_export(
        year=args.year,
        gp=args.gp,
        session=args.session,
        limit_drivers=args.limit_drivers
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
