"""
F1 Data Export to Bronze Layer

Exports telemetry, laps, weather, results from FastF1 to CSV/JSON files.
Output is written to lakehouse/01_bronze/.
"""

import fastf1
import pandas as pd
from pathlib import Path
import logging
import json
import argparse
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class F1DataExporter:
    """Exports F1 session data to bronze layer CSV/JSON files."""

    def __init__(self, bronze_path: str = None, cache_dir: str = None):
        """
        Initialize exporter.

        Args:
            bronze_path: Output directory for bronze files (default: lakehouse/01_bronze)
            cache_dir: FastF1 cache directory (default: cache)
        """
        project_root = Path(__file__).parent.parent

        if bronze_path is None:
            bronze_path = project_root / "lakehouse" / "01_bronze"
        else:
            bronze_path = Path(bronze_path)

        if cache_dir is None:
            cache_dir = project_root / "cache"
        else:
            cache_dir = Path(cache_dir)

        self.bronze_path = bronze_path
        self.bronze_path.mkdir(parents=True, exist_ok=True)

        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))

    def _slug(self, text: str) -> str:
        """Make a filesystem-safe slug."""
        text = str(text).strip()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9_\-]", "", text)
        return text

    def _save_json(self, obj, filename: str) -> None:
        """Save an object as JSON."""
        try:
            payload = {
                'exported_at': datetime.utcnow().isoformat() + 'Z',
                'data': obj
            }
            out = self.bronze_path / filename
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            logger.info(f"Saved JSON to {filename}")
        except Exception as e:
            logger.error(f"Failed to save JSON {filename}: {e}")

    def _save_df(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV, converting timedeltas to seconds."""
        if df is None or df.empty:
            logger.warning(f"Skipping {filename}: No data available.")
            return

        try:
            export_df = df.copy()

            # Convert timedelta columns to seconds
            time_cols = export_df.select_dtypes(include=['timedelta64']).columns
            for col in time_cols:
                export_df[col] = export_df[col].dt.total_seconds()

            output_path = self.bronze_path / filename
            export_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(export_df)} rows to {filename}")

        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")

    def export_session(self, year: int, gp: str, session_type: str = 'R', limit_drivers: int = None):
        """
        Export all data for a session.

        Args:
            year: Season year (e.g., 2024)
            gp: Grand Prix name (e.g., "Singapore")
            session_type: Session code (R=Race, Q=Qualifying, FP1/FP2/FP3)
            limit_drivers: Limit telemetry to N drivers (None = all)
        """
        session_id = f"{year}_{self._slug(gp)}_{self._slug(session_type)}"
        logger.info(f"Starting export for {session_id}...")

        try:
            session = fastf1.get_session(year, gp, session_type)
            try:
                session.load(laps=True, telemetry=True, weather=True, messages=True)
            except TypeError:
                session.load()
        except Exception as e:
            logger.critical(f"Critical error loading session: {e}")
            return

        # Add SessionId to DataFrames
        session.laps['SessionId'] = session_id
        session.results['SessionId'] = session_id
        session.weather_data['SessionId'] = session_id

        # Export base data
        self._save_df(session.laps, f"laps_{session_id}.csv")
        self._save_df(session.results, f"results_{session_id}.csv")
        self._save_df(session.weather_data, f"weather_{session_id}.csv")

        # Export messages and status
        self._export_messages_and_status(session, session_id)
        self._export_session_info(session, session_id, year=year, gp=gp, session_type=session_type)

        # Export telemetry
        self._export_telemetry(session, session_id, limit_drivers=limit_drivers)

        logger.info(f"Export completed for {session_id}")

    def _export_telemetry(self, session, session_id: str, limit_drivers: int = None):
        """Export telemetry data for drivers."""
        try:
            drivers = session.drivers
            if limit_drivers:
                drivers = drivers[:limit_drivers]

            logger.info(f"Exporting telemetry for {len(drivers)} drivers...")

            telemetry_chunks = []

            for drv in drivers:
                try:
                    driver_laps = session.laps.pick_drivers(drv)

                    if driver_laps.empty:
                        continue

                    telemetry = driver_laps.get_telemetry()
                    telemetry['Driver'] = drv
                    telemetry['SessionId'] = session_id

                    telemetry_chunks.append(telemetry)

                except Exception as e:
                    logger.warning(f"Could not load telemetry for driver {drv}: {e}")
                    continue

            if not telemetry_chunks:
                logger.warning("No telemetry data collected.")
                return

            combined_telemetry = pd.concat(telemetry_chunks, ignore_index=True)
            self._save_df(combined_telemetry, f"telemetry_{session_id}.csv")

        except Exception as e:
            logger.error(f"Global telemetry export failed: {e}")

    def _export_messages_and_status(self, session, session_id: str) -> None:
        """Export Race Control Messages and Track/Session Status."""
        # Race Control Messages
        try:
            rcm = getattr(session, 'race_control_messages', None)
            if rcm is not None and not rcm.empty:
                rcm = rcm.copy()
                rcm['SessionId'] = session_id
                self._save_df(rcm, f"race_control_messages_{session_id}.csv")
            else:
                logger.warning(f"No race control messages available.")
        except Exception as e:
            logger.warning(f"Could not export race control messages: {e}")

        # Track Status
        try:
            ts = getattr(session, 'track_status', None)
            if ts is not None and not ts.empty:
                ts = ts.copy()
                ts['SessionId'] = session_id
                self._save_df(ts, f"track_status_{session_id}.csv")
            else:
                logger.warning(f"No track status available.")
        except Exception as e:
            logger.warning(f"Could not export track status: {e}")

        # Session Status
        try:
            ss = getattr(session, 'session_status', None)
            if ss is not None and not ss.empty:
                ss = ss.copy()
                ss['SessionId'] = session_id
                self._save_df(ss, f"session_status_{session_id}.csv")
            else:
                logger.warning(f"No session status available.")
        except Exception as e:
            logger.warning(f"Could not export session status: {e}")

    def _export_session_info(self, session, session_id: str, year: int, gp: str, session_type: str) -> None:
        """Export session metadata as JSON."""
        try:
            info = getattr(session, 'session_info', None)
            payload = {
                'session_id': session_id,
                'year': year,
                'gp': gp,
                'session_type': session_type,
                'session_name': getattr(session, 'name', None),
                'event_name': getattr(getattr(session, 'event', None), 'EventName', None),
                'data': info
            }
            self._save_json(payload, f"session_info_{session_id}.json")
        except Exception as e:
            logger.warning(f"Could not export session_info JSON: {e}")


def check_bronze_files(year: int, gp: str, session: str, bronze_path: Path = None):
    """Check if bronze files already exist for this session."""
    if bronze_path is None:
        bronze_path = Path(__file__).parent.parent / "lakehouse" / "01_bronze"

    # Build session_id
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
        description="Export FastF1 session data to bronze CSV/JSON files."
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
        "--session", dest="session_type", type=str, default="R",
        help="Session code: R=Race, Q=Qualifying, FP1/FP2/FP3 (default: R)"
    )
    parser.add_argument(
        "--limit-drivers", type=int, default=-1,
        help="Limit telemetry to N drivers; -1 for all (default: -1)"
    )
    parser.add_argument(
        "--bronze-path", type=str, default=None,
        help="Output folder for bronze files (default: lakehouse/01_bronze)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="FastF1 cache directory (default: cache)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-export even if files exist"
    )

    args = parser.parse_args()
    limit = None if args.limit_drivers < 0 else args.limit_drivers

    # Check existing files
    session_id, existing, missing = check_bronze_files(args.year, args.gp, args.session_type)

    print("=" * 60)
    print("F1 Data Export to Bronze Layer")
    print("=" * 60)
    print(f"Session ID: {session_id}")
    print(f"Year:       {args.year}")
    print(f"GP:         {args.gp}")
    print(f"Session:    {args.session_type}")
    print(f"Drivers:    {'All' if limit is None else limit}")
    print("-" * 60)

    if existing and not args.force:
        print(f"Existing files: {len(existing)}/{len(existing) + len(missing)}")
        for f in existing:
            print(f"  [OK] {f}")
        for f in missing:
            print(f"  [MISSING] {f}")

        if not missing:
            print("\nAll bronze files exist. Use --force to re-export.")
            return
        else:
            print("\nSome files missing. Running export...")

    exporter = F1DataExporter(bronze_path=args.bronze_path, cache_dir=args.cache_dir)
    exporter.export_session(
        year=args.year,
        gp=args.gp,
        session_type=args.session_type,
        limit_drivers=limit
    )


if __name__ == "__main__":
    main()
