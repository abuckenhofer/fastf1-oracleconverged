# Oracle Converged Database Demo - Formula 1 Data

This project demonstrates Oracle Database 26ai "Converged Database" capabilities using real Formula 1 telemetry, lap times, and race data.

## What This Demo Proves

Oracle's Converged Database allows you to use **multiple data models and workloads** in a single database, eliminating the need for specialized databases for each use case:

| Oracle Feature | Use Case | Demo Query |
|----------------|----------|------------|
| **Relational** | Facts/dimensions, analytics, window functions | Pace degradation analysis, driver consistency ranking |
| **JSON** | Document storage, flexible schema, JSON_TABLE | Race control messages, session metadata extraction |
| **Time Series** | High-frequency telemetry, status timelines | Braking events, DRS usage, weather correlation |
| **Spatial** | SDO_GEOMETRY, proximity queries | Track position analysis, car proximity detection |
| **Graph** | SQL Property Graph, GRAPH_TABLE | Teammate relationships, driver-team network |
| **Vector** | Semantic similarity search, embeddings | "dangerous driving" finds INCIDENT messages without keyword match |

## Prerequisites

- **Python 3.13+** with `uv` package manager
- **Oracle Database 23ai or above** (on-prem) - for full feature set
  - Oracle 21c/19c works for most features except VECTOR and Property Graph
- **Docker** (optional) - for running Oracle XE/Free locally

### Quick Start with Oracle Free in Docker

```bash
# Oracle Free (recommended - full feature set)
docker run -d --name oraclexe \
  -p 1521:1521 \
  -e ORACLE_PASSWORD=YourPassword123 \
  -v oracle-volume:/opt/oracle/oradata \
  gvenzl/oracle-free

# Or Oracle XE (21c - no VECTOR/Graph support)
# docker run -d --name oraclexe \
#   -p 1521:1521 \
#   -e ORACLE_PASSWORD=YourPassword123 \
#   -v oracle-volume:/opt/oracle/oradata \
#   gvenzl/oracle-xe
```

Wait for the container to be healthy (~2-3 minutes):
```bash
docker logs -f oraclexe
```

**Note:** Use `FREEPDB1` as service name for oracle-free, `XEPDB1` for oracle-xe.

## Installation

```bash
# Clone and enter the project
cd fastf1-oracleconverged

# Install dependencies with uv
uv sync
```

## Configuration

Copy the environment template and edit with your Oracle credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
ORA_USER=myuser
ORA_PASSWORD=YourPassword123
# Use FREEPDB1 for gvenzl/oracle-free, XEPDB1 for gvenzl/oracle-xe
ORA_DSN=localhost:1521/FREEPDB1
```
Set appropriate privileges to the user so that the user can create resources like tables and query dictionary tables like v$database. Enable the In-Memory option by setting [inmemory_size](https://blogs.oracle.com/coretec/oracle-database-in-memory-der-schnelle-einstieg).
Alternatively, set environment variables directly:

```bash
# Linux/macOS
export ORA_USER=v
export ORA_PASSWORD=YourPassword123
export ORA_DSN=localhost:1521/FREEPDB1

# Windows PowerShell
$env:ORA_USER = "myuser"
$env:ORA_PASSWORD = "YourPassword123"
$env:ORA_DSN = "localhost:1521/FREEPDB1"
```

## Running the Demo

### Step 1: Verify Oracle Connection

```bash
uv run python -m src.00_env_check
```

Expected output:
```
============================================================
Oracle Converged Database - Environment Check
============================================================

Connecting to Oracle as abu@localhost:1521/FREEPDB1...
------------------------------------------------------------
Oracle Version: Oracle AI Database 26ai Free Release 23.26.0.0.0 - Develop, Learn, and Run for Free
Database Name:  FREE
Is CDB:         YES
Current PDB:    FREEPDB1
Current Schema: ABU
Session User:   ABU

------------------------------------------------------------
Checking Oracle 26ai Features:
  JSON Support:       Available
  VECTOR Type:        Available
  Property Graphs:    Available
  Spatial (SDO):      Available
  In-Memory Store:    Available (0.4 GB)

------------------------------------------------------------
Connection successful!

Environment check PASSED
```

### Step 2: Export F1 Data

Export the F1 telemetry data to CSV/JSON files in `lakehouse/01_bronze/`:

```bash
uv run python -m src.10_export_f1_data --year 2024 --gp "Singapore" --session R
```

### Step 3: Load Data into Oracle

```bash
uv run python -m src.20_load_oracle --year 2024 --gp "Singapore" --session R
```

This creates all tables and loads:
- Dimension tables (events, drivers, teams)
- Fact tables (results, laps)
- Time series data (telemetry, weather, status)
- Spatial data (SDO_GEOMETRY points)
- JSON documents (raw messages, session info)

### Step 4: Build Vector Embeddings

```bash
uv run python -m src.30_build_embeddings --year 2024 --gp "Singapore" --session R
```

This generates embeddings for race control messages using:
- `sentence-transformers` model (all-MiniLM-L6-v2) if available
- Deterministic hash-based fallback otherwise

### Step 5: Run Demo Use Cases

```bash
uv run python -m src.40_run_use_cases --year 2024 --gp "Singapore" --session R
```

Or run specific features:
```bash
uv run python -m src.40_run_use_cases --features relational json spatial
```

### Step 6: Generate Visualizations

```bash
uv run python -m src.50_visualizations --circuit Singapore
```

This creates interactive HTML visualizations in `output/visualizations/`. FastF1 contains X/y coordinates but not lat/lon. Therefore those coordinates are transformed into lat/lon in 50_visualizations. Currently Singapore only is configured for the transformation. Other circuits need transformation settings in CIRCUIT_CONFIGS. 

All visualizations use a consistent professional F1 theme (black background, gold/red accents):

| Oracle Capability | File | Description |
|-------------------|------|-------------|
| **Spatial** | `circuit_map_singapore.html` | Interactive Folium map with dark theme, telemetry overlay (x/y → lat/lon) |
| **Spatial** | `speed_trace.html` | Circuit layout colored by speed (blue→gold→red gradient) |
| **Graph/Relational** | `overtake_network.html` | Directed network showing who overtook whom with team colors |
| **Vector** | `embeddings_pca.html` | PCA projection of 384-dim message embeddings to 2D |
| **Vector** | `semantic_search_comparison.html` | Keyword vs semantic search results comparison |
| **Vector** | `messages_timeline.html` | Race control messages timeline by category |
| **Relational** | `data_model.html` | Sankey diagram showing star schema data flow |
| **Relational** | `table_summary.html` | Table row counts by type |
| **JSON** | `json_structure.html` | Treemap of JSON document hierarchy |

Run specific visualizations:
```bash
# Spatial visualizations
uv run python -m src.50_visualizations --viz circuit speed

# Vector visualizations
uv run python -m src.50_visualizations --viz embeddings

# Overtake network (Graph/Relational)
uv run python -m src.50_visualizations --viz overtake

# Multiple capabilities
uv run python -m src.50_visualizations --viz overtake model json

# Available options: circuit, speed, overtake, embeddings, semantic, timeline, model, summary, json, all
```

## Project Structure

```
fastf1-oracleconverged/
├── pyproject.toml              # Project dependencies
├── README.md                   # This file
├── .env.example                # Environment variables template
├── sql/
│   └── 10_views.sql            # Helper views for common query patterns
├── src/
│   ├── __init__.py
│   ├── 00_env_check.py         # Oracle connectivity + feature check
│   ├── 10_export_f1_data.py    # FastF1 data exporter
│   ├── 20_load_oracle.py       # Data loader (includes schema DDL)
│   ├── 30_build_embeddings.py  # Vector embeddings
│   ├── 40_run_use_cases.py     # Demo runner
│   └── 50_visualizations.py    # Interactive visualizations
├── lakehouse/
│   └── 01_bronze/              # Exported CSV/JSON files
├── cache                       # Buffering FastF1 data
└── output/
    └── visualizations/         # Generated HTML visualizations
```

## Data Model

### Dimension Tables

| Table | Description |
|-------|-------------|
| `dim_event` | Race session dimension |
| `dim_driver` | Driver dimension |
| `dim_team` | Team dimension |
| `dim_compound` | Tyre compound dimension (SOFT, MEDIUM, HARD, etc.) |
| `dim_result_status` | Result status dimension (Finished, +1 Lap, DNF, etc.) |
| `bridge_driver_team` | Driver-team relationship per event |

### Fact Tables

| Table | Description |
|-------|-------------|
| `fact_result` | Race results (FK to dim_result_status) |
| `fact_lap` | Lap-level timing data (FK to dim_compound) |
| `fact_telemetry` | High-frequency car telemetry |
| `fact_weather` | Weather readings |
| `fact_track_status` | Yellow flags, SC, VSC |
| `fact_session_status` | Session state changes |

### Specialized Tables

| Table | Type | Description |
|-------|------|-------------|
| `f1_raw_documents` | JSON | Raw documents with IS JSON constraint |
| `telemetry_spatial` | Spatial | SDO_GEOMETRY points with spatial index |
| `f1_messages` | Vector | Messages with VECTOR embeddings |
| `f1_graph` | Graph | SQL Property Graph over dimensions |

## Use Cases

The Python demo runner (`src/40_run_use_cases.py`) demonstrates these Oracle capabilities:

### 1. Relational Analytics
- Pace degradation per stint (LAG, AVG OVER)
- Driver consistency ranking (STDDEV, RANK)
- Position changes analysis
- Team performance comparison

### 2. JSON Queries
- JSON_TABLE extraction from documents
- JSON_VALUE for nested navigation
- Flag summary aggregation

### 3. Time Series
- Heavy braking events detection
- DRS usage patterns
- Weather correlation
- **Gap filling with MODEL clause** (interpolation for missing data)
- Continuous timeline generation with forward fill

### 4. Spatial
- SDO_WITHIN_DISTANCE proximity search
- Track sector analysis by coordinates
- Car proximity detection

### 5. Graph
- Teammate discovery via GRAPH_TABLE
- Driver-team-event network traversal

### 6. Vector
- True semantic search using **Oracle VECTOR_DISTANCE** function
- "dangerous driving" finds INCIDENT messages without keyword match
- Hybrid queries: vector similarity + SQL filters in single query
- Seed-based similarity: find messages similar to a reference message

## Troubleshooting

### Connection Issues

1. **ORA-12541: TNS:no listener**
   - Ensure Oracle is running and listener is started
   - Check DSN format: `host:port/service_name`

2. **ORA-01017: invalid username/password**
   - Verify ORA_USER and ORA_PASSWORD

3. **VECTOR type not supported**
   - Requires Oracle 23ai or above; demo falls back gracefully

### Data Issues

1. **Event not found**
   - Run `20_load_oracle.py` before `30_build_embeddings.py`

2. **Empty telemetry**
   - Check if bronze files exist in `lakehouse/01_bronze/`

### Model Download Issues

If `sentence-transformers` can't download the model:
- The demo uses a deterministic hash-based fallback
- Embeddings will be consistent but not semantically meaningful

## License

This project is for demonstration purposes. F1 data is sourced via the FastF1 library.
