"""
F1 Data Visualizations Module

Creates professional visualizations for Oracle Converged Database demo.
Theme: Black background, gold/red accents (F1 premium aesthetic)

Visualizations:
1. Spatial: Circuit map with telemetry overlay, speed trace
2. Graph/Relational: Overtake network
3. Vector: PCA projection of embeddings, semantic search comparison
4. Relational: Star schema data model
5. JSON: Document structure treemap
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

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


# =============================================================================
# PROFESSIONAL F1 THEME CONFIGURATION
# =============================================================================

class F1Theme:
    """F1-inspired professional color theme for all visualizations."""

    # Core colors
    BACKGROUND = '#0d0d0d'       # Deep black
    PAPER = '#1a1a1a'            # Slightly lighter black for contrast

    # Accent colors
    GOLD = '#D4AF37'             # Premium gold
    GOLD_BRIGHT = '#FFD700'      # Bright gold for highlights
    RED = '#E10600'              # F1 red
    RED_DARK = '#B30000'         # Darker red

    # Text colors
    TEXT_PRIMARY = '#FFFFFF'     # White
    TEXT_SECONDARY = '#B0B0B0'   # Light gray
    TEXT_MUTED = '#666666'       # Muted gray

    # Grid and borders
    GRID = '#2a2a2a'             # Dark grid lines
    BORDER = '#404040'           # Border color

    # Speed gradient (slow to fast)
    SPEED_SLOW = '#1E3A5F'       # Dark blue (slow)
    SPEED_MID = '#D4AF37'        # Gold (medium)
    SPEED_FAST = '#E10600'       # Red (fast)

    # Category colors for data types
    DIMENSION = '#4A90D9'        # Blue for dimensions
    FACT = '#50C878'             # Green for facts
    JSON_COLOR = '#FFA500'       # Orange for JSON
    VECTOR = '#9B59B6'           # Purple for Vector
    SPATIAL = '#E74C3C'          # Red for Spatial

    @classmethod
    def plotly_layout(cls, title: str = '', subtitle: str = '',
                      width: int = 1000, height: int = 700) -> dict:
        """Return standard Plotly layout with F1 theme."""
        title_text = title
        if subtitle:
            title_text = f'{title}<br><sup style="color:{cls.TEXT_SECONDARY}">{subtitle}</sup>'

        return {
            'title': {
                'text': title_text,
                'x': 0.5,
                'font': {'color': cls.GOLD, 'size': 20}
            },
            'paper_bgcolor': cls.BACKGROUND,
            'plot_bgcolor': cls.PAPER,
            'font': {'color': cls.TEXT_PRIMARY, 'family': 'Arial, sans-serif'},
            'width': width,
            'height': height,
            'xaxis': {
                'gridcolor': cls.GRID,
                'zerolinecolor': cls.GRID,
                'tickfont': {'color': cls.TEXT_SECONDARY}
            },
            'yaxis': {
                'gridcolor': cls.GRID,
                'zerolinecolor': cls.GRID,
                'tickfont': {'color': cls.TEXT_SECONDARY}
            },
            'legend': {
                'bgcolor': 'rgba(26,26,26,0.8)',
                'bordercolor': cls.BORDER,
                'borderwidth': 1,
                'font': {'color': cls.TEXT_PRIMARY}
            }
        }


@dataclass
class CircuitConfig:
    """Configuration for converting x/y telemetry to lat/lon."""
    name: str
    country: str
    anchor_lat: float
    anchor_lon: float
    rotation: float  # degrees
    scale_factor: float

    def xy_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert telemetry x/y coordinates to latitude/longitude."""
        rot_rad = math.radians(self.rotation)
        sin_r = math.sin(rot_rad)
        cos_r = math.cos(rot_rad)

        lat = self.anchor_lat + ((x * sin_r + y * cos_r) * self.scale_factor)
        lon = self.anchor_lon + ((x * cos_r - y * sin_r) * self.scale_factor)

        return lat, lon


# Circuit configurations
CIRCUIT_CONFIGS = {
    "Singapore": CircuitConfig(
        name="Marina Bay Street Circuit",
        country="Singapore",
        anchor_lat=1.2910,
        anchor_lon=103.8634,
        rotation=0.0,
        scale_factor=0.000000899
    ),
    "Monaco": CircuitConfig(
        name="Circuit de Monaco",
        country="Monaco",
        anchor_lat=43.7347,
        anchor_lon=7.4206,
        rotation=0.0,
        scale_factor=0.000000899
    ),
}


class F1Visualizer:
    """Creates professional F1 visualizations from Oracle data."""

    def __init__(self, user: str, password: str, dsn: str):
        self.user = user
        self.password = password
        self.dsn = dsn
        self.connection = None
        self.output_dir = Path(__file__).parent.parent / "output" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme = F1Theme()

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

    # =========================================================================
    # 1. SPATIAL: Circuit Map with Telemetry (Folium)
    # =========================================================================

    def create_circuit_map(self, circuit_name: str = "Singapore",
                           driver_codes: List[str] = None,
                           sample_rate: int = 5) -> str:
        """
        Create an interactive map showing the circuit with telemetry overlay.
        Uses dark map tiles with gold/red speed coloring.
        """
        import folium

        logger.info(f"Creating circuit map for {circuit_name}...")

        config = CIRCUIT_CONFIGS.get(circuit_name)
        if not config:
            logger.error(f"No circuit config found for {circuit_name}")
            return None

        cursor = self.connection.cursor()

        driver_filter = ""
        if driver_codes:
            codes = ",".join(f"'{c}'" for c in driver_codes)
            driver_filter = f"AND d.driver_code IN ({codes})"

        cursor.execute(f"""
            SELECT
                d.driver_code,
                ts.session_time_sec,
                ts.geom.SDO_POINT.X as x,
                ts.geom.SDO_POINT.Y as y,
                ts.speed,
                ts.brake
            FROM telemetry_spatial ts
            JOIN dim_driver d ON ts.driver_id = d.driver_id
            WHERE ts.speed IS NOT NULL
              AND ts.geom IS NOT NULL
              {driver_filter}
            ORDER BY d.driver_code, ts.session_time_sec
        """)

        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            logger.warning("No telemetry data found")
            return None

        rows = rows[::sample_rate]
        logger.info(f"Processing {len(rows)} telemetry points...")

        data = []
        for driver_code, time_sec, x, y, speed, brake in rows:
            if x is None or y is None:
                continue
            lat, lon = config.xy_to_latlon(float(x), float(y))
            data.append({
                'driver': driver_code,
                'time': time_sec,
                'lat': lat,
                'lon': lon,
                'speed': speed or 0,
                'brake': brake or 0
            })

        df = pd.DataFrame(data)

        center_lat = df['lat'].mean()
        center_lon = df['lon'].mean()

        # Dark-themed map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='cartodbdark_matter'  # Dark theme
        )

        # Speed color gradient: dark blue -> gold -> red
        max_speed = df['speed'].max()
        min_speed = df['speed'].min()

        def speed_to_color(speed):
            if max_speed == min_speed:
                return F1Theme.GOLD
            ratio = (speed - min_speed) / (max_speed - min_speed)
            # Blue (slow) -> Gold (medium) -> Red (fast)
            if ratio < 0.5:
                # Blue to Gold
                r = int(0x1E + (0xD4 - 0x1E) * (ratio * 2))
                g = int(0x3A + (0xAF - 0x3A) * (ratio * 2))
                b = int(0x5F + (0x37 - 0x5F) * (ratio * 2))
            else:
                # Gold to Red
                r = int(0xD4 + (0xE1 - 0xD4) * ((ratio - 0.5) * 2))
                g = int(0xAF - 0xAF * ((ratio - 0.5) * 2))
                b = int(0x37 - 0x37 * ((ratio - 0.5) * 2))
            return f'#{r:02x}{g:02x}{b:02x}'

        # Add speed-colored points
        for _, row in df.iloc[::3].iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=2,
                color=speed_to_color(row['speed']),
                fill=True,
                fillOpacity=0.8,
                popup=f"{row['driver']}: {row['speed']:.0f} km/h"
            ).add_to(m)

        # Add braking zones (bright red markers)
        brake_df = df[df['brake'] == 1].iloc[::10]
        for _, row in brake_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                color=F1Theme.RED,
                fill=True,
                fillColor=F1Theme.RED,
                fillOpacity=0.9
            ).add_to(m)

        # Professional legend with F1 theme
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: {F1Theme.BACKGROUND}; padding: 15px; border-radius: 8px;
                    border: 2px solid {F1Theme.GOLD}; font-size: 12px; color: {F1Theme.TEXT_PRIMARY};
                    font-family: Arial, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.5);">
            <b style="color: {F1Theme.GOLD}; font-size: 14px;">SPEED LEGEND</b><br><br>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="background: {F1Theme.SPEED_SLOW}; width: 20px; height: 12px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                Slow (corners)
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="background: {F1Theme.GOLD}; width: 20px; height: 12px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                Medium
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="background: {F1Theme.RED}; width: 20px; height: 12px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                Fast / Braking
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add title overlay
        title_html = f'''
        <div style="position: fixed; top: 20px; left: 50%; transform: translateX(-50%); z-index: 1000;
                    background-color: {F1Theme.BACKGROUND}; padding: 12px 24px; border-radius: 8px;
                    border: 2px solid {F1Theme.GOLD}; font-family: Arial, sans-serif;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.5);">
            <span style="color: {F1Theme.GOLD}; font-size: 18px; font-weight: bold;">
                {config.name.upper()}
            </span>
            <span style="color: {F1Theme.TEXT_SECONDARY}; font-size: 14px; margin-left: 10px;">
                Spatial Telemetry
            </span>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        output_path = self.output_dir / f"circuit_map_{circuit_name.lower()}.html"
        m.save(str(output_path))
        logger.info(f"Saved circuit map to {output_path}")

        return str(output_path)

    def create_speed_trace_plot(self, driver_codes: List[str] = None) -> str:
        """Create Plotly visualization of speed traces on circuit layout."""
        import plotly.graph_objects as go

        logger.info("Creating speed trace plot...")

        cursor = self.connection.cursor()

        driver_filter = ""
        if driver_codes:
            codes = ",".join(f"'{c}'" for c in driver_codes)
            driver_filter = f"AND d.driver_code IN ({codes})"

        cursor.execute(f"""
            SELECT
                d.driver_code,
                ts.geom.SDO_POINT.X as x,
                ts.geom.SDO_POINT.Y as y,
                ts.speed,
                ts.throttle,
                ts.brake
            FROM telemetry_spatial ts
            JOIN dim_driver d ON ts.driver_id = d.driver_id
            WHERE ts.speed IS NOT NULL
              {driver_filter}
            ORDER BY d.driver_code, ts.session_time_sec
        """)

        rows = cursor.fetchall()
        cursor.close()

        rows = rows[::20]
        df = pd.DataFrame(rows, columns=['driver', 'x', 'y', 'speed', 'throttle', 'brake'])
        df = df.dropna()

        # Create custom colorscale matching theme
        colorscale = [
            [0.0, F1Theme.SPEED_SLOW],
            [0.5, F1Theme.GOLD],
            [1.0, F1Theme.RED]
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['speed'],
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text='Speed (km/h)', font=dict(color=F1Theme.TEXT_PRIMARY)),
                    tickfont=dict(color=F1Theme.TEXT_SECONDARY),
                    bgcolor=F1Theme.PAPER,
                    bordercolor=F1Theme.BORDER,
                    borderwidth=1
                ),
                showscale=True
            ),
            text=[f"{d}: {s:.0f} km/h" for d, s in zip(df['driver'], df['speed'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

        layout = F1Theme.plotly_layout(
            title='CIRCUIT LAYOUT',
            subtitle='Speed trace from spatial telemetry (SDO_GEOMETRY)',
            width=1000,
            height=800
        )
        layout['xaxis'].update(scaleanchor='y', scaleratio=1, showgrid=False,
                               showticklabels=False, title='')
        layout['yaxis'].update(showgrid=False, showticklabels=False, title='')

        fig.update_layout(**layout)

        output_path = self.output_dir / "speed_trace.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved speed trace to {output_path}")

        return str(output_path)

    # =========================================================================
    # 2. GRAPH/RELATIONAL: Overtake Network
    # =========================================================================

    def create_overtake_network(self) -> str:
        """
        Create professional directed network showing overtakes during the race.
        Uses relational lap data to detect position changes.
        """
        import networkx as nx
        import plotly.graph_objects as go

        logger.info("Creating overtake network...")

        cursor = self.connection.cursor()

        # Get driver info
        cursor.execute("""
            SELECT d.driver_id, d.driver_code, d.full_name, t.team_color, r.final_position
            FROM dim_driver d
            JOIN bridge_driver_team bdt ON d.driver_id = bdt.driver_id
            JOIN dim_team t ON bdt.team_id = t.team_id
            LEFT JOIN fact_result r ON d.driver_id = r.driver_id
        """)
        driver_info = {}
        for driver_id, code, name, color, pos in cursor.fetchall():
            driver_info[driver_id] = {
                'code': code, 'name': name,
                'color': f'#{color}' if color else F1Theme.TEXT_MUTED,
                'position': pos
            }

        # Detect overtakes by comparing positions between consecutive laps
        cursor.execute("""
            SELECT
                l1.driver_id,
                l1.lap_number,
                l1.position AS pos_before,
                l2.position AS pos_after
            FROM fact_lap l1
            JOIN fact_lap l2 ON l1.driver_id = l2.driver_id
                AND l2.lap_number = l1.lap_number + 1
            WHERE l1.position IS NOT NULL
              AND l2.position IS NOT NULL
              AND l1.position != l2.position
            ORDER BY l1.lap_number, l1.driver_id
        """)
        position_changes = cursor.fetchall()

        # Build lap-by-lap position map
        cursor.execute("""
            SELECT driver_id, lap_number, position
            FROM fact_lap
            WHERE position IS NOT NULL
            ORDER BY lap_number, position
        """)
        lap_positions = {}
        for driver_id, lap, pos in cursor.fetchall():
            if lap not in lap_positions:
                lap_positions[lap] = {}
            lap_positions[lap][pos] = driver_id

        cursor.close()

        # Detect overtakes
        overtakes = {}

        for driver_id, lap, pos_before, pos_after in position_changes:
            if pos_after < pos_before:  # Driver gained position
                positions_lost_to = range(pos_after, pos_before)
                for lost_pos in positions_lost_to:
                    if lap in lap_positions and lost_pos in lap_positions[lap]:
                        overtaken_id = lap_positions[lap][lost_pos]
                        if overtaken_id != driver_id:
                            key = (driver_id, overtaken_id)
                            overtakes[key] = overtakes.get(key, 0) + 1

        if not overtakes:
            logger.warning("No overtakes detected")
            return None

        # Build directed graph
        G = nx.DiGraph()

        for driver_id, info in driver_info.items():
            G.add_node(info['code'],
                      name=info['name'],
                      color=info['color'],
                      position=info['position'])

        max_overtakes = max(overtakes.values())
        for (overtaker_id, overtaken_id), count in overtakes.items():
            if overtaker_id in driver_info and overtaken_id in driver_info:
                code1 = driver_info[overtaker_id]['code']
                code2 = driver_info[overtaken_id]['code']
                G.add_edge(code1, code2, weight=count, normalized=count/max_overtakes)

        # Calculate net gains
        net_gains = {}
        for node in G.nodes():
            gained = sum(d['weight'] for _, _, d in G.out_edges(node, data=True))
            lost = sum(d['weight'] for _, _, d in G.in_edges(node, data=True))
            net_gains[node] = gained - lost

        # Layout with more spacing
        pos = nx.spring_layout(G, k=3, iterations=150, seed=42)

        fig = go.Figure()

        # Draw edges with gold/red coloring based on weight
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('normalized', 0.5)

            # Gold for single overtakes, red for multiple
            if weight < 0.3:
                edge_color = f'rgba(212, 175, 55, {0.4 + weight})'  # Gold
            else:
                edge_color = f'rgba(225, 6, 0, {0.4 + weight * 0.5})'  # Red

            # Draw line
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=1 + weight * 6, color=edge_color),
                hoverinfo='text',
                hovertext=f"{edge[0]} overtook {edge[1]}: {edge[2].get('weight', 0)}x",
                showlegend=False
            ))

            # Add arrowhead
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowwidth=1 + weight * 4,
                arrowcolor=edge_color
            )

        # Add nodes with team colors and sizing based on net gains
        node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(node[1].get('color', F1Theme.TEXT_MUTED))
            net = net_gains.get(node[0], 0)
            net_str = f"+{net}" if net > 0 else str(net)
            position = node[1].get('position', '?')
            node_text.append(f"<b>{node[0]}</b><br>{node[1].get('name', '')}<br>Finished: P{position}<br>Net positions: {net_str}")
            node_sizes.append(30 + abs(net) * 4)

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color=F1Theme.GOLD)
            ),
            text=[t.split('<br>')[0].replace('<b>', '').replace('</b>', '') for t in node_text],
            textposition='top center',
            textfont=dict(color=F1Theme.TEXT_PRIMARY, size=11),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))

        layout = F1Theme.plotly_layout(
            title='OVERTAKE NETWORK',
            subtitle='Arrows show who overtook whom | Line thickness = frequency | Node size = net positions gained/lost',
            width=1100,
            height=850
        )
        layout['xaxis'].update(showgrid=False, zeroline=False, showticklabels=False)
        layout['yaxis'].update(showgrid=False, zeroline=False, showticklabels=False)

        fig.update_layout(**layout)

        output_path = self.output_dir / "overtake_network.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved overtake network to {output_path}")

        return str(output_path)

    # =========================================================================
    # 3. VECTOR: PCA of Message Embeddings
    # =========================================================================

    def create_embedding_visualization(self) -> str:
        """
        Create professional PCA visualization of race control message embeddings.
        """
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        logger.info("Creating embedding visualization...")

        cursor = self.connection.cursor()

        try:
            cursor.execute("""
                SELECT
                    doc_id,
                    JSON_VALUE(payload, '$.Category') as category,
                    JSON_VALUE(payload, '$.Flag') as flag,
                    JSON_VALUE(payload, '$.Message') as message,
                    JSON_VALUE(payload, '$.Time') as time_str,
                    JSON_VALUE(payload, '$.Lap' RETURNING NUMBER) as lap
                FROM f1_raw_documents
                WHERE doc_type = 'race_control_message'
                ORDER BY doc_id
            """)
            rows = cursor.fetchall()
            cursor.close()
        except oracledb.DatabaseError as e:
            cursor.close()
            logger.warning(f"Could not fetch messages: {e}")
            return None

        if not rows:
            logger.warning("No race control messages found")
            return None

        logger.info(f"Found {len(rows)} race control messages")

        from datetime import datetime

        timestamps = []
        for row in rows:
            time_str = row[4]
            if time_str:
                try:
                    ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(ts)
                except:
                    pass

        race_start = min(timestamps) if timestamps else None

        data = []
        texts = []
        for doc_id, category, flag, message, time_str, lap in rows:
            if message:
                time_sec = 0
                if time_str and race_start:
                    try:
                        ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        time_sec = (ts - race_start).total_seconds()
                    except:
                        pass

                data.append({
                    'doc_id': doc_id,
                    'category': category or 'Unknown',
                    'flag': flag or '',
                    'message': message,
                    'text_short': message[:50] + '...' if len(message) > 50 else message,
                    'time_sec': time_sec,
                    'lap': lap or 0
                })
                texts.append(message)

        if not texts:
            logger.warning("No message text found")
            return None

        df = pd.DataFrame(data)

        # Generate embeddings
        logger.info("Generating embeddings with sentence-transformers...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        except Exception as e:
            logger.warning(f"sentence-transformers failed: {e}")
            logger.info("Using hash-based fallback embeddings...")
            import hashlib
            embeddings = []
            for text in texts:
                hash_bytes = hashlib.sha256(text.encode()).digest()
                emb = [float(b) / 255.0 - 0.5 for b in hash_bytes[:32]]
                emb.extend([float(b) / 255.0 - 0.5 for b in hash_bytes[16:48]])
                embeddings.append(emb)
            embeddings = np.array(embeddings)
            model = None

        # PCA reduction
        logger.info("Applying PCA...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_scaled)

        df['PC1'] = embeddings_2d[:, 0]
        df['PC2'] = embeddings_2d[:, 1]

        # Category colors - gold, red, and complementary colors
        category_colors = {
            'Flag': F1Theme.GOLD,
            'SafetyCar': F1Theme.RED,
            'Drs': '#00CED1',  # Cyan/teal
            'Other': F1Theme.TEXT_MUTED,
            'Unknown': F1Theme.TEXT_MUTED
        }

        fig = go.Figure()

        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            color = category_colors.get(category, F1Theme.TEXT_SECONDARY)

            fig.add_trace(go.Scatter(
                x=cat_df['PC1'],
                y=cat_df['PC2'],
                mode='markers',
                name=category,
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(width=1, color=F1Theme.BACKGROUND),
                    opacity=0.85
                ),
                text=cat_df['text_short'],
                hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>'
            ))

        layout = F1Theme.plotly_layout(
            title='MESSAGE EMBEDDINGS',
            subtitle=f'PCA projection of 384-dim vectors | Variance explained: {sum(pca.explained_variance_ratio_):.1%}',
            width=1100,
            height=750
        )
        layout['xaxis'].update(title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        layout['yaxis'].update(title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        layout['legend'].update(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)

        fig.update_layout(**layout)

        output_path = self.output_dir / "embeddings_pca.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved embeddings PCA to {output_path}")

        # Create timeline view
        self._create_timeline_viz(df)

        # Create semantic search comparison
        if model:
            self._create_semantic_search_viz(df, texts, embeddings, model)

        return str(output_path)

    def _create_timeline_viz(self, df: pd.DataFrame) -> str:
        """Create professional timeline visualization."""
        import plotly.graph_objects as go

        df['race_minute'] = df['time_sec'] / 60
        categories = df['category'].unique()
        category_map = {cat: i for i, cat in enumerate(categories)}
        df['y_pos'] = df['category'].map(category_map)

        category_colors = {
            'Flag': F1Theme.GOLD,
            'SafetyCar': F1Theme.RED,
            'Drs': '#00CED1',
            'Other': F1Theme.TEXT_MUTED,
            'Unknown': F1Theme.TEXT_MUTED
        }

        fig = go.Figure()

        for category in categories:
            cat_df = df[df['category'] == category]
            color = category_colors.get(category, F1Theme.TEXT_SECONDARY)

            fig.add_trace(go.Scatter(
                x=cat_df['race_minute'],
                y=cat_df['y_pos'],
                mode='markers',
                name=category,
                marker=dict(size=14, color=color, symbol='diamond',
                           line=dict(width=1, color=F1Theme.BACKGROUND)),
                text=cat_df['text_short'],
                hovertemplate='<b>%{text}</b><br>Lap: %{customdata}<br>Time: %{x:.1f} min<extra></extra>',
                customdata=cat_df['lap']
            ))

        layout = F1Theme.plotly_layout(
            title='RACE CONTROL TIMELINE',
            subtitle='Messages by race time from session start',
            width=1200,
            height=500
        )
        layout['xaxis'].update(title='Race Time (minutes)', dtick=10)
        layout['yaxis'].update(
            tickmode='array',
            tickvals=list(category_map.values()),
            ticktext=list(category_map.keys()),
            title=''
        )
        layout['legend'].update(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)

        fig.update_layout(**layout)

        output_path = self.output_dir / "messages_timeline.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved messages timeline to {output_path}")

        return str(output_path)

    def _create_semantic_search_viz(self, df: pd.DataFrame, texts: list,
                                     embeddings: np.ndarray, model) -> str:
        """Create professional semantic search comparison visualization."""
        import plotly.graph_objects as go

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Semantic search for "dangerous driving"
        query = 'dangerous driving or collision between cars'
        query_emb = model.encode([query])[0]

        similarities = []
        for i, emb in enumerate(embeddings):
            sim = cosine_similarity(query_emb, emb)
            has_keyword = 'INCIDENT' in texts[i].upper()
            similarities.append((sim, texts[i], has_keyword))

        similarities.sort(reverse=True)
        top_results = similarities[:6]

        fig = go.Figure()

        # Bars colored by whether keyword was found
        colors = [F1Theme.GOLD if has_kw else F1Theme.RED for _, _, has_kw in top_results]

        fig.add_trace(go.Bar(
            y=[f"#{i+1}" for i in range(len(top_results))],
            x=[sim for sim, _, _ in top_results],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=1, color=F1Theme.TEXT_PRIMARY)
            ),
            text=[msg[:40] + '...' if len(msg) > 40 else msg for _, msg, _ in top_results],
            textposition='inside',
            textfont=dict(color=F1Theme.BACKGROUND, size=10),
            hovertemplate='<b>%{text}</b><br>Similarity: %{x:.3f}<extra></extra>'
        ))

        layout = F1Theme.plotly_layout(
            title='SEMANTIC SEARCH RESULTS',
            subtitle=f'Query: "{query}" | Gold = contains "INCIDENT" | Red = found by meaning only',
            width=950,
            height=450
        )
        layout['xaxis'].update(title='Cosine Similarity Score', range=[0, max(s for s, _, _ in top_results) * 1.1])
        layout['yaxis'].update(title='Result Rank', autorange='reversed')

        fig.update_layout(**layout)

        # Add insight annotation
        fig.add_annotation(
            x=0.3, y=5.2,
            text='Vector search finds INCIDENT messages<br>without using keyword "incident"',
            showarrow=False,
            font=dict(size=11, color=F1Theme.TEXT_SECONDARY),
            bgcolor=F1Theme.PAPER,
            borderpad=6,
            bordercolor=F1Theme.BORDER,
            borderwidth=1
        )

        output_path = self.output_dir / "semantic_search_comparison.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved semantic search comparison to {output_path}")

        return str(output_path)

    # =========================================================================
    # 4. RELATIONAL: Data Model Visualization
    # =========================================================================

    def create_data_model_diagram(self) -> str:
        """Create professional star schema visualization."""
        import plotly.graph_objects as go

        logger.info("Creating data model visualization...")

        cursor = self.connection.cursor()

        tables = [
            'dim_event', 'dim_driver', 'dim_team', 'bridge_driver_team',
            'fact_result', 'fact_lap', 'fact_telemetry', 'fact_weather',
            'f1_raw_documents', 'f1_messages', 'telemetry_spatial'
        ]

        counts = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except:
                counts[table] = 0

        cursor.close()

        labels = [
            'Bronze CSV/JSON',
            'dim_event', 'dim_driver', 'dim_team',
            'fact_result', 'fact_lap', 'fact_telemetry', 'fact_weather',
            'JSON Documents', 'Vector Embeddings', 'Spatial Points',
            'bridge_driver_team'
        ]

        links = {
            'source': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2,
                      3],
            'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      4, 5, 6, 7, 8, 9,
                      4, 5, 6, 11,
                      11],
            'value': [
                counts.get('dim_event', 1),
                counts.get('dim_driver', 1),
                counts.get('dim_team', 1),
                counts.get('fact_result', 1),
                counts.get('fact_lap', 1),
                max(counts.get('fact_telemetry', 1) // 1000, 1),
                counts.get('fact_weather', 1),
                counts.get('f1_raw_documents', 1),
                counts.get('f1_messages', 1),
                max(counts.get('telemetry_spatial', 1) // 1000, 1),
                counts.get('bridge_driver_team', 1),
                5, 10, 50, 5, 5, 5,
                5, 10, 50, 5,
                5
            ]
        }

        # Theme-consistent colors
        colors = [
            F1Theme.TEXT_MUTED,  # Bronze
            F1Theme.DIMENSION, F1Theme.DIMENSION, F1Theme.DIMENSION,  # Dimensions
            F1Theme.FACT, F1Theme.FACT, F1Theme.FACT, F1Theme.FACT,  # Facts
            F1Theme.JSON_COLOR, F1Theme.VECTOR, F1Theme.SPATIAL,  # Specialized
            F1Theme.TEXT_SECONDARY  # Bridge
        ]

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color=F1Theme.GOLD, width=1),
                label=labels,
                color=colors
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                color=f'rgba(212, 175, 55, 0.2)'  # Gold with transparency
            )
        )])

        layout = F1Theme.plotly_layout(
            title='DATA MODEL',
            subtitle='Oracle Converged Database star schema',
            width=1200,
            height=700
        )

        fig.update_layout(**layout)

        output_path = self.output_dir / "data_model.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved data model to {output_path}")

        # Create table summary
        self._create_table_summary(counts)

        return str(output_path)

    def _create_table_summary(self, counts: dict) -> str:
        """Create professional table summary."""
        import plotly.graph_objects as go

        type_map = {
            'dim_': ('Dimension', F1Theme.DIMENSION),
            'fact_': ('Fact', F1Theme.FACT),
            'bridge_': ('Bridge', F1Theme.TEXT_SECONDARY),
            'f1_raw': ('JSON', F1Theme.JSON_COLOR),
            'f1_messages': ('Vector', F1Theme.VECTOR),
            'telemetry_spatial': ('Spatial', F1Theme.SPATIAL)
        }

        summary_data = []
        for table, count in counts.items():
            ttype, color = ('Other', F1Theme.TEXT_MUTED)
            for prefix, (typename, c) in type_map.items():
                if table.startswith(prefix) or table == prefix:
                    ttype, color = typename, c
                    break
            summary_data.append({'Table': table, 'Type': ttype, 'Rows': count, 'Color': color})

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Type')

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Table</b>', '<b>Type</b>', '<b>Row Count</b>'],
                fill_color=F1Theme.PAPER,
                font=dict(color=F1Theme.GOLD, size=14),
                align='left',
                line=dict(color=F1Theme.BORDER, width=1),
                height=35
            ),
            cells=dict(
                values=[df_summary['Table'], df_summary['Type'],
                       [f'{r:,}' for r in df_summary['Rows']]],
                fill_color=[[F1Theme.BACKGROUND] * len(df_summary)],
                font=dict(color=F1Theme.TEXT_PRIMARY, size=12),
                align='left',
                line=dict(color=F1Theme.BORDER, width=1),
                height=28
            )
        )])

        layout = F1Theme.plotly_layout(
            title='TABLE SUMMARY',
            subtitle='Row counts by data type',
            width=650,
            height=500
        )

        fig.update_layout(**layout)

        output_path = self.output_dir / "table_summary.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved table summary to {output_path}")

        return str(output_path)

    # =========================================================================
    # 5. JSON: Document Structure Visualization
    # =========================================================================

    def create_json_tree(self) -> str:
        """Create professional JSON document structure visualization."""
        import plotly.graph_objects as go

        logger.info("Creating JSON structure visualization...")

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT payload FROM f1_raw_documents
            WHERE doc_type = 'session_info'
            FETCH FIRST 1 ROW ONLY
        """)

        row = cursor.fetchone()
        cursor.close()

        if not row:
            logger.warning("No session_info document found")
            return None

        import json
        doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])

        def flatten_json(obj, parent='', sep='/'):
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent}{sep}{k}" if parent else k
                    if isinstance(v, (dict, list)):
                        items.append({'id': new_key, 'parent': parent, 'value': 1, 'label': k})
                        items.extend(flatten_json(v, new_key, sep))
                    else:
                        items.append({'id': new_key, 'parent': parent, 'value': 1,
                                     'label': f"{k}: {str(v)[:25]}"})
            elif isinstance(obj, list):
                for i, v in enumerate(obj[:5]):
                    new_key = f"{parent}[{i}]"
                    items.append({'id': new_key, 'parent': parent, 'value': 1, 'label': f'[{i}]'})
                    if isinstance(v, (dict, list)):
                        items.extend(flatten_json(v, new_key, sep))
            return items

        flat = flatten_json(doc)
        flat.insert(0, {'id': '', 'parent': '', 'value': 0, 'label': 'session_info'})

        df = pd.DataFrame(flat)

        # Custom colorscale matching theme
        fig = go.Figure(go.Treemap(
            ids=df['id'],
            labels=df['label'],
            parents=df['parent'],
            values=df['value'],
            textinfo='label',
            marker=dict(
                colors=[F1Theme.GOLD if '/' not in id else F1Theme.JSON_COLOR
                       for id in df['id']],
                line=dict(width=1, color=F1Theme.BACKGROUND)
            ),
            textfont=dict(color=F1Theme.BACKGROUND, size=11)
        ))

        layout = F1Theme.plotly_layout(
            title='JSON DOCUMENT STRUCTURE',
            subtitle='Session info hierarchical structure',
            width=1000,
            height=700
        )

        fig.update_layout(**layout)

        output_path = self.output_dir / "json_structure.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved JSON structure to {output_path}")

        return str(output_path)

    # =========================================================================
    # CREATE ALL
    # =========================================================================

    def create_all(self, circuit_name: str = "Singapore"):
        """Create all visualizations with consistent professional theme."""
        results = {}

        logger.info("=" * 60)
        logger.info("Creating all visualizations (F1 Professional Theme)...")
        logger.info("=" * 60)

        # 1. Circuit map (Spatial)
        try:
            results['circuit_map'] = self.create_circuit_map(circuit_name)
        except Exception as e:
            logger.error(f"Circuit map failed: {e}")

        # 2. Speed trace (Spatial)
        try:
            results['speed_trace'] = self.create_speed_trace_plot()
        except Exception as e:
            logger.error(f"Speed trace failed: {e}")

        # 3. Overtake network (Graph/Relational)
        try:
            results['overtake_network'] = self.create_overtake_network()
        except Exception as e:
            logger.error(f"Overtake network failed: {e}")

        # 4. Embeddings / messages (Vector)
        try:
            results['embeddings'] = self.create_embedding_visualization()
        except Exception as e:
            logger.error(f"Embeddings visualization failed: {e}")

        # 5. Data model (Relational)
        try:
            results['data_model'] = self.create_data_model_diagram()
        except Exception as e:
            logger.error(f"Data model failed: {e}")

        # 6. JSON structure
        try:
            results['json_structure'] = self.create_json_tree()
        except Exception as e:
            logger.error(f"JSON structure failed: {e}")

        logger.info("=" * 60)
        logger.info("Visualization Summary:")
        for name, path in results.items():
            status = "OK" if path else "FAILED"
            logger.info(f"  {name}: {status}")
            if path:
                logger.info(f"    -> {path}")
        logger.info("=" * 60)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create professional F1 data visualizations from Oracle"
    )
    parser.add_argument(
        "--circuit", type=str, default="Singapore",
        help="Circuit name for geo visualizations (default: Singapore)"
    )
    parser.add_argument(
        "--viz", type=str, nargs="+",
        choices=["circuit", "speed", "overtake", "embeddings", "semantic",
                 "timeline", "model", "summary", "json", "all"],
        default=["all"],
        help="Which visualizations to create (default: all)"
    )

    args = parser.parse_args()

    user = os.environ.get("ORA_USER")
    password = os.environ.get("ORA_PASSWORD")
    dsn = os.environ.get("ORA_DSN")

    if not all([user, password, dsn]):
        print("ERROR: Missing Oracle connection environment variables")
        print("Required: ORA_USER, ORA_PASSWORD, ORA_DSN")
        sys.exit(1)

    viz = F1Visualizer(user, password, dsn)

    try:
        viz.connect()

        if "all" in args.viz:
            viz.create_all(circuit_name=args.circuit)
        else:
            if "circuit" in args.viz:
                viz.create_circuit_map(args.circuit)
            if "speed" in args.viz:
                viz.create_speed_trace_plot()
            if "overtake" in args.viz:
                viz.create_overtake_network()
            if "embeddings" in args.viz or "semantic" in args.viz or "timeline" in args.viz:
                viz.create_embedding_visualization()
            if "model" in args.viz or "summary" in args.viz:
                viz.create_data_model_diagram()
            if "json" in args.viz:
                viz.create_json_tree()

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)
    finally:
        viz.close()


if __name__ == "__main__":
    main()
