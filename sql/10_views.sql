-- ============================================================================
-- Oracle Converged Database Demo - F1 Data Model
-- 10_views.sql - Helper views for common query patterns
-- ============================================================================

-- ============================================================================
-- DRIVER-EVENT DENORMALIZED VIEW
-- Combines driver, team, event, and result information
-- ============================================================================

CREATE OR REPLACE VIEW v_driver_event_results AS
SELECT
    e.event_id,
    e.session_id,
    e.season,
    e.gp_name,
    e.session_code,
    e.session_name,
    e.circuit_name,
    d.driver_id,
    d.driver_code,
    d.driver_number,
    d.full_name AS driver_name,
    t.team_id,
    t.team_name,
    t.team_color,
    r.grid_position,
    r.final_position,
    r.points,
    rs.status_name AS status,
    r.laps_completed
FROM dim_event e
JOIN fact_result r ON e.event_id = r.event_id
JOIN dim_driver d ON r.driver_id = d.driver_id
JOIN bridge_driver_team bdt ON e.event_id = bdt.event_id AND d.driver_id = bdt.driver_id
JOIN dim_team t ON bdt.team_id = t.team_id
LEFT JOIN dim_result_status rs ON r.status_id = rs.status_id;

-- ============================================================================
-- LAP ANALYSIS VIEW
-- Includes driver/team context with lap data
-- ============================================================================

CREATE OR REPLACE VIEW v_lap_analysis AS
SELECT
    e.event_id,
    e.session_id,
    e.gp_name,
    d.driver_id,
    d.driver_code,
    d.full_name AS driver_name,
    t.team_name,
    l.lap_number,
    l.lap_time_sec,
    l.sector1_sec,
    l.sector2_sec,
    l.sector3_sec,
    l.stint_number,
    c.compound_name AS compound,
    l.tyre_life,
    l.position,
    l.is_personal_best,
    l.is_accurate,
    l.speed_i1,
    l.speed_i2,
    l.speed_fl
FROM fact_lap l
JOIN dim_event e ON l.event_id = e.event_id
JOIN dim_driver d ON l.driver_id = d.driver_id
JOIN bridge_driver_team bdt ON e.event_id = bdt.event_id AND d.driver_id = bdt.driver_id
JOIN dim_team t ON bdt.team_id = t.team_id
LEFT JOIN dim_compound c ON l.compound_id = c.compound_id;

-- ============================================================================
-- STINT SUMMARY VIEW
-- Aggregated stint-level metrics
-- ============================================================================

CREATE OR REPLACE VIEW v_stint_summary AS
SELECT
    l.event_id,
    l.driver_id,
    l.stint_number,
    c.compound_name AS compound,
    MIN(l.lap_number) AS stint_start_lap,
    MAX(l.lap_number) AS stint_end_lap,
    COUNT(*) AS lap_count,
    ROUND(AVG(l.lap_time_sec), 3) AS avg_lap_time,
    ROUND(MIN(l.lap_time_sec), 3) AS best_lap_time,
    ROUND(MAX(l.lap_time_sec), 3) AS worst_lap_time,
    ROUND(STDDEV(l.lap_time_sec), 3) AS lap_time_stddev,
    MAX(l.tyre_life) AS max_tyre_age
FROM fact_lap l
LEFT JOIN dim_compound c ON l.compound_id = c.compound_id
WHERE l.lap_time_sec IS NOT NULL
  AND l.is_accurate = 1
GROUP BY l.event_id, l.driver_id, l.stint_number, c.compound_name;

-- ============================================================================
-- TELEMETRY SUMMARY VIEW (sampled for performance)
-- Provides aggregated telemetry per driver per second
-- ============================================================================

CREATE OR REPLACE VIEW v_telemetry_1sec AS
SELECT
    event_id,
    driver_id,
    TRUNC(session_time_sec) AS time_bucket_sec,
    ROUND(AVG(speed), 1) AS avg_speed,
    ROUND(MAX(speed), 1) AS max_speed,
    ROUND(AVG(rpm), 0) AS avg_rpm,
    ROUND(AVG(throttle), 1) AS avg_throttle,
    SUM(brake) AS brake_samples,
    COUNT(*) AS sample_count
FROM fact_telemetry
GROUP BY event_id, driver_id, TRUNC(session_time_sec);

-- ============================================================================
-- RACE CONTROL MESSAGES (JSON extraction view)
-- Extracts structured data from JSON landing table
-- ============================================================================

CREATE OR REPLACE VIEW v_race_control_messages AS
SELECT
    d.doc_id,
    d.event_id,
    e.gp_name,
    JSON_VALUE(d.payload, '$.Time') AS msg_time,
    JSON_VALUE(d.payload, '$.Category') AS category,
    JSON_VALUE(d.payload, '$.Flag') AS flag,
    JSON_VALUE(d.payload, '$.Scope') AS scope,
    JSON_VALUE(d.payload, '$.Message') AS message_text,
    JSON_VALUE(d.payload, '$.RacingNumber') AS racing_number,
    JSON_VALUE(d.payload, '$.Lap' RETURNING NUMBER) AS lap
FROM f1_raw_documents d
JOIN dim_event e ON d.event_id = e.event_id
WHERE d.doc_type = 'race_control_message';

-- ============================================================================
-- TEAMMATES VIEW
-- Shows driver pairs within the same team for an event
-- ============================================================================

CREATE OR REPLACE VIEW v_teammates AS
SELECT
    e.event_id,
    e.session_id,
    e.gp_name,
    t.team_name,
    d1.driver_code AS driver1_code,
    d1.full_name AS driver1_name,
    d2.driver_code AS driver2_code,
    d2.full_name AS driver2_name
FROM bridge_driver_team bdt1
JOIN bridge_driver_team bdt2
    ON bdt1.event_id = bdt2.event_id
    AND bdt1.team_id = bdt2.team_id
    AND bdt1.driver_id < bdt2.driver_id
JOIN dim_event e ON bdt1.event_id = e.event_id
JOIN dim_team t ON bdt1.team_id = t.team_id
JOIN dim_driver d1 ON bdt1.driver_id = d1.driver_id
JOIN dim_driver d2 ON bdt2.driver_id = d2.driver_id;

-- ============================================================================
-- WEATHER TIMELINE VIEW
-- Joins weather with session status for context
-- ============================================================================

CREATE OR REPLACE VIEW v_weather_timeline AS
SELECT
    w.event_id,
    e.gp_name,
    w.time_sec,
    w.air_temp,
    w.track_temp,
    w.humidity,
    w.wind_speed,
    w.rainfall,
    (SELECT MAX(ss.status) KEEP (DENSE_RANK FIRST ORDER BY ss.time_sec DESC)
     FROM fact_session_status ss
     WHERE ss.event_id = w.event_id AND ss.time_sec <= w.time_sec) AS session_status
FROM fact_weather w
JOIN dim_event e ON w.event_id = e.event_id;

COMMIT;
