-- Create the base→incident travel matrix table for OpenRouteService outputs.
-- Run with: psql -d <database> -f create_base_incident_travel_matrix.sql

CREATE TABLE IF NOT EXISTS base_incident_travel_matrix (
    origin_base_id TEXT NOT NULL REFERENCES base_stations(id) ON DELETE CASCADE,
    destination_incident_id TEXT NOT NULL REFERENCES incident_locations(incident_id) ON DELETE CASCADE,
    distance_miles DOUBLE PRECISION NOT NULL,
    travel_time_minutes DOUBLE PRECISION NOT NULL,
    provider TEXT NOT NULL,
    profile TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (origin_base_id, destination_incident_id)
);

CREATE INDEX IF NOT EXISTS idx_base_incident_origin
    ON base_incident_travel_matrix (origin_base_id);

CREATE INDEX IF NOT EXISTS idx_base_incident_destination
    ON base_incident_travel_matrix (destination_incident_id);


