## Postgres setup prompts (tables + population)

This file is a **copy/paste checklist** of the commands used to set up the Postgres tables
expected by the simulator and the incident generator.

### 0) Set connection environment (optional)

The Python code reads these variables:

- `COMP594_DB_NAME` (default `comp594`)
- `COMP594_DB_USER` (default `postgres`)
- `COMP594_DB_PASS` (default `postgres`)
- `COMP594_DB_HOST` (default `localhost`)
- `COMP594_DB_PORT` (default `5432`)

put in an `.env` file

### 1) Create the database

```bash
createdb comp594
```

Or with psql:

```bash
psql -d postgres -c "CREATE DATABASE comp594;"
```

### 2) Create the core location tables

Run these once (or paste into `psql -d comp594`).

```sql
CREATE TABLE IF NOT EXISTS base_stations (
    id TEXT PRIMARY KEY,
    station_number TEXT,
    name TEXT,
    agency TEXT,
    address TEXT,
    phone TEXT,
    phones TEXT,
    coordinates TEXT,
    capabilities TEXT,
    units TEXT,
    number_of_units INTEGER
);

CREATE TABLE IF NOT EXISTS hospitals (
    id TEXT PRIMARY KEY,
    name TEXT,
    facility_code TEXT,
    address TEXT,
    phone TEXT,
    agency TEXT,
    coordinates TEXT,
    units TEXT,
    number_of_units INTEGER
);

-- Note: `base_incident_travel_matrix` references `incident_locations(incident_id)`.
CREATE TABLE IF NOT EXISTS incident_locations (
    incident_id TEXT PRIMARY KEY,
    name TEXT,
    municipality TEXT,
    county TEXT,
    state TEXT,
    location_type TEXT,
    coordinates TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

-- Optional (the simulator can run without it).
CREATE TABLE IF NOT EXISTS popular_incident_locations (
    id TEXT PRIMARY KEY,
    name TEXT,
    municipality TEXT,
    county TEXT,
    state TEXT,
    location_type TEXT,
    coordinates TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);
```

### 3) Create the travel matrix tables

#### 3a) base → base

```sql
CREATE TABLE IF NOT EXISTS base_base_travel_matrix (
    origin_base_id TEXT NOT NULL REFERENCES base_stations(id) ON DELETE CASCADE,
    destination_base_id TEXT NOT NULL REFERENCES base_stations(id) ON DELETE CASCADE,
    distance_miles DOUBLE PRECISION NOT NULL,
    travel_time_minutes DOUBLE PRECISION NOT NULL,
    provider TEXT NOT NULL,
    profile TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (origin_base_id, destination_base_id)
);
```

#### 3b) base → hospital

```sql
CREATE TABLE IF NOT EXISTS base_hospital_travel_matrix (
    origin_base_id TEXT NOT NULL REFERENCES base_stations(id) ON DELETE CASCADE,
    destination_hospital_id TEXT NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    distance_miles DOUBLE PRECISION NOT NULL,
    travel_time_minutes DOUBLE PRECISION NOT NULL,
    provider TEXT NOT NULL,
    profile TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (origin_base_id, destination_hospital_id)
);
```

#### 3c) base → incident

Use the provided SQL file:

```bash
psql -d comp594 -f util/simulator/database/sql/create_base_incident_travel_matrix.sql
```

### 4) Populate base stations / hospitals / incident locations

How you populate these tables depends on your source data. Two common options:

#### Option A: `\copy` from CSV

```bash
# Example only — replace paths + columns to match your CSV.
psql -d comp594 -c "\\copy base_stations(id,station_number,name,agency,address,phone,phones,coordinates,capabilities,units,number_of_units) FROM 'PATH/TO/base_stations.csv' WITH (FORMAT csv, HEADER true)"
psql -d comp594 -c "\\copy hospitals(id,name,facility_code,address,phone,agency,coordinates,units,number_of_units) FROM 'PATH/TO/hospitals.csv' WITH (FORMAT csv, HEADER true)"
psql -d comp594 -c "\\copy incident_locations(incident_id,name,municipality,county,state,location_type,coordinates,latitude,longitude) FROM 'PATH/TO/incident_locations.csv' WITH (FORMAT csv, HEADER true)"
```

#### Option B: Upsert via Python (manual)

The repo also has upsert helpers:

- `util/simulator/database/base_stations_service.py`
- `util/simulator/database/hospitals_service.py`

You can write a small Python script to read your source data and call those upsert functions.

### 5) Populate the travel matrices (OpenRouteService)

This step calls the OpenRouteService Matrix API and stores results in Postgres.

Set an API key (either env var works):

```bash
setx OPENROUTESERVICE_API_KEY "YOUR_KEY_HERE"
```

Then run:

```bash
python -m util.simulator.scripts.update_distance_matrix_tables --verbose
```

Notes:

- This script expects the tables in steps 2–3 to already exist.
- It reads coordinates from `base_stations`, `hospitals`, and `incident_locations`.
- It writes into `base_base_travel_matrix`, `base_hospital_travel_matrix`, `base_incident_travel_matrix`.
