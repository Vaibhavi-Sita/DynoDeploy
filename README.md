## Ambulance redeployment simulator

This folder contains the simulator + helper scripts used for COMP-594.

If you want to run simulations, follow the steps below from the folder that contains:

- `util/`
- `resources/`

### What’s in here (quick map)

- **Simulator**: `util/simulator/`
- **Rule definitions**: `util/simulator/rules.yaml`
- **Rule reference table (human-readable)**: `resources/original_incident_records/COMP 594.csv`
- **DB helpers**: `util/simulator/database/`
- **DB maintenance scripts**: `util/simulator/scripts/`
- **Incident record generator**: `util/incident_record_generator/`
- **Prompts (commands/params)**:
  - `resources/simulated_records/*/prompt.sh`
  - `util/incident_record_generator/one_*/prompt.sh`

### Rules (what is Rule_A, Rule_B, …?)

The simulator uses rules defined in:

- `util/simulator/rules.yaml`

If you want a simple table that explains what each rule means (what toggles are on/off),
see:

- `resources/original_incident_records/COMP 594.csv`

This CSV is **for reference/documentation**. You do not need to open it to run the simulator.

### Step 1 — Install Python packages

Open a terminal in the folder (the one that contains `util/` and `resources/`).

On Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Connect to the database

The simulator reads database settings from environment variables, or from a `.env` file.

- `COMP594_DB_NAME` (default `comp594`)
- `COMP594_DB_USER` (default `postgres`)
- `COMP594_DB_PASS` (default `postgres`)
- `COMP594_DB_HOST` (default `localhost`)
- `COMP594_DB_PORT` (default `5432`)

Create a `.env` file in the folder (recommended):

```bash
COMP594_DB_NAME=comp594
COMP594_DB_USER=postgres
COMP594_DB_PASS=postgres
COMP594_DB_HOST=localhost
COMP594_DB_PORT=5432
```

### Step 3 — Make sure the tables exist

You need these Postgres tables:

- `base_stations`
- `hospitals`
- `incident_locations` (and optionally `popular_incident_locations`)
- Travel matrices:
  - `base_base_travel_matrix`
  - `base_hospital_travel_matrix`
  - `base_incident_travel_matrix`

If your DB is missing travel matrices, look at:

- `util/simulator/scripts/update_distance_matrix_tables.py`
- `util/simulator/database/sql/create_base_incident_travel_matrix.sql`

### Step 4 — Generate incident-history JSON (optional)

The simulator needs incident-history JSON files (examples: `day.json`, `week.json`).
If you don’t have them, generate them using the included prompt scripts.

Example:

```bash
bash resources/simulated_records/baseline/prompt.sh
```

Or generate a smaller one-off file:

```bash
bash util/incident_record_generator/one_day/prompt.sh
```

### Step 5 — Run the simulator

Pick one of these options.

#### Option A: Notebook

Open `util/simulator/simulation_runner.ipynb` and run the cells.

The notebook expects incident files under `resources/simulated_records/...`.

#### Option B: CLI runner

Run it from the root:

```bash
python -m util.simulator.scripts.run_rule_matrix --help
```

#### Option C: Python API

```python
from pathlib import Path

from util.simulator.simulator import SimulationRunner

runner = SimulationRunner(prefer_database=True)
result = runner.run(
    rule_id="Rule_A",
    template_name="day",
    history_paths=[Path("resources/simulated_records/baseline/day.json")],
)

print(result.metrics)
```

### What’s not included

- Test suites
- Large outputs (charts/CSVs) produced by exploratory notebooks
