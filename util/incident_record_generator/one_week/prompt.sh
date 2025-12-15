python -m util.incident_record_generator \
  --days 7 \
  --incidents 900 \
  --location-sample-size 620 \
  --urban-rural-split urban:0.80,rural:0.20 \
  --peak-hour 13:1.9 \
  --overlap-probability 0.32 \
  --overlap-same-location 0.55 \
  --overlap-window 12 36 \
  --max-incidents-per-location 360 \
  --output resources/simulated_records/sim_week_baseline.json \
  --verbose
python -m util.incident_record_generator --days 7 --incidents 900 --location-sample-size 620 --urban-rural-split urban:0.80,rural:0.20 --peak-hour 13:1.9 --overlap-probability 0.32 --overlap-same-location 0.55 --overlap-window 12 36 --max-incidents-per-location 360 --output resources/simulated_records/sim_week_baseline.json --verbose

python -m util.incident_record_generator \
  --days 7 \
  --incidents 1260 \
  --location-sample-size 720 \
  --urban-rural-split urban:0.88,rural:0.12 \
  --peak-hour 12:1.7 --peak-hour 13:2.2 --peak-hour 17:1.6 \
  --peak-weekday 1:1.2 --peak-weekday 2:1.2 --peak-weekday 3:1.2 --peak-weekday 4:1.1 \
  --overlap-probability 0.40 \
  --overlap-same-location 0.70 \
  --overlap-window 8 24 \
  --max-incidents-per-location 420 \
  --output resources/simulated_records/sim_week_peak1pm.json \
  --verbose
python -m util.incident_record_generator --days 7 --incidents 1260 --location-sample-size 720 --urban-rural-split urban:0.88,rural:0.12 --peak-hour 12:1.7 --peak-hour 13:2.2 --peak-hour 17:1.6 --peak-weekday 1:1.2 --peak-weekday 2:1.2 --peak-weekday 3:1.2 --peak-weekday 4:1.1 --overlap-probability 0.40 --overlap-same-location 0.70 --overlap-window 8 24 --max-incidents-per-location 420 --output resources/simulated_records/sim_week_peak1pm.json --verbose

python -m util.incident_record_generator \
  --days 7 \
  --incidents 1800 \
  --location-sample-size 860 \
  --urban-rural-split urban:0.92,rural:0.08 \
  --peak-hour 9:1.3 --peak-hour 12:1.6 --peak-hour 14:2.4 --peak-hour 18:1.8 \
  --peak-weekday 1:1.3 --peak-weekday 2:1.3 --peak-weekday 3:1.25 --peak-weekday 4:1.25 --peak-weekday 5:1.1 \
  --overlap-probability 0.55 \
  --overlap-same-location 0.80 \
  --overlap-window 6 32 \
  --max-incidents-per-location 520 \
  --output resources/simulated_records/sim_week_hotspots.json \
  --verbose
python -m util.incident_record_generator --days 7 --incidents 1800 --location-sample-size 860 --urban-rural-split urban:0.92,rural:0.08 --peak-hour 9:1.3 --peak-hour 12:1.6 --peak-hour 14:2.4 --peak-hour 18:1.8 --peak-weekday 1:1.3 --peak-weekday 2:1.3 --peak-weekday 3:1.25 --peak-weekday 4:1.25 --peak-weekday 5:1.1 --overlap-probability 0.55 --overlap-same-location 0.80 --overlap-window 6 32 --max-incidents-per-location 520 --output resources/simulated_records/sim_week.json --verbose