python -m util.incident_record_generator \
  --days 90 \
  --incidents 10800 \
  --location-sample-size 6000 \
  --urban-rural-split urban:0.80,rural:0.20 \
  --peak-hour 13:1.9 \
  --peak-weekday 1:1.1 --peak-weekday 2:1.1 --peak-weekday 3:1.1 \
  --overlap-probability 0.32 \
  --overlap-same-location 0.50 \
  --overlap-window 18 48 \
  --max-incidents-per-location 3200 \
  --output resources/simulated_records/sim_quarter_baseline.json \
  --verbose
python -m util.incident_record_generator --days 90 --incidents 10800 --location-sample-size 6000 --urban-rural-split urban:0.80,rural:0.20 --peak-hour 13:1.9 --peak-weekday 1:1.1 --peak-weekday 2:1.1 --peak-weekday 3:1.1 --overlap-probability 0.32 --overlap-same-location 0.50 --overlap-window 18 48 --max-incidents-per-location 3200 --output resources/simulated_records/sim_quarter_baseline.json --verbose

python -m util.incident_record_generator \
  --days 90 \
  --incidents 13500 \
  --location-sample-size 6800 \
  --urban-rural-split urban:0.88,rural:0.12 \
  --peak-hour 10:1.4 --peak-hour 13:2.3 --peak-hour 17:1.9 \
  --peak-weekday 1:1.25 --peak-weekday 2:1.25 --peak-weekday 3:1.2 --peak-weekday 4:1.15 \
  --overlap-probability 0.48 \
  --overlap-same-location 0.70 \
  --overlap-window 10 34 \
  --max-incidents-per-location 4200 \
  --output resources/simulated_records/sim_quarter_dynamic.json \
  --verbose
python -m util.incident_record_generator --days 90 --incidents 13500 --location-sample-size 6800 --urban-rural-split urban:0.88,rural:0.12 --peak-hour 10:1.4 --peak-hour 13:2.3 --peak-hour 17:1.9 --peak-weekday 1:1.25 --peak-weekday 2:1.25 --peak-weekday 3:1.2 --peak-weekday 4:1.15 --overlap-probability 0.48 --overlap-same-location 0.70 --overlap-window 10 34 --max-incidents-per-location 4200 --output resources/simulated_records/sim_quarter_dynamic.json --verbose

python -m util.incident_record_generator \
  --days 90 \
  --incidents 16200 \
  --location-sample-size 7600 \
  --urban-rural-split urban:0.93,rural:0.07 \
  --peak-hour 7:1.1 --peak-hour 11:1.5 --peak-hour 14:2.5 --peak-hour 18:2.1 --peak-hour 22:1.5 \
  --peak-weekday 1:1.35 --peak-weekday 2:1.35 --peak-weekday 3:1.3 --peak-weekday 4:1.25 --peak-weekday 5:1.2 \
  --overlap-probability 0.62 \
  --overlap-same-location 0.82 \
  --overlap-window 8 30 \
  --max-incidents-per-location 5200 \
  --output resources/simulated_records/sim_quarter_hotspots.json \
  --verbose
python -m util.incident_record_generator --days 90 --incidents 16200 --location-sample-size 7600 --urban-rural-split urban:0.93,rural:0.07 --peak-hour 7:1.1 --peak-hour 11:1.5 --peak-hour 14:2.5 --peak-hour 18:2.1 --peak-hour 22:1.5 --peak-weekday 1:1.35 --peak-weekday 2:1.35 --peak-weekday 3:1.3 --peak-weekday 4:1.25 --peak-weekday 5:1.2 --overlap-probability 0.62 --overlap-same-location 0.82 --overlap-window 8 30 --max-incidents-per-location 5200 --output resources/simulated_records/sim_quarter_hotspots.json --verbose