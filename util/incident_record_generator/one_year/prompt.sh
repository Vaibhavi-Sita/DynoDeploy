python -m util.incident_record_generator \
  --days 365 \
  --incidents 43000 \
  --location-sample-size 18000 \
  --urban-rural-split urban:0.78,rural:0.22 \
  --peak-hour 13:1.9 \
  --peak-weekday 1:1.05 --peak-weekday 2:1.05 --peak-weekday 3:1.05 \
  --overlap-probability 0.30 \
  --overlap-same-location 0.45 \
  --overlap-window 18 60 \
  --max-incidents-per-location 11000 \
  --output resources/simulated_records/sim_year_baseline.json \
  --verbose
python -m util.incident_record_generator --days 365 --incidents 43000 --location-sample-size 18000 --urban-rural-split urban:0.78,rural:0.22 --peak-hour 13:1.9 --peak-weekday 1:1.05 --peak-weekday 2:1.05 --peak-weekday 3:1.05 --overlap-probability 0.30 --overlap-same-location 0.45 --overlap-window 18 60 --max-incidents-per-location 11000 --output resources/simulated_records/sim_year_baseline.json --verbose

python -m util.incident_record_generator \
  --days 365 \
  --incidents 52000 \
  --location-sample-size 21000 \
  --urban-rural-split urban:0.86,rural:0.14 \
  --peak-hour 10:1.3 --peak-hour 13:2.2 --peak-hour 17:1.7 \
  --peak-weekday 1:1.15 --peak-weekday 2:1.15 --peak-weekday 3:1.1 --peak-weekday 4:1.1 \
  --overlap-probability 0.42 \
  --overlap-same-location 0.65 \
  --overlap-window 12 45 \
  --max-incidents-per-location 14000 \
  --output resources/simulated_records/sim_year_dynamic.json \
  --verbose
python -m util.incident_record_generator --days 365 --incidents 52000 --location-sample-size 21000 --urban-rural-split urban:0.86,rural:0.14 --peak-hour 10:1.3 --peak-hour 13:2.2 --peak-hour 17:1.7 --peak-weekday 1:1.15 --peak-weekday 2:1.15 --peak-weekday 3:1.1 --peak-weekday 4:1.1 --overlap-probability 0.42 --overlap-same-location 0.65 --overlap-window 12 45 --max-incidents-per-location 14000 --output resources/simulated_records/sim_year_dynamic.json --verbose

python -m util.incident_record_generator \
  --days 365 \
  --incidents 65000 \
  --location-sample-size 24000 \
  --urban-rural-split urban:0.92,rural:0.08 \
  --peak-hour 7:1.0 --peak-hour 11:1.5 --peak-hour 14:2.5 --peak-hour 18:2.0 --peak-hour 22:1.4 \
  --peak-weekday 1:1.25 --peak-weekday 2:1.25 --peak-weekday 3:1.2 --peak-weekday 4:1.2 --peak-weekday 5:1.15 \
  --overlap-probability 0.58 \
  --overlap-same-location 0.80 \
  --overlap-window 10 40 \
  --max-incidents-per-location 17500 \
  --output resources/simulated_records/sim_year_hotspots.json \
  --verbose
python -m util.incident_record_generator --days 365 --incidents 65000 --location-sample-size 24000 --urban-rural-split urban:0.92,rural:0.08 --peak-hour 7:1.0 --peak-hour 11:1.5 --peak-hour 14:2.5 --peak-hour 18:2.0 --peak-hour 22:1.4 --peak-weekday 1:1.25 --peak-weekday 2:1.25 --peak-weekday 3:1.2 --peak-weekday 4:1.2 --peak-weekday 5:1.15 --overlap-probability 0.58 --overlap-same-location 0.80 --overlap-window 10 40 --max-incidents-per-location 17500 --output resources/simulated_records/sim_year_hotspots.json --verbose