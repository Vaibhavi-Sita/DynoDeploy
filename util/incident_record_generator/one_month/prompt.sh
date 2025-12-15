python -m util.incident_record_generator \
  --days 30 \
  --incidents 3600 \
  --location-sample-size 2100 \
  --urban-rural-split urban:0.82,rural:0.18 \
  --peak-hour 13:2.0 \
  --peak-weekday 1:1.1 --peak-weekday 2:1.1 --peak-weekday 3:1.1 \
  --overlap-probability 0.35 \
  --overlap-same-location 0.55 \
  --overlap-window 12 42 \
  --max-incidents-per-location 1200 \
  --output resources/simulated_records/sim_month_baseline.json \
  --verbose
python -m util.incident_record_generator --days 30 --incidents 3600 --location-sample-size 2100 --urban-rural-split urban:0.82,rural:0.18 --peak-hour 13:2.0 --peak-weekday 1:1.1 --peak-weekday 2:1.1 --peak-weekday 3:1.1 --overlap-probability 0.35 --overlap-same-location 0.55 --overlap-window 12 42 --max-incidents-per-location 1200 --output resources/simulated_records/sim_month_baseline.json --verbose

python -m util.incident_record_generator \
  --days 30 \
  --incidents 5400 \
  --location-sample-size 2500 \
  --urban-rural-split urban:0.90,rural:0.10 \
  --peak-hour 9:1.4 --peak-hour 13:2.4 --peak-hour 17:2.0 \
  --peak-weekday 1:1.3 --peak-weekday 2:1.3 --peak-weekday 3:1.2 --peak-weekday 4:1.2 \
  --overlap-probability 0.55 \
  --overlap-same-location 0.78 \
  --overlap-window 6 30 \
  --max-incidents-per-location 1500 \
  --output resources/simulated_records/sim_month_peak_multi.json \
  --verbose
python -m util.incident_record_generator --days 30 --incidents 5400 --location-sample-size 2500 --urban-rural-split urban:0.90,rural:0.10 --peak-hour 9:1.4 --peak-hour 13:2.4 --peak-hour 17:2.0 --peak-weekday 1:1.3 --peak-weekday 2:1.3 --peak-weekday 3:1.2 --peak-weekday 4:1.2 --overlap-probability 0.55 --overlap-same-location 0.78 --overlap-window 6 30 --max-incidents-per-location 1500 --output resources/simulated_records/sim_month_peak_multi.json --verbose

python -m util.incident_record_generator \
  --days 30 \
  --incidents 7200 \
  --location-sample-size 2800 \
  --urban-rural-split urban:0.93,rural:0.07 \
  --peak-hour 7:1.2 --peak-hour 11:1.6 --peak-hour 14:2.6 --peak-hour 18:2.1 --peak-hour 22:1.4 \
  --peak-weekday 1:1.35 --peak-weekday 2:1.35 --peak-weekday 3:1.3 --peak-weekday 4:1.25 --peak-weekday 5:1.15 \
  --overlap-probability 0.65 \
  --overlap-same-location 0.85 \
  --overlap-window 5 28 \
  --max-incidents-per-location 1800 \
  --output resources/simulated_records/sim_month_hot_bed.json \
  --verbose
python -m util.incident_record_generator --days 30 --incidents 7200 --location-sample-size 2800 --urban-rural-split urban:0.93,rural:0.07 --peak-hour 7:1.2 --peak-hour 11:1.6 --peak-hour 14:2.6 --peak-hour 18:2.1 --peak-hour 22:1.4 --peak-weekday 1:1.35 --peak-weekday 2:1.35 --peak-weekday 3:1.3 --peak-weekday 4:1.25 --peak-weekday 5:1.15 --overlap-probability 0.65 --overlap-same-location 0.85 --overlap-window 5 28 --max-incidents-per-location 1800 --output resources/simulated_records/sim_month_hot_bed.json --verbose