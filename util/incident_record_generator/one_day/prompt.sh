python -m util.incident_record_generator \
  --days 1 \
  --incidents 180 \
  --location-sample-size 120 \
  --urban-rural-split urban:0.78,rural:0.22 \
  --peak-hour 13:2.0 \
  --overlap-probability 0.30 \
  --overlap-same-location 0.55 \
  --overlap-window 10 22 \
  --max-incidents-per-location 80 \
  --output resources/simulated_records/sim_day_peak1pm.json \
  --verbose
python -m util.incident_record_generator --days 1 --incidents 180 --location-sample-size 120 --urban-rural-split urban:0.78,rural:0.22 --peak-hour 13:2.0 --overlap-probability 0.30 --overlap-same-location 0.55 --overlap-window 10 22 --max-incidents-per-location 80 --output resources/simulated_records/sim_day_peak1pm.json --verbose

python -m util.incident_record_generator \
  --days 1 \
  --incidents 260 \
  --location-sample-size 180 \
  --urban-rural-split urban:0.84,rural:0.16 \
  --peak-hour 11:1.5 --peak-hour 13:2.3 --peak-hour 17:1.6 \
  --overlap-probability 0.45 \
  --overlap-same-location 0.70 \
  --overlap-window 6 28 \
  --max-incidents-per-location 110 \
  --output resources/simulated_records/sim_day_peak_multi.json \
  --verbose
python -m util.incident_record_generator --days 1 --incidents 260 --location-sample-size 180 --urban-rural-split urban:0.84,rural:0.16 --peak-hour 11:1.5 --peak-hour 13:2.3 --peak-hour 17:1.6 --overlap-probability 0.45 --overlap-same-location 0.70 --overlap-window 6 28 --max-incidents-per-location 110 --output resources/simulated_records/sim_day_peak_multi.json --verbose


python -m util.incident_record_generator --days 1 --incidents 360 --location-sample-size 210 --urban-rural-split urban:0.90,rural:0.10 --peak-hour 9:1.4 --peak-hour 13:2.6 --peak-hour 17:2.0 --peak-hour 22:1.3 --overlap-probability 0.60 --overlap-same-location 0.80 --overlap-window 4 32 --max-incidents-per-location 150 --output resources/simulated_records/sim_day.json --verbose