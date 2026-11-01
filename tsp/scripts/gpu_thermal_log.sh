#!/usr/bin/env bash
# Samples the GPU temperature, clock and thermal slowdown every 30 s: a hot T4 drops to 300 MHz, which changes every
# latency measured while it lasts. Runs until the file results/phase5/open_language/thermal.stop exists.
OUT=results/phase5/open_language
while [ ! -e $OUT/thermal.stop ]; do
    echo "$(date +%T), $(nvidia-smi --query-gpu=temperature.gpu,clocks.gr,power.draw,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown --format=csv,noheader,nounits)" >> $OUT/thermal.csv
    sleep 30
done
