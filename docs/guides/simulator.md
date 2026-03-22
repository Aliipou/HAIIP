# Machine Simulator

The built-in simulator generates realistic industrial sensor data for demos and testing without real hardware.

## Start the Simulator

```bash
# Simulate a rotating motor
python -m haiip.simulator motor \
    --speed 1500  # RPM \
    --load 0.75   # 75% load \
    --inject-fault bearing_wear --fault-time 600  # fault after 10 min

# Output to MQTT
python -m haiip.simulator motor --output mqtt://localhost:1883/sensors/motor-01
```

## Fault Injection

```bash
# Available faults
python -m haiip.simulator list-faults

# bearing_wear      — gradual vibration increase
# imbalance         — periodic vibration spike
# overheating       — temperature drift
# cavitation        — pump pressure oscillation
# electrical_fault  — motor current spike
```

## Scenario Files

```yaml
# scenarios/press-failure.yml
machine: hydraulic_press
duration_seconds: 3600
sensors:
  - name: pressure
    base_value: 250    # bar
    noise: 0.5
  - name: temperature
    base_value: 45     # C
    noise: 0.3
faults:
  - type: hydraulic_leak
    start_seconds: 1800
    severity: gradual
```

```bash
python -m haiip.simulator scenario scenarios/press-failure.yml
```
