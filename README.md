# NT Summary Stats

Fast computation of traditional summary statistics for neutrino telescopes.

## Install

```bash
pip install nt_summary_stats
```

## Usage

```python
import numpy as np
from nt_summary_stats import compute_summary_stats

# Basic usage
times = np.array([10.0, 15.0, 25.0, 100.0])      # shape: (N,), dtype: float
charges = np.array([1.0, 2.0, 1.5, 0.5])         # shape: (N,), dtype: float
stats = compute_summary_stats(times, charges)     # returns: np.ndarray, shape (9,)

print(stats[0])  # total_charge: 5.0
print(stats[3])  # first_pulse_time: 10.0
print(stats[7])  # charge_weighted_mean_time: 26.0
```

Process [Prometheus](https://github.com/Harvard-Neutrino/prometheus) events:

```python
from nt_summary_stats import process_prometheus_event

# Input: Prometheus event dictionary
event_data = {
    'photons': {
        'sensor_pos_x': [0.0, 0.0, 100.0],  # list[float], length M
        'sensor_pos_y': [0.0, 0.0, 0.0],    # list[float], length M
        'sensor_pos_z': [0.0, 0.0, 50.0],   # list[float], length M
        'string_id': [1, 1, 2],             # list[int], length M
        'sensor_id': [1, 1, 1],             # list[int], length M
        't': [10.0, 15.0, 20.0]             # list[float], length M
    }
}

# Default: no grouping (uses all hits as-is)
sensor_positions, sensor_stats = process_prometheus_event(event_data)

# Optional: group hits within time windows
sensor_positions, sensor_stats = process_prometheus_event(event_data, grouping_window_ns=2.0)
# sensor_positions: np.ndarray, shape (N_sensors, 3), dtype: float64
# sensor_stats: np.ndarray, shape (N_sensors, 9), dtype: float64
# Arrays are aligned: sensor_positions[i] corresponds to sensor_stats[i]
```

Process individual sensor data:

```python
from nt_summary_stats import process_sensor_data

# Input: sensor hit data
sensor_times = [10.0, 10.5, 15.0, 100.0]    # list[float] or np.ndarray(N,)
sensor_charges = [1.0, 0.5, 2.0, 1.0]       # list[float] or np.ndarray(N,), optional

# Default: no grouping (uses all hits as-is)
stats = process_sensor_data(sensor_times, sensor_charges)  # returns: np.ndarray, shape (9,)

# Optional: group hits within time windows
stats = process_sensor_data(sensor_times, sensor_charges, grouping_window_ns=2.0)
```

## Summary Statistics

Computes 9 traditional summary statistics for neutrino telescope sensors as described in the [IceCube paper](https://arxiv.org/abs/2101.11589). All functions return numpy arrays with statistics in the following order:

```python
stats = compute_summary_stats(times, charges)  # shape: (9,)

# Array indices:
stats[0]  # total_charge: Total charge collected
stats[1]  # charge_100ns: Charge within 100ns of first pulse
stats[2]  # charge_500ns: Charge within 500ns of first pulse
stats[3]  # first_pulse_time: Time of first pulse
stats[4]  # last_pulse_time: Time of last pulse
stats[5]  # charge_20_percent_time: Time at which 20% of charge is collected
stats[6]  # charge_50_percent_time: Time at which 50% of charge is collected
stats[7]  # charge_weighted_mean_time: Charge-weighted mean time
stats[8]  # charge_weighted_std_time: Charge-weighted standard deviation
```

## API

### `compute_summary_stats(times, charges)`

**Args:**
- `times`: `np.ndarray` or `list`, shape `(N,)` - pulse arrival times in ns
- `charges`: `np.ndarray` or `list`, shape `(N,)` - pulse charges

**Returns:** `np.ndarray`, shape `(9,)` - array with 9 summary statistics in order shown above

### `process_prometheus_event(event_data, grouping_window_ns=None)`

**Args:**
- `event_data`: `dict` or `awkward.Array` - Prometheus event data. Supports:
  - Dictionary with `photons` key containing sensor data
  - Awkward array with `photons` field (requires `awkward` package)
  - Direct photon data structure with required fields
- `grouping_window_ns`: `float` or `None` - time window for grouping hits (default: None, no grouping)

**Returns:** `tuple[np.ndarray, np.ndarray]`
- `sensor_positions`: `np.ndarray`, shape `(N_sensors, 3)` - sensor positions
- `sensor_stats`: `np.ndarray`, shape `(N_sensors, 9)` - statistics for each sensor (aligned with positions)

### `process_sensor_data(sensor_times, sensor_charges=None, grouping_window_ns=None)`

**Args:**
- `sensor_times`: `np.ndarray` or `list`, shape `(N,)` - hit times for sensor
- `sensor_charges`: `np.ndarray` or `list`, shape `(N,)` - hit charges (optional, defaults to 1.0)
- `grouping_window_ns`: `float` or `None` - time window for grouping hits (default: None, no grouping)

**Returns:** `np.ndarray`, shape `(9,)` - array with 9 summary statistics in order shown above

## License

MIT
