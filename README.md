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
stats = compute_summary_stats(times, charges)     # returns: dict[str, float]

print(stats['total_charge'])              # 5.0
print(stats['first_pulse_time'])          # 10.0
print(stats['charge_weighted_mean_time']) # 26.0
```

Process Prometheus events:

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

# Returns: (positions, stats)
sensor_positions, sensor_stats = process_prometheus_event(event_data)
# sensor_positions: np.ndarray, shape (N_sensors, 3), dtype: float64
# sensor_stats: list[dict[str, float]], length N_sensors
```

Process individual sensor data:

```python
from nt_summary_stats import process_sensor_data

# Input: sensor hit data
sensor_times = [10.0, 10.5, 15.0, 100.0]    # list[float] or np.ndarray(N,)
sensor_charges = [1.0, 0.5, 2.0, 1.0]       # list[float] or np.ndarray(N,), optional

# Returns: dict[str, float]
stats = process_sensor_data(sensor_times, sensor_charges, grouping_window_ns=2.0)
```

## What it does

Computes 9 traditional summary statistics for neutrino telescope sensors:

- `total_charge`: Total charge collected
- `charge_100ns`: Charge within 100ns of first pulse
- `charge_500ns`: Charge within 500ns of first pulse
- `first_pulse_time`: Time of first pulse
- `last_pulse_time`: Time of last pulse
- `charge_20_percent_time`: Time at which 20% of charge is collected
- `charge_50_percent_time`: Time at which 50% of charge is collected
- `charge_weighted_mean_time`: Charge-weighted mean time
- `charge_weighted_std_time`: Charge-weighted standard deviation

As described in the [IceCube paper](https://arxiv.org/abs/2101.11589).

## API

### `compute_summary_stats(times, charges)`

**Args:**
- `times`: `np.ndarray` or `list`, shape `(N,)` - pulse arrival times in ns
- `charges`: `np.ndarray` or `list`, shape `(N,)` - pulse charges

**Returns:** `dict[str, float]` - dictionary with 9 summary statistics

### `process_prometheus_event(event_data, grouping_window_ns=2.0)`

**Args:**
- `event_data`: `dict` - Prometheus event with `photons` key containing sensor data
- `grouping_window_ns`: `float` - time window for grouping hits (default: 2.0 ns)

**Returns:** `tuple[np.ndarray, list[dict]]`
- `sensor_positions`: `np.ndarray`, shape `(N_sensors, 3)` - sensor positions
- `sensor_stats`: `list[dict[str, float]]` - statistics for each sensor

### `process_sensor_data(sensor_times, sensor_charges=None, grouping_window_ns=2.0)`

**Args:**
- `sensor_times`: `np.ndarray` or `list`, shape `(N,)` - hit times for sensor
- `sensor_charges`: `np.ndarray` or `list`, shape `(N,)` - hit charges (optional, defaults to 1.0)
- `grouping_window_ns`: `float` - time window for grouping hits (default: 2.0 ns)

**Returns:** `dict[str, float]` - dictionary with 9 summary statistics

## License

MIT
