# Snow Detector Python

A starter application for interfacing with the Infineon DEMO-BGT60TR13C radar development board. This application provides live visualization of raw radar measurements.

## Features

- Interface with the DEMO-BGT60TR13C radar board via the Infineon Radar SDK
- Real-time visualization of radar data with multiple plot types:
  - **Time Domain**: Raw IF signal visualization
  - **Range Profile**: FFT-based range detection
  - **Range-Doppler Map**: 2D FFT for simultaneous range and velocity detection
- Simulation mode for development and testing without hardware
- Configurable radar parameters (samples, chirps, update rate)

## Prerequisites

### Hardware
- [Infineon DEMO-BGT60TR13C](https://www.infineon.com/cms/en/product/evaluation-boards/demo-bgt60tr13c/) radar development kit
- USB connection to host computer

### Software
- Python 3.8 or higher
- Infineon Radar Development Kit (RDK) with the `ifxdaq` package

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/chrishajduk84/snow_detector_python.git
   cd snow_detector_python
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the Infineon Radar SDK:
   - Download the Radar Development Kit from [Infineon's website](https://www.infineon.com/cms/en/product/sensor/radar-sensors/radar-sensors-for-iot/60ghz-radar/demo-bgt60tr13c/)
   - Follow Infineon's installation instructions to install the `ifxdaq` Python package
   - The SDK includes necessary drivers for the radar board

## Usage

### Quick Start with Simulation Mode

To test the application without radar hardware:

```bash
python main.py --simulate
```

### Running with Hardware

Connect the DEMO-BGT60TR13C board via USB and run:

```bash
python main.py
```

### Command Line Options

```
Usage: python main.py [OPTIONS]

Options:
  --simulate          Run with simulated radar data (no hardware required)
  --plot-type TYPE    Visualization type: time_domain, range_profile, 
                      range_doppler, or all (default: all)
  --samples N         Number of samples per chirp (default: 64)
  --chirps N          Number of chirps per frame (default: 16)
  --update-rate MS    Plot update interval in milliseconds (default: 50)
```

### Examples

```bash
# Run simulation with all plot types
python main.py --simulate --plot-type all

# Show only range profile with hardware
python main.py --plot-type range_profile

# Custom configuration
python main.py --simulate --samples 128 --chirps 32 --update-rate 100
```

## Project Structure

```
snow_detector_python/
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── src/
    ├── __init__.py
    ├── radar_interface.py  # Radar board interface module
    └── live_plotter.py     # Real-time visualization module
```

## API Reference

### RadarInterface

```python
from src.radar_interface import RadarInterface

# Basic usage with context manager
with RadarInterface() as radar:
    radar.start_acquisition()
    frame = radar.get_frame()  # Returns numpy array
    radar.stop_acquisition()

# Custom configuration
config = {
    "num_samples_per_chirp": 128,
    "num_chirps_per_frame": 32,
    "lower_frequency_hz": 60_000_000_000,
    "upper_frequency_hz": 61_500_000_000,
}
radar = RadarInterface(config)
```

### LivePlotter

```python
from src.live_plotter import LivePlotter

# Create plotter
plotter = LivePlotter(
    num_samples=64,
    num_chirps=16,
    plot_type="all"  # or "time_domain", "range_profile", "range_doppler"
)

# Push frame data
plotter.push_frame(frame_data)

# Start visualization
plotter.start(blocking=True)
```

## Radar Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate_hz` | 1,000,000 | ADC sample rate |
| `num_samples_per_chirp` | 64 | Samples per chirp |
| `num_chirps_per_frame` | 16 | Chirps per frame |
| `lower_frequency_hz` | 60 GHz | Start frequency |
| `upper_frequency_hz` | 61.5 GHz | End frequency |
| `tx_power_level` | 31 | Transmit power (0-31) |
| `if_gain_db` | 33 | IF amplifier gain |
| `frame_repetition_time_s` | 0.1 | Time between frames |

## Troubleshooting

### "ifxdaq package not installed"
The Infineon Radar SDK is not installed. Please download and install the Radar Development Kit from Infineon's website.

### "Failed to connect to radar"
- Ensure the DEMO-BGT60TR13C board is connected via USB
- Check that the correct drivers are installed
- Verify only one application is accessing the board

### Plot not updating
- Ensure data is being pushed to the plotter with `push_frame()`
- Check that the plot window is in focus
- Try reducing the update rate (increase `--update-rate` value)

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.