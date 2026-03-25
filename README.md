# Radar Dielectric Profiler

Estimate per-range-bin material properties (**εr**, **tan δ**) from FMCW radar
measurements. Supports Infineon BGT60TR13C and TI IWR1443BOOST sensors.

## Features

- **Dual-sensor support** — BGT60TR13C (60 GHz, 5 GHz BW) and IWR1443BOOST (77 GHz, 4 GHz BW)
- **1D U-Net model** — predicts εr, tan δ, and material presence at every range bin
- **Transfer-matrix simulator** — generates unlimited synthetic training data
- **HDF5 storage** — labeled sessions with complex I/Q and ground-truth layer stackups
- **Unified CLI** — `collect`, `simulate`, `train`, `predict`, `visualize` subcommands

## Prerequisites

### Hardware (at least one)
- [Infineon DEMO-BGT60TR13C](https://www.infineon.com/cms/en/product/evaluation-boards/demo-bgt60tr13c/) + USB
- [TI IWR1443BOOST](https://www.ti.com/tool/IWR1443BOOST) + UART (two COM ports)

### Software
- Python 3.11–3.13
- Poetry (`pip install poetry`)

## Installation

```bash
git clone https://github.com/chrishajduk84/snow_detector_python.git
cd snow_detector_python
poetry install
```

## Quick Start

### 1. Generate synthetic training data

```bash
python main.py simulate --num-samples 1000
```

### 2. Train the model

```bash
python main.py train --epochs 100 --augment
```

### 3. Collect real data

Place known materials (e.g. 5 cm HDPE with εr=2.3) in front of the radar:

```bash
python main.py collect --sensor bgt60 --layers "5cm:2.3:0.0004" --num-samples 100
```

Layer format: `<thickness><unit>:<εr>:<tan δ>`, comma-separated for multiple layers.

### 4. Live prediction

```bash
python main.py predict --sensor bgt60
```

### 5. Visualize a session

```bash
python main.py visualize data/real/session_20250101_120000.h5
```

## Project Structure

```
main.py                    CLI entry point
configs/
    bgt60tr13c.json        BGT60TR13C radar config
    iwr1443.cfg            IWR1443 chirp config (TI CLI format)
src/
    sensors/               Sensor abstraction (base, bgt60, iwr1443)
    data/                  HDF5 storage, PyTorch dataset, simulator
    models/                1D U-Net property estimator + trainer
    processing/            FFT, range profile, Doppler processing
    visualization/         Plotting utilities
data/
    real/                  Collected measurement sessions (.h5)
    synthetic/             Simulated training data (.h5)
models/
    best_model.pth         Trained model checkpoint
```

## IWR1443 Setup

For the TI IWR1443BOOST, specify both serial ports:

```bash
python main.py collect --sensor iwr1443 --cli-port COM3 --data-port COM4 \
    --layers "5cm:2.3:0.0004"
```

The board must be running TI's out-of-box demo firmware.

## License

MIT