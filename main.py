#!/usr/bin/env python3
"""
Radar Dielectric Profiler — Main Entry Point

Estimate material properties (εr, tan δ) at each range bin using
FMCW radar sensors. Supports Infineon BGT60TR13C and TI IWR1443BOOST.

Subcommands:
    collect   — Collect labeled measurement data
    simulate  — Generate synthetic training data
    train     — Train the property estimator model
    predict   — Live prediction from a connected sensor
    visualize — Plot stored measurement sessions
"""

import argparse
import signal
import sys
from pathlib import Path

import numpy as np
import torch


def setup_signal_handler(stop_fn=None):
    def handler(signum, frame):
        print("\nShutting down...")
        if stop_fn:
            stop_fn()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)


# ─────────────────────────────────────────────
# Sensor factory
# ─────────────────────────────────────────────

def create_sensor(args):
    """Instantiate the appropriate radar sensor from CLI args."""
    if args.sensor == "bgt60":
        from src.sensors.bgt60tr13c import BGT60TR13CSensor
        config_path = getattr(args, 'config', None) or "configs/bgt60tr13c.json"
        return BGT60TR13CSensor(config_path=config_path)
    elif args.sensor == "iwr1443":
        from src.sensors.iwr1443 import IWR1443Sensor
        config_path = getattr(args, 'config', None) or "configs/iwr1443.cfg"
        return IWR1443Sensor(
            cli_port=args.cli_port,
            data_port=args.data_port,
            config_path=config_path,
        )
    else:
        print(f"Unknown sensor: {args.sensor}")
        sys.exit(1)


# ─────────────────────────────────────────────
# Subcommand: collect
# ─────────────────────────────────────────────

def cmd_collect(args):
    """Collect radar data with known layer stackup labels."""
    from src.data.hdf5_store import MeasurementSession

    # Parse layer specification: "5cm:2.3:0.0004,5cm:6.5:0.005"
    layers = parse_layers(args.layers)
    print(f"Layer stackup ({len(layers)} layers):")
    for i, (t, e, d) in enumerate(layers):
        print(f"  Layer {i+1}: {t*100:.1f} cm, εr={e:.2f}, tan(δ)={d:.4f}")

    layer_array = np.array(layers, dtype=np.float64)

    sensor = create_sensor(args)
    with sensor:
        config = sensor.get_config()
        print(f"\nSensor: {args.sensor}")
        print(f"Bandwidth: {config.bandwidth_hz/1e9:.1f} GHz")
        print(f"Range resolution: {config.range_resolution_m*100:.1f} cm")
        print(f"\nCollecting {args.num_samples} frames...")

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        h5_path = output_dir / f"session_{timestamp}.h5"

        with MeasurementSession(
            h5_path, config, layer_array,
            sensor_type=args.sensor,
            notes=args.notes,
        ) as session:
            for i in range(args.num_samples):
                try:
                    frame = sensor.get_frame()
                    session.add_frame(frame)
                    if (i + 1) % 10 == 0:
                        print(f"  {i+1}/{args.num_samples} frames collected")
                except RuntimeError as e:
                    print(f"  Frame dropped: {e}")

        print(f"\nSaved to {h5_path}")


def parse_layers(spec: str) -> list[tuple[float, float, float]]:
    """Parse layer spec string like '5cm:2.3:0.0004,10cm:6.5:0.005'.

    Returns list of (thickness_m, eps_r, tan_delta) tuples.
    """
    layers = []
    for part in spec.split(','):
        fields = part.strip().split(':')
        if len(fields) != 3:
            print(f"Error: Invalid layer spec '{part}'. Expected 'THICKNESSunit:EPS_R:TAN_DELTA'")
            sys.exit(1)

        thickness_str = fields[0].strip().lower()
        eps_r = float(fields[1])
        tan_delta = float(fields[2])

        # Parse thickness with unit
        if thickness_str.endswith('cm'):
            thickness_m = float(thickness_str[:-2]) / 100.0
        elif thickness_str.endswith('mm'):
            thickness_m = float(thickness_str[:-2]) / 1000.0
        elif thickness_str.endswith('m'):
            thickness_m = float(thickness_str[:-1])
        else:
            thickness_m = float(thickness_str)  # assume meters

        layers.append((thickness_m, eps_r, tan_delta))
    return layers


# ─────────────────────────────────────────────
# Subcommand: simulate
# ─────────────────────────────────────────────

def cmd_simulate(args):
    """Generate synthetic training data via transfer matrix simulation."""
    from src.sensors.base import RadarConfig
    from src.data.simulator import generate_synthetic_dataset

    # Use a default config matching the BGT60TR13C 5 GHz bandwidth setup
    config = RadarConfig(
        start_freq_hz=58e9,
        end_freq_hz=63e9,
        num_samples_per_chirp=128,
        num_chirps_per_frame=64,
        num_rx=3,
        sample_rate_hz=2e6,
        chirp_repetition_time_s=0.0005,
        frame_repetition_time_s=0.1,
    )

    print(f"Generating {args.num_samples} synthetic stackups...")
    print(f"  Frames per stackup: {args.frames_per_stackup}")
    print(f"  Total samples: {args.num_samples * args.frames_per_stackup}")

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        config=config,
        num_samples=args.num_samples,
        frames_per_stackup=args.frames_per_stackup,
        seed=args.seed,
    )


# ─────────────────────────────────────────────
# Subcommand: train
# ─────────────────────────────────────────────

def cmd_train(args):
    """Train the property estimator model."""
    from src.data.dataset import DielectricProfileDataset, NoiseAugmentation
    from src.data.hdf5_store import list_sessions
    from src.models.property_estimator import PropertyEstimator
    from src.models.trainer import PropertyEstimatorTrainer

    # Collect HDF5 files
    h5_paths = []

    real_dir = Path(args.real_data_dir)
    if real_dir.exists():
        real_sessions = list_sessions(real_dir)
        print(f"Found {len(real_sessions)} real measurement sessions")
        h5_paths.extend(real_sessions)

    synthetic_dir = Path(args.synthetic_data_dir)
    if synthetic_dir.exists():
        synthetic_sessions = list_sessions(synthetic_dir)
        print(f"Found {len(synthetic_sessions)} synthetic data files")
        h5_paths.extend(synthetic_sessions)

    if not h5_paths:
        print("Error: No training data found. Run 'collect' or 'simulate' first.")
        sys.exit(1)

    print(f"Loading {len(h5_paths)} data files...")
    transform = NoiseAugmentation() if args.augment else None
    dataset = DielectricProfileDataset(h5_paths, transform=transform)
    print(f"Total samples: {len(dataset)}")

    # Determine input channels from first sample
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]

    model = PropertyEstimator(in_channels=in_channels, base_filters=args.base_filters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    trainer = PropertyEstimatorTrainer(model, device=device, lr=args.lr)

    if args.resume:
        checkpoint = Path(args.checkpoint_dir) / "best_model.pth"
        if checkpoint.exists():
            trainer.load_checkpoint(checkpoint)
            print(f"Resumed from {checkpoint}")

    history = trainer.train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Plot training curves
    if not args.no_plot:
        from src.visualization.plots import plot_training_curves
        import matplotlib.pyplot as plt
        fig = plot_training_curves(history)
        fig.savefig(Path(args.checkpoint_dir) / "training_curves.png", dpi=150)
        plt.show()


# ─────────────────────────────────────────────
# Subcommand: predict
# ─────────────────────────────────────────────

def cmd_predict(args):
    """Live prediction of dielectric properties from sensor data."""
    from src.models.property_estimator import PropertyEstimator
    from src.visualization.plots import plot_property_map
    import matplotlib.pyplot as plt

    # Load model
    checkpoint = Path(args.checkpoint_dir) / "best_model.pth"
    if not checkpoint.exists():
        print(f"Error: No model found at {checkpoint}. Run 'train' first.")
        sys.exit(1)

    sensor = create_sensor(args)
    with sensor:
        config = sensor.get_config()
        in_channels = config.num_rx * 2
        model = PropertyEstimator(in_channels=in_channels, base_filters=args.base_filters)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()

        print(f"Live prediction from {args.sensor}")
        print(f"Range resolution: {config.range_resolution_m*100:.1f} cm")
        print("Press Ctrl+C to stop.\n")

        setup_signal_handler()

        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        while True:
            try:
                frame = sensor.get_frame()
            except RuntimeError:
                continue

            # Aggregate chirps and convert to model input
            avg = np.mean(frame.complex_data, axis=0)  # (samples, rx)
            channels = np.zeros((1, config.num_rx * 2, config.num_samples_per_chirp), dtype=np.float32)
            for rx in range(config.num_rx):
                channels[0, 2 * rx] = avg[:, rx].real
                channels[0, 2 * rx + 1] = avg[:, rx].imag

            with torch.no_grad():
                pred = model(torch.from_numpy(channels))
            pred_np = pred[0].numpy()

            # Update plots
            x = np.arange(config.num_samples_per_chirp) * config.range_resolution_m

            for ax in axes:
                ax.clear()

            axes[0].plot(x, pred_np[0], 'b-')
            axes[0].set_ylabel("εr")
            axes[0].set_title("Live Dielectric Profile")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(x, pred_np[1], 'g-')
            axes[1].set_ylabel("tan(δ)")
            axes[1].grid(True, alpha=0.3)

            axes[2].fill_between(x, pred_np[2], alpha=0.5, color='orange')
            axes[2].set_ylabel("Presence")
            axes[2].set_xlabel("Range (m)")
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.05)


# ─────────────────────────────────────────────
# Subcommand: visualize
# ─────────────────────────────────────────────

def cmd_visualize(args):
    """Visualize a stored measurement session."""
    from src.data.hdf5_store import load_session
    from src.visualization.plots import plot_range_profile, plot_property_map
    import matplotlib.pyplot as plt

    session = load_session(args.session)
    config = session['config']

    print(f"Session: {args.session}")
    print(f"  Sensor: {session['metadata']['sensor_type']}")
    print(f"  Date: {session['metadata']['date']}")
    print(f"  Samples: {len(session['samples'])}")
    print(f"  Layers: {session['layer_stackup'].shape[0]}")

    # Average all frames
    all_iq = np.stack([s['complex_iq'] for s in session['samples']])
    avg_frame = np.mean(np.mean(all_iq, axis=0), axis=0)  # (samples, rx)

    plot_range_profile(avg_frame, config=config, title="Averaged Range Profile")

    # Show ground truth property labels
    labels = session['range_bin_labels'].T  # (3, num_bins)
    plot_property_map(labels, config=config, title="Ground Truth Properties")

    plt.show()


# ─────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radar Dielectric Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── collect ──
    p = subparsers.add_parser("collect", help="Collect labeled measurement data")
    p.add_argument("--sensor", choices=["bgt60", "iwr1443"], required=True)
    p.add_argument("--layers", required=True,
                   help="Layer stackup: 'THICKNESSunit:EPS_R:TAN_D,...' e.g. '5cm:2.3:0.0004,5cm:6.5:0.005'")
    p.add_argument("--num-samples", type=int, default=100, help="Frames to collect")
    p.add_argument("--output-dir", default="data/real", help="Output directory")
    p.add_argument("--config", default=None, help="Sensor config file path")
    p.add_argument("--notes", default="", help="Session notes")
    # IWR1443-specific
    p.add_argument("--cli-port", default="COM3", help="IWR1443 CLI serial port")
    p.add_argument("--data-port", default="COM4", help="IWR1443 data serial port")

    # ── simulate ──
    p = subparsers.add_parser("simulate", help="Generate synthetic training data")
    p.add_argument("--num-samples", type=int, default=1000, help="Number of unique stackups")
    p.add_argument("--frames-per-stackup", type=int, default=10, help="Noise variations per stackup")
    p.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    # ── train ──
    p = subparsers.add_parser("train", help="Train the property estimator model")
    p.add_argument("--real-data-dir", default="data/real", help="Real data directory")
    p.add_argument("--synthetic-data-dir", default="data/synthetic", help="Synthetic data directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-filters", type=int, default=32, help="Base filter count for U-Net")
    p.add_argument("--checkpoint-dir", default="models")
    p.add_argument("--augment", action="store_true", help="Enable noise augmentation")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--no-plot", action="store_true", help="Skip training curve plot")

    # ── predict ──
    p = subparsers.add_parser("predict", help="Live prediction from sensor")
    p.add_argument("--sensor", choices=["bgt60", "iwr1443"], required=True)
    p.add_argument("--config", default=None, help="Sensor config file path")
    p.add_argument("--checkpoint-dir", default="models")
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--cli-port", default="COM3")
    p.add_argument("--data-port", default="COM4")

    # ── visualize ──
    p = subparsers.add_parser("visualize", help="Visualize stored session")
    p.add_argument("session", help="Path to HDF5 session file")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "collect": cmd_collect,
        "simulate": cmd_simulate,
        "train": cmd_train,
        "predict": cmd_predict,
        "visualize": cmd_visualize,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
