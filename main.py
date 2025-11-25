#!/usr/bin/env python3
"""
Main application for DEMO-BGT60TR13C Radar Data Visualization.

This script provides a starter application to interface with the Infineon
DEMO-BGT60TR13C radar development board and visualize raw measurements live.

Usage:
    python main.py [--simulate] [--plot-type TYPE] [--samples N] [--chirps N]

Options:
    --simulate      Run with simulated radar data (no hardware required)
    --plot-type     Type of plot: time_domain, range_profile, range_doppler, all
    --samples       Number of samples per chirp (default: 64)
    --chirps        Number of chirps per frame (default: 16)

Examples:
    # Run with real radar hardware
    python main.py

    # Run with simulated data
    python main.py --simulate

    # Show all plot types with simulated data
    python main.py --simulate --plot-type all
"""

import argparse
import sys
import signal
from typing import Callable

from src.live_plotter import LivePlotter, SimulatedRadarPlotter


def setup_signal_handler(stop_callback: Callable[[], None]) -> None:
    """Set up signal handler for graceful shutdown.

    Args:
        stop_callback: Function to call when shutdown signal is received.
    """
    def handler(signum, frame):
        print("\nShutting down...")
        stop_callback()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="DEMO-BGT60TR13C Radar Data Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --simulate              Run with simulated data
  %(prog)s --plot-type all         Show all visualization types
  %(prog)s --samples 128 --chirps 32  Custom frame configuration
        """,
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run with simulated radar data (no hardware required)",
    )

    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["time_domain", "range_profile", "range_doppler", "all"],
        help="Type of visualization to display (default: all)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of samples per chirp (default: 64)",
    )

    parser.add_argument(
        "--chirps",
        type=int,
        default=16,
        help="Number of chirps per frame (default: 16)",
    )

    parser.add_argument(
        "--update-rate",
        type=int,
        default=50,
        help="Plot update interval in milliseconds (default: 50)",
    )

    return parser.parse_args()


def run_with_hardware(args: argparse.Namespace) -> None:
    """Run the application with real radar hardware.

    Args:
        args: Parsed command line arguments.
    """
    try:
        from src.radar_interface import RadarInterface
    except ImportError as e:
        print(f"Error: {e}")
        print(
            "\nThe Infineon Radar SDK is required to use real hardware."
        )
        print(
            "Please install the Radar Development Kit from Infineon."
        )
        print("\nAlternatively, run with --simulate to use simulated data.")
        sys.exit(1)

    # Configure radar
    config = {
        "num_samples_per_chirp": args.samples,
        "num_chirps_per_frame": args.chirps,
    }

    # Create plotter
    plotter = LivePlotter(
        num_samples=args.samples,
        num_chirps=args.chirps,
        plot_type=args.plot_type,
        update_interval_ms=args.update_rate,
    )

    # Set up signal handler for graceful shutdown
    setup_signal_handler(plotter.stop)

    print("DEMO-BGT60TR13C Radar Data Visualization")
    print("=" * 40)
    print(f"Samples per chirp: {args.samples}")
    print(f"Chirps per frame: {args.chirps}")
    print(f"Plot type: {args.plot_type}")
    print("=" * 40)
    print("Connecting to radar...")

    try:
        with RadarInterface(config) as radar:
            print("Connected! Starting data acquisition...")
            print("Close the plot window or press Ctrl+C to exit.")

            # Define callback to push frames to plotter
            def frame_callback(frame) -> bool:
                plotter.push_frame(frame)
                return plotter.is_running()

            # Start plotter in non-blocking mode
            plotter.start(blocking=False)

            # Stream frames
            radar.stream_frames(frame_callback)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_with_simulation(args: argparse.Namespace) -> None:
    """Run the application with simulated radar data.

    Args:
        args: Parsed command line arguments.
    """
    print("DEMO-BGT60TR13C Radar Data Visualization (SIMULATION MODE)")
    print("=" * 60)
    print(f"Samples per chirp: {args.samples}")
    print(f"Chirps per frame: {args.chirps}")
    print(f"Plot type: {args.plot_type}")
    print("=" * 60)
    print("Starting simulated data visualization...")
    print("Close the plot window or press Ctrl+C to exit.")

    sim_plotter = SimulatedRadarPlotter(
        num_samples=args.samples,
        num_chirps=args.chirps,
        plot_type=args.plot_type,
        update_interval_ms=args.update_rate,
    )

    # Set up signal handler for graceful shutdown
    setup_signal_handler(sim_plotter.stop)

    try:
        sim_plotter.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sim_plotter.stop()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    if args.simulate:
        run_with_simulation(args)
    else:
        run_with_hardware(args)


if __name__ == "__main__":
    main()
