#!/usr/bin/env python3
"""
Main application for DEMO-BGT60TR13C Radar Data Visualization.

This script provides a starter application to interface with the Infineon
DEMO-BGT60TR13C radar development board and visualize raw measurements live.

Usage:
    python main.py [--plot-type TYPE] [--samples N] [--chirps N]

Options:
    --plot-type     Type of plot: time_domain, range_profile, range_doppler, all
    --samples       Number of samples per chirp (default: 128)
    --chirps        Number of chirps per frame (default: 64)

Examples:
    # Run with real radar hardware
    python main.py

    # Show all plot types
    python main.py --plot-type all
"""

import argparse
import sys
import signal
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from src.live_plotter import LivePlotter
from src.beamforming import Beamformer, BeamformingConfig
from src.material_classifier import MaterialClassifier


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
        "--plot-type",
        type=str,
        default="all",
        choices=["time_domain", "range_profile", "range_doppler", "range_angle", "all"],
        help="Type of visualization to display (default: all)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of samples per chirp (default: 128)",
    )

    parser.add_argument(
        "--chirps",
        type=int,
        default=64,
        help="Number of chirps per frame (default: 64)",
    )

    parser.add_argument(
        "--update-rate",
        type=int,
        default=50,
        help="Plot update interval in milliseconds (default: 50)",
    )
    
    parser.add_argument(
        "--enable-beamforming",
        action="store_true",
        help="Enable beamforming for 3D detection and material classification",
    )
    
    parser.add_argument(
        "--num-antennas",
        type=int,
        default=3,
        help="Number of RX antennas (default: 3)",
    )
    
    parser.add_argument(
        "--detect-materials",
        action="store_true",
        help="Enable material classification (requires --enable-beamforming)",
    )

    return parser.parse_args()


def run_with_hardware(args: argparse.Namespace) -> None:
    """Run the application with real radar hardware.

    Args:
        args: Parsed command line arguments.
    """
    try:
        from ifxdaq.sensor.radar_ifx import RadarIfxFmcw
    except ImportError as e:
        print(f"Error: {e}")
        print(
            "\nThe Infineon Radar SDK is required to use real hardware."
        )
        print(
            "Please install the Radar Development Kit from Infineon."
        )
        sys.exit(1)

    # Initialize beamformer if requested
    beamformer = None
    classifier = None
    
    if args.enable_beamforming:
        bf_config = BeamformingConfig(
            num_antennas=args.num_antennas,
            num_beams=64,
            min_angle_deg=-60.0,
            max_angle_deg=60.0,
            d_by_lambda=0.5
        )
        beamformer = Beamformer(bf_config)
        print(f"Beamforming enabled with {args.num_antennas} antennas")
        
        if args.detect_materials:
            classifier = MaterialClassifier()
            print("Material classification enabled")

    # Create plotter
    plotter = LivePlotter(
        num_samples=args.samples,
        num_chirps=args.chirps,
        plot_type=args.plot_type,
        update_interval_ms=args.update_rate,
        beamformer=beamformer,
    )

    # Set up signal handler for graceful shutdown
    setup_signal_handler(plotter.stop)

    print("DEMO-BGT60TR13C Radar Data Visualization")
    print("=" * 40)
    print(f"Samples per chirp: {args.samples}")
    print(f"Chirps per frame: {args.chirps}")
    print(f"Plot type: {args.plot_type}")
    if args.enable_beamforming:
        print(f"Beamforming: Enabled ({args.num_antennas} antennas)")
        print(f"Material classification: {'Enabled' if args.detect_materials else 'Disabled'}")
    print("=" * 40)
    print("Connecting to radar...")

    # Create default config file
    config_file = RadarIfxFmcw.create_default_config_file()
    print(f"Using config: {config_file}")

    try:
        with RadarIfxFmcw(config_file) as radar:
            print("Connected! Starting data acquisition...")
            print("Close the plot window or press Ctrl+C to exit.")

            # Start plotter in non-blocking mode
            plotter.start(blocking=False)

            frame_count = 0
            dropped_frames = 0
            
            # Iterate over frames from radar
            while plotter.is_running():
                try:
                    frame_data = next(iter(radar))
                except StopIteration:
                    break
                except Exception as e:
                    # Handle frame acquisition errors gracefully
                    if "FRAME_ACQUISITION_FAILED" in str(e):
                        dropped_frames += 1
                        if dropped_frames % 10 == 1:  # Print every 10th drop
                            print(f"Warning: Frame dropped (total: {dropped_frames})")
                        continue
                    else:
                        raise
                
                frame_count += 1
                if frame_data is None:
                    continue
                
                # Extract radar frame (Frame object from ifxdaq)
                radar_frame = frame_data.get('radar')
                temperature = frame_data.get('temperature')

                if radar_frame is None:
                    continue
                
                # Convert Frame object to numpy array
                # Frame object has .data attribute containing the actual array
                if hasattr(radar_frame, 'data'):
                    radar_array = np.array(radar_frame.data)
                else:
                    # If it's already array-like, convert directly
                    radar_array = np.array(radar_frame)
                
                # Material classification (if enabled)
                if classifier is not None and beamformer is not None and frame_count % 10 == 0:
                    # Process every 10th frame for material classification
                    range_angle_map = beamformer.process_frame(radar_array)
                    points_3d, amplitudes = beamformer.range_angle_to_3d(
                        range_angle_map,
                        threshold_db=-25.0,
                        max_range_m=10.0
                    )
                    
                    if len(points_3d) > 0:
                        print(f"\n--- Frame {frame_count} Detection Results ---")
                        angles = beamformer.get_angle_labels()
                        
                        # Classify up to 5 strongest detections
                        sorted_indices = np.argsort(amplitudes)[-5:][::-1]
                        
                        for idx in sorted_indices:
                            if idx >= len(points_3d):
                                continue
                            point = points_3d[idx]
                            amplitude = amplitudes[idx]
                            
                            # Find beam and range indices
                            range_val = np.linalg.norm(point[:2])
                            angle_val = np.rad2deg(np.arctan2(point[0], point[1]))
                            beam_idx = np.argmin(np.abs(angles - angle_val))
                            range_idx = int(range_val * (range_angle_map.shape[1] / 10.0))
                            range_idx = min(range_idx, range_angle_map.shape[1] - 1)
                            
                            features = classifier.extract_features(
                                range_angle_map,
                                beam_idx,
                                range_idx,
                                angles,
                                max_range_m=10.0
                            )
                            material = classifier.classify_material(features)
                            
                            print(f"  Point [{point[0]:5.2f}, {point[1]:5.2f}, {point[2]:5.2f}] m: "
                                  f"{material:12s} | RCS: {amplitude:5.1f} dB | "
                                  f"Texture: {features.texture:.2f} | "
                                  f"Sharpness: {features.peak_sharpness:.2f}")
                
                # Print useful information periodically (every 30 frames)
                elif frame_count % 30 == 1:
                    print(f"Frame {frame_count}: shape={radar_array.shape}, "
                          f"range=[{radar_array.min():.2f}, {radar_array.max():.2f}], "
                          f"temp={temperature}, dropped={dropped_frames}")
                
                # Push to plotter
                plotter.push_frame(radar_array)
                
                # Flush matplotlib events to allow animation to update
                # Use smaller pause to reduce processing time
                plt.pause(0.0001)

            print(f"\nAcquisition stopped. Processed {frame_count} frames, dropped {dropped_frames} frames.")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        plotter.stop()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    run_with_hardware(args)


if __name__ == "__main__":
    main()
