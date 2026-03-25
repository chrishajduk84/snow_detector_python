"""Texas Instruments IWR1443BOOST radar sensor via UART/USB.

Supports the TI out-of-box demo firmware. Sends chirp configuration
commands over the CLI UART port and reads raw ADC data (complex I/Q)
from the data UART port.

The IWR1443 natively produces complex (I/Q) ADC samples, so no Hilbert
transform is needed (unlike the BGT60TR13C).
"""

import struct
import time
from pathlib import Path

import numpy as np
import serial

from .base import RadarSensor, RadarFrame, RadarConfig

# TI mmWave UART protocol constants
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# TLV types from TI out-of-box demo
TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_AZIMUTH_STATIC_HEATMAP = 4
TLV_RANGE_DOPPLER_HEATMAP = 5
TLV_STATS = 6
TLV_DETECTED_POINTS_SIDE_INFO = 7
TLV_AZIMUTH_ELEVATION_HEATMAP = 8
TLV_RAW_ADC_DATA = 9  # Custom TLV for raw ADC (requires firmware modification)


class IWR1443Sensor(RadarSensor):
    """TI IWR1443BOOST sensor wrapper.

    Args:
        cli_port: Serial port for CLI commands (e.g. 'COM3' or '/dev/ttyACM0').
        data_port: Serial port for data output (e.g. 'COM4' or '/dev/ttyACM1').
        config_path: Path to TI chirp config file (.cfg).
    """

    def __init__(
        self,
        cli_port: str,
        data_port: str,
        config_path: str | Path,
    ):
        self._cli_port_name = cli_port
        self._data_port_name = data_port
        self._config_path = Path(config_path)
        self._cli_serial: serial.Serial | None = None
        self._data_serial: serial.Serial | None = None
        self._config: RadarConfig | None = None
        self._start_time = 0.0

    def connect(self) -> None:
        self._cli_serial = serial.Serial(
            self._cli_port_name,
            baudrate=115200,
            timeout=1.0,
        )
        self._data_serial = serial.Serial(
            self._data_port_name,
            baudrate=921600,
            timeout=1.0,
        )

        # Allow board to boot
        time.sleep(0.1)

        # Parse config and send to device
        self._config = self._parse_config(self._config_path)
        self._send_config(self._config_path)
        self._start_time = time.monotonic()

    def disconnect(self) -> None:
        # Send stop command
        if self._cli_serial and self._cli_serial.is_open:
            self._cli_serial.write(b'sensorStop\n')
            time.sleep(0.1)
            self._cli_serial.close()
        if self._data_serial and self._data_serial.is_open:
            self._data_serial.close()
        self._cli_serial = None
        self._data_serial = None

    def get_config(self) -> RadarConfig:
        if self._config is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._config

    def get_frame(self) -> RadarFrame:
        if self._data_serial is None:
            raise RuntimeError("Not connected. Call connect() first.")

        config = self.get_config()

        # Read until we find the magic word
        header = self._read_header()

        # Read TLV data based on header
        complex_data = self._read_frame_data(header, config)

        timestamp = time.monotonic() - self._start_time

        return RadarFrame(
            complex_data=complex_data,
            timestamp=timestamp,
            config=config,
        )

    def _read_header(self) -> dict:
        """Synchronize on magic word and read frame header."""
        # Synchronize on magic word
        buf = b''
        while True:
            byte = self._data_serial.read(1)
            if not byte:
                raise RuntimeError("Timeout waiting for data")
            buf += byte
            if len(buf) > len(MAGIC_WORD):
                buf = buf[-len(MAGIC_WORD):]
            if buf == MAGIC_WORD:
                break

        # Read remaining header (32 bytes after magic word in TI format)
        header_bytes = self._data_serial.read(32)
        if len(header_bytes) < 32:
            raise RuntimeError("Incomplete header received")

        # Parse header fields
        version, total_packet_len, platform, frame_number, time_cpu_cycles, \
            num_detected_obj, num_tlvs, subframe_number = struct.unpack(
                '<IIIIIIII', header_bytes
            )

        return {
            'version': version,
            'total_packet_len': total_packet_len,
            'platform': platform,
            'frame_number': frame_number,
            'time_cpu_cycles': time_cpu_cycles,
            'num_detected_obj': num_detected_obj,
            'num_tlvs': num_tlvs,
            'subframe_number': subframe_number,
        }

    def _read_frame_data(self, header: dict, config: RadarConfig) -> np.ndarray:
        """Read TLV payloads and extract complex I/Q data.

        For the out-of-box demo, we extract from the range profile TLV.
        For raw ADC mode (custom firmware), we extract interleaved I/Q samples.
        """
        remaining = header['total_packet_len'] - 40  # 8 magic + 32 header
        payload = self._data_serial.read(remaining)
        if len(payload) < remaining:
            raise RuntimeError(f"Incomplete payload: got {len(payload)}, expected {remaining}")

        # Parse TLVs
        offset = 0
        range_profile = None

        while offset < len(payload) - 8:
            tlv_type, tlv_length = struct.unpack('<II', payload[offset:offset + 8])
            offset += 8
            tlv_data = payload[offset:offset + tlv_length]
            offset += tlv_length

            if tlv_type == TLV_RAW_ADC_DATA:
                # Raw ADC: interleaved int16 I/Q samples
                # Layout: [I0_rx0, Q0_rx0, I0_rx1, Q0_rx1, ..., I0_rxN, Q0_rxN, I1_rx0, ...]
                samples = np.frombuffer(tlv_data, dtype=np.int16)
                num_rx = config.num_rx
                num_iq = 2 * num_rx
                total_samples = len(samples) // num_iq
                samples = samples[:total_samples * num_iq].reshape(-1, num_iq)

                complex_data = np.zeros(
                    (config.num_chirps_per_frame, config.num_samples_per_chirp, num_rx),
                    dtype=np.complex64,
                )
                for rx in range(num_rx):
                    i_data = samples[:, 2 * rx].astype(np.float32)
                    q_data = samples[:, 2 * rx + 1].astype(np.float32)
                    iq = i_data + 1j * q_data
                    complex_data[:, :, rx] = iq.reshape(
                        config.num_chirps_per_frame,
                        config.num_samples_per_chirp,
                    )
                return complex_data

            elif tlv_type == TLV_RANGE_PROFILE:
                # Range profile from out-of-box demo: Q-format complex values
                range_profile = np.frombuffer(tlv_data, dtype=np.int16)

        # If we got a range profile but no raw ADC, construct a synthetic frame
        # from the available data (limited — only 1 chirp equivalent)
        if range_profile is not None:
            num_bins = len(range_profile) // 2
            i_data = range_profile[0::2].astype(np.float32)
            q_data = range_profile[1::2].astype(np.float32)
            profile = i_data + 1j * q_data

            # Tile across chirps (range profile is already integrated)
            complex_data = np.zeros(
                (config.num_chirps_per_frame, min(num_bins, config.num_samples_per_chirp), config.num_rx),
                dtype=np.complex64,
            )
            for rx in range(config.num_rx):
                complex_data[:, :, rx] = profile[:config.num_samples_per_chirp]
            return complex_data

        raise RuntimeError("No usable data TLV found in frame")

    def _send_config(self, config_path: Path) -> None:
        """Send chirp configuration commands to the sensor via CLI UART."""
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                self._cli_serial.write((line + '\n').encode('ascii'))
                # Wait for response (TI firmware echoes + sends "Done")
                time.sleep(0.05)
                response = self._cli_serial.read(self._cli_serial.in_waiting or 1)

    @staticmethod
    def _parse_config(config_path: Path) -> RadarConfig:
        """Parse TI .cfg file to extract radar parameters."""
        params = {
            'start_freq_ghz': 77.0,
            'freq_slope_mhz_per_us': 29.982,
            'idle_time_us': 7.0,
            'ramp_end_time_us': 60.0,
            'adc_samples': 256,
            'adc_sample_rate_ksps': 10000,
            'num_chirps': 128,
            'num_frames': 0,
            'frame_period_ms': 100,
            'rx_mask': 0b1111,
            'tx_mask': 0b001,
        }

        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                parts = line.split()
                cmd = parts[0]

                if cmd == 'profileCfg':
                    # profileCfg <id> <start_freq> <idle_time> <adc_start_time>
                    #   <ramp_end_time> <tx_out_power> <tx_phase_shifter>
                    #   <freq_slope> <tx_start_time> <adc_samples>
                    #   <sample_rate> <hpf1_corner> <hpf2_corner> <rx_gain>
                    params['start_freq_ghz'] = float(parts[2])
                    params['idle_time_us'] = float(parts[3])
                    params['ramp_end_time_us'] = float(parts[5])
                    params['freq_slope_mhz_per_us'] = float(parts[8])
                    params['adc_samples'] = int(parts[10])
                    params['adc_sample_rate_ksps'] = int(parts[11])

                elif cmd == 'frameCfg':
                    # frameCfg <chirp_start> <chirp_end> <num_loops>
                    #   <num_frames> <frame_period> <trigger_select> <frame_trigger_delay>
                    params['num_chirps'] = int(parts[3])
                    params['num_frames'] = int(parts[4])
                    params['frame_period_ms'] = float(parts[5])

                elif cmd == 'channelCfg':
                    # channelCfg <rx_mask> <tx_mask> <cascade>
                    params['rx_mask'] = int(parts[1])
                    params['tx_mask'] = int(parts[2])

        # Compute derived parameters
        bandwidth_mhz = params['freq_slope_mhz_per_us'] * params['ramp_end_time_us']
        start_freq_hz = params['start_freq_ghz'] * 1e9
        end_freq_hz = start_freq_hz + bandwidth_mhz * 1e6
        chirp_time_s = (params['idle_time_us'] + params['ramp_end_time_us']) * 1e-6
        num_rx = bin(params['rx_mask']).count('1')

        return RadarConfig(
            start_freq_hz=start_freq_hz,
            end_freq_hz=end_freq_hz,
            num_samples_per_chirp=params['adc_samples'],
            num_chirps_per_frame=params['num_chirps'],
            num_rx=num_rx,
            sample_rate_hz=params['adc_sample_rate_ksps'] * 1e3,
            chirp_repetition_time_s=chirp_time_s,
            frame_repetition_time_s=params['frame_period_ms'] * 1e-3,
        )
