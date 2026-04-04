import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

MPL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


SEED = 17
rng = np.random.default_rng(SEED)

NUM_DRONES = 5
DT = 2.0
MS_PER_SECOND = 1000.0
SOUND_SPEED_MPS = 1500.0
KNOT_TO_MPS = 0.514444

COLLISION_SIM_DURATION_S = 6 * 3600.0
LOW_COLLISION_TRIALS = 60
LOW_MISSION_TRIALS = 40
HIGH_COLLISION_TRIALS = 200
HIGH_MISSION_TRIALS = 250
MISSION_TIMEOUT_S = 2 * 3600.0
TRACK_SAMPLE_EVERY_STEPS = 15

CRUISE_SPEED_MPS = 3.0 * KNOT_TO_MPS
MAX_ACCEL_MPS2 = 0.20
WAYPOINT_HIT_RADIUS_M = 10.0
COLLISION_RADIUS_M = 1.0
SAFE_SEPARATION_M = 8.0
AVOIDANCE_GAIN = 5.0
LOCAL_PROXIMITY_THRESHOLD_M = 5.0
LOCAL_PROXIMITY_GAIN = 12.0
NOMINAL_SPACING_M = 7.0
DEAD_RECKONING_HORIZON_S = 12.0
MOTION_NOISE_STD_M = 0.03
GOSSIP_PACKET_BYTES = 32
GOSSIP_SENDERS_PER_STEP = 5
GOSSIP_ACCESS_MISS_PROB = 0.01
FREQUENCY_DIVERSITY_OFFSETS_KHZ = (-2.0, 2.0)
COMMAND_REQUEST_INTERVAL_S = 15.0

OPERATOR_DATA_RATE_BPS = 5120.0
SWARM_DATA_RATE_BPS = 10240.0
PREAMBLE_S = 0.08
MAX_HANDSHAKE_ATTEMPTS = 4
HANDSHAKE_BACKOFF_S = 10.0

TX_LEVEL_DB = 188.0
OPERATOR_TO_DRONE_FREQ_KHZ = 28.0
DRONE_TO_DRONE_FREQ_KHZ = 50.0
BANDWIDTH_HZ = 3000.0
RECEIVER_NOISE_FIGURE_DB = 6.0
SPREADING_K = 18.0
SURFACE_REFLECTION_COEFF = 0.85
BOTTOM_REFLECTION_COEFF = 0.55
MAX_MULTIPATH_PENALTY_DB = 18.0
MAX_DOPPLER_PENALTY_DB = 8.0
SHADOWING_SIGMA_DB = 2.5
BAD_FADE_PROB = 0.10
BAD_FADE_DB_MIN = 10.0
BAD_FADE_DB_MAX = 25.0
NARROWBAND_FADE_PROB = 0.16
NARROWBAND_FADE_DB_MIN = 8.0
NARROWBAND_FADE_DB_MAX = 18.0
NARROWBAND_FADE_WIDTH_KHZ = 1.0
OPERATOR_TRANSIENT_FADE_TRIGGER_PROB = 0.04
SWARM_TRANSIENT_FADE_TRIGGER_PROB = 0.025
TRANSIENT_FADE_DURATION_MIN_S = 2.0
TRANSIENT_FADE_DURATION_MAX_S = 5.0
TRANSIENT_FADE_COOLDOWN_SCALE_MIN = 0.5
TRANSIENT_FADE_COOLDOWN_SCALE_MAX = 2.0
OPERATOR_TRANSIENT_FADE_BIAS_DB_MIN = 7.0
OPERATOR_TRANSIENT_FADE_BIAS_DB_MAX = 14.0
SWARM_TRANSIENT_FADE_BIAS_DB_MIN = 4.0
SWARM_TRANSIENT_FADE_BIAS_DB_MAX = 9.0
LATENCY_JITTER_MIN = 0.90
LATENCY_JITTER_MAX = 1.30
MISSION_SHADOW_MEAN_DB_MIN = 6.0
MISSION_SHADOW_MEAN_DB_MAX = 10.0

OPERATOR_POS = np.array([-1000.0, 0.0, 0.0])

FORMATION_OFFSETS = np.array(
    [
        [-NOMINAL_SPACING_M, 0.0, 0.0],
        [NOMINAL_SPACING_M, 0.0, 0.0],
        [0.0, -NOMINAL_SPACING_M, 0.0],
        [0.0, NOMINAL_SPACING_M, 0.0],
        [0.0, 0.0, 0.0],
    ]
)

PATROL_START_CENTER = np.array([80.0, 0.0, 45.0])
COLLISION_START_CENTER = np.array([540.0, 0.0, 45.0])
MISSION_WAYPOINT_CENTERS = np.array(
    [
        [220.0, 120.0, 45.0],
        [380.0, -90.0, 50.0],
        [540.0, 140.0, 45.0],
        [700.0, -120.0, 50.0],
        [860.0, 20.0, 45.0],
    ]
)
PATROL_CENTERS = np.vstack([PATROL_START_CENTER, MISSION_WAYPOINT_CENTERS])
PATROL_TARGETS = PATROL_CENTERS[:, None, :] + FORMATION_OFFSETS[None, :, :]
COLLISION_PATROL_CENTERS = np.array(
    [
        [540.0, 0.0, 45.0],
        [620.0, 20.0, 45.0],
        [700.0, 0.0, 45.0],
        [620.0, -20.0, 45.0],
        [540.0, 0.0, 45.0],
        [460.0, 20.0, 45.0],
        [380.0, 0.0, 45.0],
        [460.0, -20.0, 45.0],
    ],
    dtype=float,
)
COLLISION_PATROL_TARGETS = COLLISION_PATROL_CENTERS[:, None, :] + FORMATION_OFFSETS[None, :, :]


@dataclass(frozen=True)
class AcousticEnvironment:
    name: str
    water_depth_m: float
    wind_speed_mps: float
    shipping_activity: float
    current_mean_mps: np.ndarray
    current_sigma_mps: float
    comm_shadow_mean_db: float


@dataclass(frozen=True)
class Mitigation:
    name: str
    hopping_enabled: bool
    redundancy_copies: int
    payload_scale: float


@dataclass(frozen=True)
class PacketProfile:
    name: str
    packet_bytes: int
    encrypted: bool


@dataclass
class PacketResult:
    success: bool
    latency_s: float
    snr_db: float
    transmissions: int


@dataclass
class HandshakeResult:
    success: bool
    latency_s: float
    packet_tx_count: int
    packet_success_count: int
    retries: int
    snr_samples: list
    leader_idx: int


@dataclass
class CollisionTrialResult:
    trial_seed: int
    collision_free: bool
    min_separation_m: float
    collision_time_s: Optional[float]
    gossip_delivery_rate: float
    mean_peer_age_s: float
    mean_gossip_latency_ms: float
    mean_gossip_bitrate_bps: float
    track_samples: np.ndarray


@dataclass
class MissionTrialResult:
    trial_seed: int
    success: bool
    collision_free: bool
    completed_waypoints: int
    total_time_s: float
    min_separation_m: float
    command_loss_rate: float
    arrival_loss_rate: float
    mean_command_latency_ms: float
    mean_arrival_latency_ms: float
    mean_retries_per_waypoint: float
    mean_snr_db: float
    gossip_delivery_rate: float
    mean_peer_age_s: float
    mean_gossip_latency_ms: float
    mean_gossip_bitrate_bps: float
    track_samples: np.ndarray
    chosen_leaders: list


@dataclass(frozen=True)
class TrialPreset:
    name: str
    collision_trials: int
    mission_trials: int


@dataclass
class SwarmBeliefState:
    perceived_positions: np.ndarray
    perceived_velocities: np.ndarray
    peer_age_s: np.ndarray
    next_sender_idx: int


@dataclass
class TransientFadeState:
    bias_db: float = 0.0
    degraded_remaining_s: float = 0.0
    pending_cooldown_s: float = 0.0
    cooldown_remaining_s: float = 0.0


TRIAL_PRESETS = {
    "low": TrialPreset(name="low", collision_trials=LOW_COLLISION_TRIALS, mission_trials=LOW_MISSION_TRIALS),
    "high": TrialPreset(name="high", collision_trials=HIGH_COLLISION_TRIALS, mission_trials=HIGH_MISSION_TRIALS),
}
DEFAULT_TRIAL_PRESET = "low"


COLLISION_ENVIRONMENT = AcousticEnvironment(
    name="Collision patrol",
    water_depth_m=140.0,
    wind_speed_mps=5.0,
    shipping_activity=0.3,
    current_mean_mps=np.array([0.08, 0.03, 0.0]),
    current_sigma_mps=0.04,
    comm_shadow_mean_db=0.0,
)

MISSION_ENVIRONMENT = AcousticEnvironment(
    name="Operator mission",
    water_depth_m=140.0,
    wind_speed_mps=10.0,
    shipping_activity=0.7,
    current_mean_mps=np.array([0.10, 0.04, 0.0]),
    current_sigma_mps=0.05,
    comm_shadow_mean_db=8.0,
)

BASELINE = Mitigation(name="No mitigation", hopping_enabled=False, redundancy_copies=1, payload_scale=1.0)
MITIGATED = Mitigation(name="Frequency hopping + redundancy", hopping_enabled=True, redundancy_copies=2, payload_scale=1.0)
PACKET_PROFILES = [
    PacketProfile(name="32 B clear", packet_bytes=32, encrypted=False),
    PacketProfile(name="64 B encrypted", packet_bytes=64, encrypted=True),
]

OPERATOR_DIVERSITY_FREQS_KHZ = tuple(OPERATOR_TO_DRONE_FREQ_KHZ + offset for offset in FREQUENCY_DIVERSITY_OFFSETS_KHZ)
DRONE_DIVERSITY_FREQS_KHZ = tuple(DRONE_TO_DRONE_FREQ_KHZ + offset for offset in FREQUENCY_DIVERSITY_OFFSETS_KHZ)
OPERATOR_HOP_PAIRS_KHZ = (
    (24.0, 28.0),
    (26.0, 30.0),
    (28.0, 32.0),
)
DRONE_HOP_PAIRS_KHZ = (
    (46.0, 50.0),
    (48.0, 52.0),
    (50.0, 54.0),
)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_unit(vec):
    mag = np.linalg.norm(vec)
    if mag < 1e-9:
        return np.zeros_like(vec)
    return vec / mag


def safe_unit_rows(array):
    magnitudes = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, np.maximum(magnitudes, 1e-9), out=np.zeros_like(array), where=magnitudes > 1e-9)


def pairwise_min_separation(positions):
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper_indices = np.triu_indices(NUM_DRONES, k=1)
    return float(np.min(distances[upper_indices]))


def mean_peer_age(peer_age_s):
    mask = ~np.eye(NUM_DRONES, dtype=bool)
    return float(np.mean(peer_age_s[mask]))


def init_swarm_belief_state(positions, velocities):
    repeated_positions = np.repeat(positions[np.newaxis, :, :], NUM_DRONES, axis=0)
    repeated_velocities = np.repeat(velocities[np.newaxis, :, :], NUM_DRONES, axis=0)
    peer_age_s = np.zeros((NUM_DRONES, NUM_DRONES), dtype=float)
    return SwarmBeliefState(
        perceived_positions=repeated_positions,
        perceived_velocities=repeated_velocities,
        peer_age_s=peer_age_s,
        next_sender_idx=0,
    )


def update_swarm_beliefs(positions, velocities, belief_state, environment, mitigation, packet_bytes, trial_rng, swarm_link_state):
    belief_state.peer_age_s += DT
    diag_idx = np.arange(NUM_DRONES)
    belief_state.perceived_positions[diag_idx, diag_idx] = positions
    belief_state.perceived_velocities[diag_idx, diag_idx] = velocities
    belief_state.peer_age_s[diag_idx, diag_idx] = 0.0

    tx_count = 0
    success_count = 0
    successful_latency_sum_s = 0.0
    transmitted_bits_sum = 0.0
    access_miss_prob = GOSSIP_ACCESS_MISS_PROB
    packet_bytes_on_air = effective_packet_bytes(packet_bytes, mitigation)
    gossip_slot_elapsed_s = DT / max(GOSSIP_SENDERS_PER_STEP * (NUM_DRONES - 1), 1)

    for slot_idx in range(GOSSIP_SENDERS_PER_STEP):
        sender_idx = (belief_state.next_sender_idx + slot_idx) % NUM_DRONES
        sender_pos = positions[sender_idx]
        sender_vel = velocities[sender_idx]
        for receiver_idx in range(NUM_DRONES):
            if receiver_idx == sender_idx:
                continue
            tx_count += 1
            if trial_rng.random() < access_miss_prob:
                advance_transient_fade_state(swarm_link_state, gossip_slot_elapsed_s)
                continue
            gossip = transmit_packet(
                sender_pos,
                positions[receiver_idx],
                sender_vel,
                velocities[receiver_idx],
                packet_bytes,
                environment,
                mitigation,
                trial_rng,
                DRONE_TO_DRONE_FREQ_KHZ,
                swarm_link_state,
            )
            transmitted_bits_sum += gossip.transmissions * packet_bytes_on_air * 8.0
            if gossip.success:
                belief_state.perceived_positions[receiver_idx, sender_idx] = sender_pos
                belief_state.perceived_velocities[receiver_idx, sender_idx] = sender_vel
                belief_state.peer_age_s[receiver_idx, sender_idx] = 0.0
                success_count += 1
                successful_latency_sum_s += gossip.latency_s
            advance_transient_fade_state(swarm_link_state, gossip_slot_elapsed_s)

    belief_state.next_sender_idx = (belief_state.next_sender_idx + GOSSIP_SENDERS_PER_STEP) % NUM_DRONES
    return tx_count, success_count, successful_latency_sum_s, transmitted_bits_sum


def init_fleet(trial_rng, start_center=PATROL_START_CENTER):
    start = start_center + FORMATION_OFFSETS
    jitter = trial_rng.normal(0.0, [0.8, 0.8, 0.6], size=(NUM_DRONES, 3))
    positions = start + jitter
    velocities = np.zeros((NUM_DRONES, 3))
    control_bias = trial_rng.normal(0.0, [0.015, 0.015, 0.005], size=(NUM_DRONES, 3))
    control_bias[:, 2] = 0.0
    return positions, velocities, control_bias


def sample_current(environment, trial_rng):
    current = environment.current_mean_mps + trial_rng.normal(0.0, environment.current_sigma_mps, size=3)
    current[2] = 0.0
    return current


def shipping_noise_db(freq_khz, shipping_activity):
    return 40.0 + 20.0 * (shipping_activity - 0.5) + 26.0 * math.log10(freq_khz) - 60.0 * math.log10(freq_khz + 0.03)


def wind_noise_db(freq_khz, wind_speed_mps):
    return 50.0 + 7.5 * math.sqrt(max(wind_speed_mps, 0.0)) + 20.0 * math.log10(freq_khz) - 40.0 * math.log10(freq_khz + 0.4)


def linear_to_db(value):
    return 10.0 * math.log10(max(value, 1e-12))


def db_to_linear(value):
    return 10.0 ** (value / 10.0)


def thorp_absorption_db_per_km(freq_khz):
    f2 = freq_khz * freq_khz
    return 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


def environment_noise_db(environment):
    return environment_noise_db_for_freq(environment, OPERATOR_TO_DRONE_FREQ_KHZ)


def environment_noise_db_for_freq(environment, freq_khz):
    thermal = -15.0 + 20.0 * math.log10(freq_khz)
    spectral_density_db = linear_to_db(
        db_to_linear(shipping_noise_db(freq_khz, environment.shipping_activity))
        + db_to_linear(wind_noise_db(freq_khz, environment.wind_speed_mps))
        + db_to_linear(thermal)
    )
    return spectral_density_db + 10.0 * math.log10(BANDWIDTH_HZ) + RECEIVER_NOISE_FIGURE_DB


def transmission_loss_db(range_m, freq_khz):
    safe_range = max(range_m, 1.0)
    spreading = SPREADING_K * math.log10(safe_range)
    absorption = thorp_absorption_db_per_km(freq_khz) * (safe_range / 1000.0)
    return spreading + absorption


def data_rate_bps_for_freq(freq_khz):
    if abs(freq_khz - OPERATOR_TO_DRONE_FREQ_KHZ) <= abs(freq_khz - DRONE_TO_DRONE_FREQ_KHZ):
        return OPERATOR_DATA_RATE_BPS
    return SWARM_DATA_RATE_BPS


def transient_fade_parameters(freq_khz):
    if abs(freq_khz - OPERATOR_TO_DRONE_FREQ_KHZ) <= abs(freq_khz - DRONE_TO_DRONE_FREQ_KHZ):
        return (
            OPERATOR_TRANSIENT_FADE_TRIGGER_PROB,
            OPERATOR_TRANSIENT_FADE_BIAS_DB_MIN,
            OPERATOR_TRANSIENT_FADE_BIAS_DB_MAX,
        )
    return (
        SWARM_TRANSIENT_FADE_TRIGGER_PROB,
        SWARM_TRANSIENT_FADE_BIAS_DB_MIN,
        SWARM_TRANSIENT_FADE_BIAS_DB_MAX,
    )


def advance_transient_fade_state(link_state, elapsed_s):
    remaining = max(0.0, elapsed_s)
    while remaining > 1e-9:
        if link_state.degraded_remaining_s > 0.0:
            step = min(remaining, link_state.degraded_remaining_s)
            link_state.degraded_remaining_s -= step
            remaining -= step
            if link_state.degraded_remaining_s <= 1e-9:
                link_state.degraded_remaining_s = 0.0
                link_state.bias_db = 0.0
                if link_state.pending_cooldown_s > 0.0:
                    link_state.cooldown_remaining_s = link_state.pending_cooldown_s
                    link_state.pending_cooldown_s = 0.0
            continue

        if link_state.cooldown_remaining_s > 0.0:
            step = min(remaining, link_state.cooldown_remaining_s)
            link_state.cooldown_remaining_s -= step
            remaining -= step
            if link_state.cooldown_remaining_s <= 1e-9:
                link_state.cooldown_remaining_s = 0.0
            continue

        break


def sample_transient_fade_bias_db(link_state, freq_khz, trial_rng):
    if link_state.degraded_remaining_s <= 0.0 and link_state.cooldown_remaining_s <= 0.0 and link_state.pending_cooldown_s <= 0.0:
        trigger_prob, bias_min_db, bias_max_db = transient_fade_parameters(freq_khz)
        if trial_rng.random() < trigger_prob:
            degraded_duration_s = trial_rng.uniform(TRANSIENT_FADE_DURATION_MIN_S, TRANSIENT_FADE_DURATION_MAX_S)
            link_state.bias_db = trial_rng.uniform(bias_min_db, bias_max_db)
            link_state.degraded_remaining_s = degraded_duration_s
            link_state.pending_cooldown_s = trial_rng.uniform(
                TRANSIENT_FADE_COOLDOWN_SCALE_MIN * degraded_duration_s,
                TRANSIENT_FADE_COOLDOWN_SCALE_MAX * degraded_duration_s,
            )

    if link_state.degraded_remaining_s > 0.0:
        return link_state.bias_db * trial_rng.uniform(0.75, 1.05)
    return 0.0


def multipath_penalty_db(range_m, tx_depth_m, rx_depth_m, environment, freq_khz):
    if range_m < 5.0:
        return 0.0

    direct = math.hypot(range_m, tx_depth_m - rx_depth_m)
    surface = math.hypot(range_m, tx_depth_m + rx_depth_m)
    bottom = math.hypot(range_m, 2.0 * environment.water_depth_m - tx_depth_m - rx_depth_m)

    direct_amp = 1.0 / max(direct, 1.0)
    reflected_amp = (
        SURFACE_REFLECTION_COEFF / max(surface, 1.0)
        + BOTTOM_REFLECTION_COEFF / max(bottom, 1.0)
    )
    excess_path = max(surface - direct, bottom - direct, 0.0)
    phase_term = 2.0 * math.pi * freq_khz * 1000.0 * excess_path / SOUND_SPEED_MPS

    coherent_gain = abs(direct_amp + reflected_amp * math.cos(phase_term)) / max(direct_amp, 1e-9)
    fading_penalty = -20.0 * math.log10(clamp(coherent_gain, 1e-4, 1e4))
    symbol_s = 1.0 / data_rate_bps_for_freq(freq_khz)
    delay_spread_s = excess_path / SOUND_SPEED_MPS
    isi_penalty = 10.0 * math.log10(1.0 + delay_spread_s / max(symbol_s, 1e-6))
    return clamp(fading_penalty + isi_penalty, 0.0, MAX_MULTIPATH_PENALTY_DB)


def doppler_penalty_db(radial_velocity_mps, freq_khz):
    doppler_hz = abs(radial_velocity_mps) * freq_khz * 1000.0 / SOUND_SPEED_MPS
    normalized = doppler_hz / max(data_rate_bps_for_freq(freq_khz) / 4.0, 1.0)
    return clamp(10.0 * math.log10(1.0 + normalized * normalized), 0.0, MAX_DOPPLER_PENALTY_DB)


def link_snr_db(tx_pos, rx_pos, tx_vel, rx_vel, environment, fading_db, freq_khz):
    delta = rx_pos - tx_pos
    range_m = float(np.linalg.norm(delta))
    radial_unit = safe_unit(delta)
    radial_velocity = float(np.dot(rx_vel - tx_vel, radial_unit))

    tl_db = transmission_loss_db(range_m, freq_khz)
    mp_db = multipath_penalty_db(range_m, tx_pos[2], rx_pos[2], environment, freq_khz)
    doppler_db = doppler_penalty_db(radial_velocity, freq_khz)
    noise_db = environment_noise_db_for_freq(environment, freq_khz)
    snr_db = TX_LEVEL_DB - tl_db - mp_db - doppler_db - fading_db - noise_db
    return snr_db, range_m


def packet_error_rate(snr_db, packet_bytes):
    eb_n0 = db_to_linear(snr_db)
    ber = 0.5 * math.erfc(math.sqrt(max(eb_n0, 0.0)))
    bits = packet_bytes * 8
    return clamp(1.0 - (1.0 - ber) ** bits, 0.0, 1.0)


def effective_packet_bytes(packet_bytes, mitigation):
    return max(8, int(round(packet_bytes * mitigation.payload_scale)))


def packet_copy_frequencies_khz(base_freq_khz, mitigation, trial_rng):
    if mitigation.redundancy_copies <= 1:
        return (base_freq_khz,)

    if mitigation.hopping_enabled:
        if math.isclose(base_freq_khz, OPERATOR_TO_DRONE_FREQ_KHZ, rel_tol=0.0, abs_tol=1e-9):
            pair_idx = int(trial_rng.integers(len(OPERATOR_HOP_PAIRS_KHZ)))
            return OPERATOR_HOP_PAIRS_KHZ[pair_idx]
        if math.isclose(base_freq_khz, DRONE_TO_DRONE_FREQ_KHZ, rel_tol=0.0, abs_tol=1e-9):
            pair_idx = int(trial_rng.integers(len(DRONE_HOP_PAIRS_KHZ)))
            return DRONE_HOP_PAIRS_KHZ[pair_idx]

    if math.isclose(base_freq_khz, OPERATOR_TO_DRONE_FREQ_KHZ, rel_tol=0.0, abs_tol=1e-9):
        return OPERATOR_DIVERSITY_FREQS_KHZ
    if math.isclose(base_freq_khz, DRONE_TO_DRONE_FREQ_KHZ, rel_tol=0.0, abs_tol=1e-9):
        return DRONE_DIVERSITY_FREQS_KHZ
    return tuple(base_freq_khz + offset for offset in FREQUENCY_DIVERSITY_OFFSETS_KHZ[: mitigation.redundancy_copies])


def sample_narrowband_notch(base_freq_khz, trial_rng):
    if trial_rng.random() >= NARROWBAND_FADE_PROB:
        return None, 0.0

    notch_center_khz = base_freq_khz + float(trial_rng.choice(np.array([-4.0, -2.0, 0.0, 2.0, 4.0])))
    notch_depth_db = trial_rng.uniform(NARROWBAND_FADE_DB_MIN, NARROWBAND_FADE_DB_MAX)
    return notch_center_khz, notch_depth_db


def narrowband_penalty_db(copy_freq_khz, notch_center_khz, notch_depth_db):
    if notch_center_khz is None or notch_depth_db <= 0.0:
        return 0.0

    normalized_offset = (copy_freq_khz - notch_center_khz) / max(NARROWBAND_FADE_WIDTH_KHZ, 1e-6)
    return notch_depth_db * math.exp(-0.5 * normalized_offset * normalized_offset)


def transmit_packet(tx_pos, rx_pos, tx_vel, rx_vel, packet_bytes, environment, mitigation, trial_rng, freq_khz, link_fade_state=None):
    packet_bytes = effective_packet_bytes(packet_bytes, mitigation)
    copy_freqs_khz = packet_copy_frequencies_khz(freq_khz, mitigation, trial_rng)
    best_snr_db = -1e9
    parallel_copies = len(copy_freqs_khz)
    common_shadow_db = max(0.0, trial_rng.normal(environment.comm_shadow_mean_db, SHADOWING_SIGMA_DB))
    transient_bias_db = sample_transient_fade_bias_db(link_fade_state, freq_khz, trial_rng) if link_fade_state is not None else 0.0
    notch_center_khz, notch_depth_db = sample_narrowband_notch(freq_khz, trial_rng)
    success_latencies = []
    last_latency_s = 0.0

    for copy_freq_khz in copy_freqs_khz:
        selective_fade_db = max(0.0, trial_rng.normal(0.0, 1.1))
        frequency_selective_db = narrowband_penalty_db(copy_freq_khz, notch_center_khz, notch_depth_db)
        fading_db = common_shadow_db + transient_bias_db + selective_fade_db + frequency_selective_db
        snr_db, range_m = link_snr_db(tx_pos, rx_pos, tx_vel, rx_vel, environment, fading_db, copy_freq_khz)
        per = packet_error_rate(snr_db, packet_bytes)
        data_rate_bps = data_rate_bps_for_freq(copy_freq_khz)
        latency_s = range_m / SOUND_SPEED_MPS + PREAMBLE_S + packet_bytes * 8.0 / data_rate_bps
        latency_s *= trial_rng.uniform(LATENCY_JITTER_MIN, LATENCY_JITTER_MAX)
        last_latency_s = latency_s
        best_snr_db = max(best_snr_db, snr_db)
        if trial_rng.random() >= per:
            success_latencies.append(latency_s)

    if success_latencies:
        return PacketResult(True, min(success_latencies), best_snr_db, parallel_copies)

    return PacketResult(False, last_latency_s, best_snr_db, parallel_copies)


def choose_ack_drone(positions):
    ranges = np.linalg.norm(positions - OPERATOR_POS, axis=1)
    return int(np.argmin(ranges))


def step_fleet(
    positions,
    velocities,
    belief_state,
    swarm_link_state,
    control_bias,
    current_vector,
    center_target,
    environment,
    mitigation,
    gossip_packet_bytes,
    trial_rng,
    station_keep=False,
):
    gossip_tx_count, gossip_success_count, gossip_latency_sum_s, gossip_bits_sum = update_swarm_beliefs(
        positions,
        velocities,
        belief_state,
        environment,
        mitigation,
        gossip_packet_bytes,
        trial_rng,
        swarm_link_state,
    )
    targets = center_target + FORMATION_OFFSETS
    target_errors = targets - positions
    target_distances = np.linalg.norm(target_errors, axis=1)
    if station_keep:
        nominal_speeds = np.where(target_distances < WAYPOINT_HIT_RADIUS_M, 0.25, CRUISE_SPEED_MPS)
    else:
        nominal_speeds = np.full(NUM_DRONES, CRUISE_SPEED_MPS)

    desired_velocities = safe_unit_rows(target_errors) * nominal_speeds[:, None] + current_vector + control_bias

    prediction_age_s = np.minimum(belief_state.peer_age_s, DEAD_RECKONING_HORIZON_S)
    predicted_peer_positions = belief_state.perceived_positions + belief_state.perceived_velocities * prediction_age_s[:, :, None]
    pairwise_delta = positions[:, None, :] - predicted_peer_positions[:, :, :]
    pairwise_distance = np.linalg.norm(pairwise_delta, axis=2)
    repulsion_mask = (pairwise_distance < SAFE_SEPARATION_M) & (pairwise_distance > 1e-6)
    closeness = np.where(repulsion_mask, (SAFE_SEPARATION_M - pairwise_distance) / SAFE_SEPARATION_M, 0.0)
    emergency_margin = np.where(repulsion_mask, np.maximum(pairwise_distance - COLLISION_RADIUS_M, 0.35), 1.0)
    repulsion_gain = np.where(repulsion_mask, AVOIDANCE_GAIN * (closeness + 1.0 / emergency_margin), 0.0)
    repulsion_dir = np.divide(
        pairwise_delta,
        np.maximum(pairwise_distance[:, :, None], 1e-9),
        out=np.zeros_like(pairwise_delta),
        where=pairwise_distance[:, :, None] > 1e-9,
    )
    desired_velocities += np.sum(repulsion_dir * repulsion_gain[:, :, None], axis=1)

    # Last-ditch close-range sensing approximates the short-range collision sensor.
    true_pairwise_delta = positions[:, None, :] - positions[None, :, :]
    true_pairwise_distance = np.linalg.norm(true_pairwise_delta, axis=2)
    local_mask = (true_pairwise_distance < LOCAL_PROXIMITY_THRESHOLD_M) & (true_pairwise_distance > 1e-6)
    local_closeness = np.where(
        local_mask,
        (LOCAL_PROXIMITY_THRESHOLD_M - true_pairwise_distance) / max(LOCAL_PROXIMITY_THRESHOLD_M - COLLISION_RADIUS_M, 1e-6),
        0.0,
    )
    local_margin = np.where(local_mask, np.maximum(true_pairwise_distance - COLLISION_RADIUS_M, 0.20), 1.0)
    local_repulsion_gain = np.where(local_mask, LOCAL_PROXIMITY_GAIN * (local_closeness + 1.0 / local_margin), 0.0)
    local_repulsion_dir = np.divide(
        true_pairwise_delta,
        np.maximum(true_pairwise_distance[:, :, None], 1e-9),
        out=np.zeros_like(true_pairwise_delta),
        where=true_pairwise_distance[:, :, None] > 1e-9,
    )
    desired_velocities += np.sum(local_repulsion_dir * local_repulsion_gain[:, :, None], axis=1)

    velocity_error = desired_velocities - velocities
    error_magnitudes = np.linalg.norm(velocity_error, axis=1)
    delta_v = safe_unit_rows(velocity_error) * np.minimum(error_magnitudes, MAX_ACCEL_MPS2 * DT)[:, None]
    new_velocities = velocities + delta_v

    motion_noise = trial_rng.normal(0.0, MOTION_NOISE_STD_M, size=(NUM_DRONES, 3))
    motion_noise[:, 2] *= 0.3
    new_positions = positions + new_velocities * DT + motion_noise
    diag_idx = np.arange(NUM_DRONES)
    belief_state.perceived_positions[diag_idx, diag_idx] = new_positions
    belief_state.perceived_velocities[diag_idx, diag_idx] = new_velocities
    belief_state.peer_age_s[diag_idx, diag_idx] = 0.0
    return (
        new_positions,
        new_velocities,
        gossip_tx_count,
        gossip_success_count,
        gossip_latency_sum_s,
        gossip_bits_sum,
        mean_peer_age(belief_state.peer_age_s),
    )


def simulate_hold(
    positions,
    velocities,
    belief_state,
    swarm_link_state,
    control_bias,
    current_vector,
    center_target,
    environment,
    mitigation,
    gossip_packet_bytes,
    duration_s,
    trial_rng,
    sample_store,
    min_sep,
):
    steps = max(1, int(math.ceil(duration_s / DT)))
    collision_time = None
    gossip_tx_count = 0
    gossip_success_count = 0
    gossip_latency_sum_s = 0.0
    gossip_bits_sum = 0.0
    peer_age_sum = 0.0
    for step_idx in range(steps):
        positions, velocities, step_tx_count, step_success_count, step_gossip_latency_s, step_gossip_bits, step_peer_age = step_fleet(
            positions,
            velocities,
            belief_state,
            swarm_link_state,
            control_bias,
            current_vector,
            center_target,
            environment,
            mitigation,
            gossip_packet_bytes,
            trial_rng,
            station_keep=True,
        )
        gossip_tx_count += step_tx_count
        gossip_success_count += step_success_count
        gossip_latency_sum_s += step_gossip_latency_s
        gossip_bits_sum += step_gossip_bits
        peer_age_sum += step_peer_age
        min_sep = min(min_sep, pairwise_min_separation(positions))
        if min_sep < COLLISION_RADIUS_M and collision_time is None:
            collision_time = 0.0
        if sample_store is not None and step_idx % TRACK_SAMPLE_EVERY_STEPS == 0:
            sample_store.append(positions[:, :2].copy())
    return (
        positions,
        velocities,
        min_sep,
        collision_time,
        steps * DT,
        gossip_tx_count,
        gossip_success_count,
        gossip_latency_sum_s,
        gossip_bits_sum,
        peer_age_sum,
        steps,
    )


def simulate_to_waypoint(
    positions,
    velocities,
    belief_state,
    swarm_link_state,
    control_bias,
    current_vector,
    center_target,
    environment,
    mitigation,
    gossip_packet_bytes,
    time_budget_s,
    trial_rng,
    sample_store,
    min_sep,
):
    elapsed = 0.0
    collision_time = None
    arrived = False
    gossip_tx_count = 0
    gossip_success_count = 0
    gossip_latency_sum_s = 0.0
    gossip_bits_sum = 0.0
    peer_age_sum = 0.0

    step_idx = 0
    while elapsed < time_budget_s:
        positions, velocities, step_tx_count, step_success_count, step_gossip_latency_s, step_gossip_bits, step_peer_age = step_fleet(
            positions,
            velocities,
            belief_state,
            swarm_link_state,
            control_bias,
            current_vector,
            center_target,
            environment,
            mitigation,
            gossip_packet_bytes,
            trial_rng,
            station_keep=False,
        )
        gossip_tx_count += step_tx_count
        gossip_success_count += step_success_count
        gossip_latency_sum_s += step_gossip_latency_s
        gossip_bits_sum += step_gossip_bits
        peer_age_sum += step_peer_age
        if sample_store is not None and step_idx % TRACK_SAMPLE_EVERY_STEPS == 0:
            sample_store.append(positions[:, :2].copy())
        elapsed += DT
        step_idx += 1
        min_sep = min(min_sep, pairwise_min_separation(positions))
        if min_sep < COLLISION_RADIUS_M and collision_time is None:
            collision_time = elapsed
        targets = center_target + FORMATION_OFFSETS
        if np.all(np.linalg.norm(positions - targets, axis=1) <= WAYPOINT_HIT_RADIUS_M):
            arrived = True
            break

    return (
        positions,
        velocities,
        arrived,
        min_sep,
        collision_time,
        elapsed,
        gossip_tx_count,
        gossip_success_count,
        gossip_latency_sum_s,
        gossip_bits_sum,
        peer_age_sum,
        max(step_idx, 1),
    )


def perform_command_handshake(positions, velocities, environment, mitigation, packet_profile, trial_rng, operator_link_state):
    leader_idx = choose_ack_drone(positions)
    leader_pos = positions[leader_idx]
    leader_vel = velocities[leader_idx]

    total_latency_s = 0.0
    total_tx_count = 0
    total_success_count = 0
    snr_samples = []

    for attempt in range(MAX_HANDSHAKE_ATTEMPTS):
        syn = transmit_packet(
            OPERATOR_POS,
            leader_pos,
            np.zeros(3),
            leader_vel,
            packet_profile.packet_bytes,
            environment,
            mitigation,
            trial_rng,
            OPERATOR_TO_DRONE_FREQ_KHZ,
            operator_link_state,
        )
        total_tx_count += 1
        total_latency_s += syn.latency_s
        snr_samples.append(syn.snr_db)
        advance_transient_fade_state(operator_link_state, syn.latency_s)
        if syn.success:
            total_success_count += 1
            ack = transmit_packet(
                leader_pos,
                OPERATOR_POS,
                leader_vel,
                np.zeros(3),
                packet_profile.packet_bytes,
                environment,
                mitigation,
                trial_rng,
                OPERATOR_TO_DRONE_FREQ_KHZ,
                operator_link_state,
            )
            total_tx_count += 1
            total_latency_s += ack.latency_s
            snr_samples.append(ack.snr_db)
            advance_transient_fade_state(operator_link_state, ack.latency_s)
            if ack.success:
                total_success_count += 1
                return HandshakeResult(True, total_latency_s, total_tx_count, total_success_count, attempt, snr_samples, leader_idx)

        if attempt < MAX_HANDSHAKE_ATTEMPTS - 1:
            total_latency_s += HANDSHAKE_BACKOFF_S
            advance_transient_fade_state(operator_link_state, HANDSHAKE_BACKOFF_S)

    return HandshakeResult(False, total_latency_s, total_tx_count, total_success_count, MAX_HANDSHAKE_ATTEMPTS - 1, snr_samples, leader_idx)


def perform_arrival_handshake(positions, velocities, environment, mitigation, packet_profile, trial_rng, operator_link_state):
    leader_idx = choose_ack_drone(positions)
    leader_pos = positions[leader_idx]
    leader_vel = velocities[leader_idx]

    total_latency_s = 0.0
    total_tx_count = 0
    total_success_count = 0
    snr_samples = []

    for attempt in range(MAX_HANDSHAKE_ATTEMPTS):
        syn = transmit_packet(
            leader_pos,
            OPERATOR_POS,
            leader_vel,
            np.zeros(3),
            packet_profile.packet_bytes,
            environment,
            mitigation,
            trial_rng,
            OPERATOR_TO_DRONE_FREQ_KHZ,
            operator_link_state,
        )
        total_tx_count += 1
        total_latency_s += syn.latency_s
        snr_samples.append(syn.snr_db)
        advance_transient_fade_state(operator_link_state, syn.latency_s)
        if syn.success:
            total_success_count += 1
            ack = transmit_packet(
                OPERATOR_POS,
                leader_pos,
                np.zeros(3),
                leader_vel,
                packet_profile.packet_bytes,
                environment,
                mitigation,
                trial_rng,
                OPERATOR_TO_DRONE_FREQ_KHZ,
                operator_link_state,
            )
            total_tx_count += 1
            total_latency_s += ack.latency_s
            snr_samples.append(ack.snr_db)
            advance_transient_fade_state(operator_link_state, ack.latency_s)
            if ack.success:
                total_success_count += 1
                return HandshakeResult(True, total_latency_s, total_tx_count, total_success_count, attempt, snr_samples, leader_idx)

        if attempt < MAX_HANDSHAKE_ATTEMPTS - 1:
            total_latency_s += HANDSHAKE_BACKOFF_S
            advance_transient_fade_state(operator_link_state, HANDSHAKE_BACKOFF_S)

    return HandshakeResult(False, total_latency_s, total_tx_count, total_success_count, MAX_HANDSHAKE_ATTEMPTS - 1, snr_samples, leader_idx)


def run_collision_trial(trial_seed, mitigation, packet_profile, store_track=True):
    trial_rng = np.random.default_rng(trial_seed)
    positions, velocities, control_bias = init_fleet(trial_rng, start_center=COLLISION_START_CENTER)
    belief_state = init_swarm_belief_state(positions, velocities)
    swarm_link_state = TransientFadeState()
    current_vector = sample_current(COLLISION_ENVIRONMENT, trial_rng)
    target_idx = 1
    min_sep = float("inf")
    collision_time_s = None
    track_samples = [positions[:, :2].copy()] if store_track else None
    gossip_tx_count = 0
    gossip_success_count = 0
    gossip_latency_sum_s = 0.0
    gossip_bits_sum = 0.0
    peer_age_sum = 0.0

    total_steps = int(COLLISION_SIM_DURATION_S / DT)
    for step in range(total_steps):
        center_target = COLLISION_PATROL_CENTERS[target_idx]
        targets = COLLISION_PATROL_TARGETS[target_idx]
        positions, velocities, step_tx_count, step_success_count, step_gossip_latency_s, step_gossip_bits, step_peer_age = step_fleet(
            positions,
            velocities,
            belief_state,
            swarm_link_state,
            control_bias,
            current_vector,
            center_target,
            COLLISION_ENVIRONMENT,
            mitigation,
            packet_profile.packet_bytes,
            trial_rng,
            station_keep=False,
        )
        gossip_tx_count += step_tx_count
        gossip_success_count += step_success_count
        gossip_latency_sum_s += step_gossip_latency_s
        gossip_bits_sum += step_gossip_bits
        peer_age_sum += step_peer_age
        if track_samples is not None and step % TRACK_SAMPLE_EVERY_STEPS == 0:
            track_samples.append(positions[:, :2].copy())

        min_sep = min(min_sep, pairwise_min_separation(positions))
        if min_sep < COLLISION_RADIUS_M and collision_time_s is None:
            collision_time_s = step * DT

        if np.all(np.linalg.norm(positions - targets, axis=1) <= WAYPOINT_HIT_RADIUS_M):
            target_idx = (target_idx + 1) % len(COLLISION_PATROL_CENTERS)

    return CollisionTrialResult(
        trial_seed=trial_seed,
        collision_free=collision_time_s is None,
        min_separation_m=min_sep,
        collision_time_s=collision_time_s,
        gossip_delivery_rate=gossip_success_count / max(gossip_tx_count, 1),
        mean_peer_age_s=peer_age_sum / max(total_steps, 1),
        mean_gossip_latency_ms=(gossip_latency_sum_s / max(gossip_success_count, 1)) * MS_PER_SECOND,
        mean_gossip_bitrate_bps=gossip_bits_sum / max(COLLISION_SIM_DURATION_S, DT),
        track_samples=np.array(track_samples) if track_samples is not None else np.empty((0, NUM_DRONES, 2)),
    )


def run_mission_trial(trial_seed, mitigation, packet_profile, store_track=True):
    trial_rng = np.random.default_rng(trial_seed)
    positions, velocities, control_bias = init_fleet(trial_rng)
    belief_state = init_swarm_belief_state(positions, velocities)
    swarm_link_state = TransientFadeState()
    operator_link_state = TransientFadeState()
    mission_environment = AcousticEnvironment(
        name=MISSION_ENVIRONMENT.name,
        water_depth_m=MISSION_ENVIRONMENT.water_depth_m,
        wind_speed_mps=MISSION_ENVIRONMENT.wind_speed_mps,
        shipping_activity=MISSION_ENVIRONMENT.shipping_activity,
        current_mean_mps=MISSION_ENVIRONMENT.current_mean_mps,
        current_sigma_mps=MISSION_ENVIRONMENT.current_sigma_mps,
        comm_shadow_mean_db=float(trial_rng.uniform(MISSION_SHADOW_MEAN_DB_MIN, MISSION_SHADOW_MEAN_DB_MAX)),
    )
    current_vector = sample_current(mission_environment, trial_rng)
    current_center = PATROL_START_CENTER

    total_time_s = 0.0
    min_sep = float("inf")
    collision_free = True
    track_samples = [positions[:, :2].copy()] if store_track else None
    chosen_leaders = []

    command_tx = 0
    command_successes = 0
    arrival_tx = 0
    arrival_successes = 0
    command_latencies_ms = []
    arrival_latencies_ms = []
    retries_per_waypoint = []
    snr_samples = []
    gossip_tx_count = 0
    gossip_success_count = 0
    gossip_latency_sum_s = 0.0
    gossip_bits_sum = 0.0
    peer_age_sum = 0.0
    peer_age_steps = 0

    completed_waypoints = 0
    previous_arrival_confirmed = True

    for waypoint_center in MISSION_WAYPOINT_CENTERS:
        retries_per_waypoint.append(0)
        command_wait_s = 0.0 if completed_waypoints == 0 or previous_arrival_confirmed else COMMAND_REQUEST_INTERVAL_S
        command_acquired = False

        while total_time_s < MISSION_TIMEOUT_S:
            if command_wait_s > 0.0:
                (
                    positions,
                    velocities,
                    min_sep,
                    collision_during_hold,
                    hold_elapsed,
                    hold_tx_count,
                    hold_success_count,
                    hold_gossip_latency_sum_s,
                    hold_gossip_bits_sum,
                    hold_peer_age_sum,
                    hold_steps,
                ) = simulate_hold(
                    positions,
                    velocities,
                    belief_state,
                    swarm_link_state,
                    control_bias,
                    current_vector,
                    current_center,
                    mission_environment,
                    mitigation,
                    packet_profile.packet_bytes,
                    min(command_wait_s, MISSION_TIMEOUT_S - total_time_s),
                    trial_rng,
                    track_samples,
                    min_sep,
                )
                gossip_tx_count += hold_tx_count
                gossip_success_count += hold_success_count
                gossip_latency_sum_s += hold_gossip_latency_sum_s
                gossip_bits_sum += hold_gossip_bits_sum
                peer_age_sum += hold_peer_age_sum
                peer_age_steps += hold_steps
                total_time_s += hold_elapsed
                advance_transient_fade_state(operator_link_state, hold_elapsed)
                if collision_during_hold is not None:
                    collision_free = False
                if total_time_s >= MISSION_TIMEOUT_S:
                    break

            handshake = perform_command_handshake(
                positions,
                velocities,
                mission_environment,
                mitigation,
                packet_profile,
                trial_rng,
                operator_link_state,
            )
            chosen_leaders.append(handshake.leader_idx)
            retries_per_waypoint[-1] += handshake.retries
            command_tx += handshake.packet_tx_count
            command_successes += handshake.packet_success_count
            command_latencies_ms.append(handshake.latency_s * MS_PER_SECOND)
            snr_samples.extend(handshake.snr_samples)

            (
                positions,
                velocities,
                min_sep,
                collision_during_hold,
                hold_elapsed,
                hold_tx_count,
                hold_success_count,
                hold_gossip_latency_sum_s,
                hold_gossip_bits_sum,
                hold_peer_age_sum,
                hold_steps,
            ) = simulate_hold(
                positions,
                velocities,
                belief_state,
                swarm_link_state,
                control_bias,
                current_vector,
                current_center,
                mission_environment,
                mitigation,
                packet_profile.packet_bytes,
                handshake.latency_s,
                trial_rng,
                track_samples,
                min_sep,
            )
            gossip_tx_count += hold_tx_count
            gossip_success_count += hold_success_count
            gossip_latency_sum_s += hold_gossip_latency_sum_s
            gossip_bits_sum += hold_gossip_bits_sum
            peer_age_sum += hold_peer_age_sum
            peer_age_steps += hold_steps
            total_time_s += hold_elapsed
            if collision_during_hold is not None:
                collision_free = False

            if handshake.success:
                command_acquired = True
                break

            command_wait_s = COMMAND_REQUEST_INTERVAL_S

        if not command_acquired or total_time_s >= MISSION_TIMEOUT_S:
            break

        (
            positions,
            velocities,
            arrived,
            min_sep,
            collision_during_travel,
            travel_elapsed,
            travel_tx_count,
            travel_success_count,
            travel_gossip_latency_sum_s,
            travel_gossip_bits_sum,
            travel_peer_age_sum,
            travel_steps,
        ) = simulate_to_waypoint(
            positions,
            velocities,
            belief_state,
            swarm_link_state,
            control_bias,
            current_vector,
            waypoint_center,
            mission_environment,
            mitigation,
            packet_profile.packet_bytes,
            MISSION_TIMEOUT_S - total_time_s,
            trial_rng,
            track_samples,
            min_sep,
        )
        gossip_tx_count += travel_tx_count
        gossip_success_count += travel_success_count
        gossip_latency_sum_s += travel_gossip_latency_sum_s
        gossip_bits_sum += travel_gossip_bits_sum
        peer_age_sum += travel_peer_age_sum
        peer_age_steps += travel_steps
        total_time_s += travel_elapsed
        advance_transient_fade_state(operator_link_state, travel_elapsed)
        if collision_during_travel is not None:
            collision_free = False

        if not arrived or total_time_s >= MISSION_TIMEOUT_S:
            break

        current_center = waypoint_center
        completed_waypoints += 1
        arrival = perform_arrival_handshake(
            positions,
            velocities,
            mission_environment,
            mitigation,
            packet_profile,
            trial_rng,
            operator_link_state,
        )
        chosen_leaders.append(arrival.leader_idx)
        retries_per_waypoint[-1] += arrival.retries
        arrival_tx += arrival.packet_tx_count
        arrival_successes += arrival.packet_success_count
        arrival_latencies_ms.append(arrival.latency_s * MS_PER_SECOND)
        snr_samples.extend(arrival.snr_samples)

        (
            positions,
            velocities,
            min_sep,
            collision_during_hold,
            hold_elapsed,
            hold_tx_count,
            hold_success_count,
            hold_gossip_latency_sum_s,
            hold_gossip_bits_sum,
            hold_peer_age_sum,
            hold_steps,
        ) = simulate_hold(
            positions,
            velocities,
            belief_state,
            swarm_link_state,
            control_bias,
            current_vector,
            current_center,
            mission_environment,
            mitigation,
            packet_profile.packet_bytes,
            arrival.latency_s,
            trial_rng,
            track_samples,
            min_sep,
        )
        gossip_tx_count += hold_tx_count
        gossip_success_count += hold_success_count
        gossip_latency_sum_s += hold_gossip_latency_sum_s
        gossip_bits_sum += hold_gossip_bits_sum
        peer_age_sum += hold_peer_age_sum
        peer_age_steps += hold_steps
        total_time_s += hold_elapsed
        if collision_during_hold is not None:
            collision_free = False

        previous_arrival_confirmed = arrival.success

    success = completed_waypoints == len(MISSION_WAYPOINT_CENTERS) and collision_free
    command_loss_rate = 1.0 - (command_successes / max(command_tx, 1))
    arrival_loss_rate = 1.0 - (arrival_successes / max(arrival_tx, 1))

    return MissionTrialResult(
        trial_seed=trial_seed,
        success=success,
        collision_free=collision_free,
        completed_waypoints=completed_waypoints,
        total_time_s=total_time_s,
        min_separation_m=min_sep,
        command_loss_rate=command_loss_rate,
        arrival_loss_rate=arrival_loss_rate,
        mean_command_latency_ms=float(np.mean(command_latencies_ms)) if command_latencies_ms else 0.0,
        mean_arrival_latency_ms=float(np.mean(arrival_latencies_ms)) if arrival_latencies_ms else 0.0,
        mean_retries_per_waypoint=float(np.mean(retries_per_waypoint)) if retries_per_waypoint else 0.0,
        mean_snr_db=float(np.mean(snr_samples)) if snr_samples else 0.0,
        gossip_delivery_rate=gossip_success_count / max(gossip_tx_count, 1),
        mean_peer_age_s=peer_age_sum / max(peer_age_steps, 1),
        mean_gossip_latency_ms=(gossip_latency_sum_s / max(gossip_success_count, 1)) * MS_PER_SECOND,
        mean_gossip_bitrate_bps=gossip_bits_sum / max(total_time_s, DT),
        track_samples=np.array(track_samples) if track_samples is not None else np.empty((0, NUM_DRONES, 2)),
        chosen_leaders=chosen_leaders,
    )


def collision_trial_task(task):
    trial_seed, mitigation, packet_profile, store_track = task
    return run_collision_trial(int(trial_seed), mitigation, packet_profile, store_track=store_track)


def mission_trial_task(task):
    trial_seed, mitigation, packet_profile, store_track = task
    return run_mission_trial(int(trial_seed), mitigation, packet_profile, store_track=store_track)


def resolve_worker_count(requested_workers, task_count):
    if requested_workers is not None:
        return max(1, min(int(requested_workers), task_count))
    return max(1, min(os.cpu_count() or 1, task_count))


def parallel_trial_map(task_func, tasks, worker_count):
    if worker_count <= 1 or len(tasks) <= 1:
        return [task_func(task) for task in tasks]

    try:
        if os.name == "nt":
            mp_context = None
        else:
            mp_context = mp.get_context("fork")

        chunk_size = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            return list(executor.map(task_func, tasks, chunksize=chunk_size))
    except (OSError, PermissionError):
        return [task_func(task) for task in tasks]


def run_collision_monte_carlo(trial_count, worker_count):
    summary = {}
    seeds = rng.integers(0, 2**31 - 1, size=trial_count)
    for packet_profile in PACKET_PROFILES:
        summary[packet_profile.name] = {
            "trials": trial_count,
            "packet_bytes": packet_profile.packet_bytes,
            "encrypted": packet_profile.encrypted,
            "results": {},
        }
        for mitigation in [BASELINE, MITIGATED]:
            tasks = [(int(seed), mitigation, packet_profile, False) for seed in seeds]
            results = parallel_trial_map(collision_trial_task, tasks, resolve_worker_count(worker_count, len(tasks)))
            best_seed = max(results, key=lambda result: result.min_separation_m).trial_seed
            best_example = run_collision_trial(best_seed, mitigation, packet_profile, store_track=True)
            collision_flags = [result.collision_free for result in results]
            min_separations = [result.min_separation_m for result in results]
            gossip_delivery_rates = [result.gossip_delivery_rate for result in results]
            gossip_loss_rates = [1.0 - result.gossip_delivery_rate for result in results]
            peer_ages = [result.mean_peer_age_s for result in results]
            gossip_latencies_ms = [result.mean_gossip_latency_ms for result in results]
            gossip_bitrate_bps = [result.mean_gossip_bitrate_bps for result in results]
            collision_ci_low, collision_ci_high = binomial_confidence_interval(collision_flags)
            mean_min_sep, p10_min_sep, p90_min_sep = metric_summary(min_separations)
            mean_gossip_loss_rate, p10_gossip_loss_rate, p90_gossip_loss_rate = metric_summary(gossip_loss_rates)
            mean_peer_age_s, p10_peer_age_s, p90_peer_age_s = metric_summary(peer_ages)
            mean_gossip_latency_ms, p10_gossip_latency_ms, p90_gossip_latency_ms = metric_summary(gossip_latencies_ms)
            mean_gossip_bitrate_bps, p10_gossip_bitrate_bps, p90_gossip_bitrate_bps = metric_summary(gossip_bitrate_bps)
            summary[packet_profile.name]["results"][mitigation.name] = {
                "collision_free_rate": float(np.mean(collision_flags)),
                "collision_free_ci_low": collision_ci_low,
                "collision_free_ci_high": collision_ci_high,
                "mean_min_separation_m": mean_min_sep,
                "p05_min_separation_m": float(np.percentile(min_separations, 5)),
                "p10_min_separation_m": p10_min_sep,
                "p90_min_separation_m": p90_min_sep,
                "mean_gossip_delivery_rate": float(np.mean(gossip_delivery_rates)),
                "mean_gossip_loss_rate": mean_gossip_loss_rate,
                "p10_gossip_loss_rate": p10_gossip_loss_rate,
                "p90_gossip_loss_rate": p90_gossip_loss_rate,
                "mean_peer_age_s": mean_peer_age_s,
                "p10_peer_age_s": p10_peer_age_s,
                "p90_peer_age_s": p90_peer_age_s,
                "mean_gossip_latency_ms": mean_gossip_latency_ms,
                "p10_gossip_latency_ms": p10_gossip_latency_ms,
                "p90_gossip_latency_ms": p90_gossip_latency_ms,
                "mean_gossip_bitrate_bps": mean_gossip_bitrate_bps,
                "p10_gossip_bitrate_bps": p10_gossip_bitrate_bps,
                "p90_gossip_bitrate_bps": p90_gossip_bitrate_bps,
                "example": best_example,
                "results": results,
            }
    return summary


def run_operator_monte_carlo(trial_count, worker_count):
    summary = {}
    for packet_profile in PACKET_PROFILES:
        seeds = rng.integers(0, 2**31 - 1, size=trial_count)
        summary[packet_profile.name] = {
            "trials": trial_count,
            "packet_bytes": packet_profile.packet_bytes,
            "encrypted": packet_profile.encrypted,
            "results": {},
        }
        for mitigation in [BASELINE, MITIGATED]:
            tasks = [(int(seed), mitigation, packet_profile, False) for seed in seeds]
            results = parallel_trial_map(mission_trial_task, tasks, resolve_worker_count(worker_count, len(tasks)))
            successful_results = [result for result in results if result.success]
            example_seed = (
                successful_results[0].trial_seed if successful_results else max(results, key=lambda result: result.completed_waypoints).trial_seed
            )
            example = run_mission_trial(example_seed, mitigation, packet_profile, store_track=True)
            success_flags = [result.success for result in results]
            collision_flags = [result.collision_free for result in results]
            success_ci_low, success_ci_high = binomial_confidence_interval(success_flags)
            collision_ci_low, collision_ci_high = binomial_confidence_interval(collision_flags)
            mean_completed, p10_completed, p90_completed = metric_summary([result.completed_waypoints for result in results])
            mean_total_time, p10_total_time, p90_total_time = metric_summary([result.total_time_s for result in results])
            mean_min_sep, p10_min_sep, p90_min_sep = metric_summary([result.min_separation_m for result in results])
            mean_cmd_loss, p10_cmd_loss, p90_cmd_loss = metric_summary([result.command_loss_rate for result in results])
            mean_arrival_loss, p10_arrival_loss, p90_arrival_loss = metric_summary([result.arrival_loss_rate for result in results])
            mean_cmd_latency, p10_cmd_latency, p90_cmd_latency = metric_summary([result.mean_command_latency_ms for result in results])
            mean_arrival_latency, p10_arrival_latency, p90_arrival_latency = metric_summary([result.mean_arrival_latency_ms for result in results])
            mean_retries, p10_retries, p90_retries = metric_summary([result.mean_retries_per_waypoint for result in results])
            mean_snr, p10_snr, p90_snr = metric_summary([result.mean_snr_db for result in results])
            mean_gossip_delivery = float(np.mean([result.gossip_delivery_rate for result in results]))
            mean_gossip_loss, p10_gossip_loss, p90_gossip_loss = metric_summary([1.0 - result.gossip_delivery_rate for result in results])
            mean_peer_age_s, p10_peer_age_s, p90_peer_age_s = metric_summary([result.mean_peer_age_s for result in results])
            mean_gossip_latency_ms, p10_gossip_latency_ms, p90_gossip_latency_ms = metric_summary(
                [result.mean_gossip_latency_ms for result in results]
            )
            mean_gossip_bitrate_bps, p10_gossip_bitrate_bps, p90_gossip_bitrate_bps = metric_summary(
                [result.mean_gossip_bitrate_bps for result in results]
            )
            summary[packet_profile.name]["results"][mitigation.name] = {
                "mission_success_rate": float(np.mean(success_flags)),
                "mission_success_ci_low": success_ci_low,
                "mission_success_ci_high": success_ci_high,
                "collision_free_rate": float(np.mean(collision_flags)),
                "collision_free_ci_low": collision_ci_low,
                "collision_free_ci_high": collision_ci_high,
                "mean_completed_waypoints": mean_completed,
                "p10_completed_waypoints": p10_completed,
                "p90_completed_waypoints": p90_completed,
                "mean_total_time_s": mean_total_time,
                "p10_total_time_s": p10_total_time,
                "p90_total_time_s": p90_total_time,
                "mean_min_separation_m": mean_min_sep,
                "p10_min_separation_m": p10_min_sep,
                "p90_min_separation_m": p90_min_sep,
                "mean_command_loss_rate": mean_cmd_loss,
                "p10_command_loss_rate": p10_cmd_loss,
                "p90_command_loss_rate": p90_cmd_loss,
                "mean_arrival_loss_rate": mean_arrival_loss,
                "p10_arrival_loss_rate": p10_arrival_loss,
                "p90_arrival_loss_rate": p90_arrival_loss,
                "mean_command_latency_ms": mean_cmd_latency,
                "p10_command_latency_ms": p10_cmd_latency,
                "p90_command_latency_ms": p90_cmd_latency,
                "mean_arrival_latency_ms": mean_arrival_latency,
                "p10_arrival_latency_ms": p10_arrival_latency,
                "p90_arrival_latency_ms": p90_arrival_latency,
                "mean_retries_per_waypoint": mean_retries,
                "p10_retries_per_waypoint": p10_retries,
                "p90_retries_per_waypoint": p90_retries,
                "mean_snr_db": mean_snr,
                "p10_snr_db": p10_snr,
                "p90_snr_db": p90_snr,
                "mean_gossip_delivery_rate": mean_gossip_delivery,
                "mean_gossip_loss_rate": mean_gossip_loss,
                "p10_gossip_loss_rate": p10_gossip_loss,
                "p90_gossip_loss_rate": p90_gossip_loss,
                "mean_peer_age_s": mean_peer_age_s,
                "p10_peer_age_s": p10_peer_age_s,
                "p90_peer_age_s": p90_peer_age_s,
                "mean_gossip_latency_ms": mean_gossip_latency_ms,
                "p10_gossip_latency_ms": p10_gossip_latency_ms,
                "p90_gossip_latency_ms": p90_gossip_latency_ms,
                "mean_gossip_bitrate_bps": mean_gossip_bitrate_bps,
                "p10_gossip_bitrate_bps": p10_gossip_bitrate_bps,
                "p90_gossip_bitrate_bps": p90_gossip_bitrate_bps,
                "example": example,
            }
    return summary


def operator_range_bounds():
    all_positions = [PATROL_START_CENTER + FORMATION_OFFSETS]
    for center in MISSION_WAYPOINT_CENTERS:
        all_positions.append(center + FORMATION_OFFSETS)
    stacked = np.vstack(all_positions)
    ranges = np.linalg.norm(stacked - OPERATOR_POS, axis=1)
    return float(np.min(ranges)), float(np.max(ranges))


def range_bounds_for_centers(center_path):
    stacked = center_path[:, None, :] + FORMATION_OFFSETS[None, :, :]
    ranges = np.linalg.norm(stacked.reshape(-1, 3) - OPERATOR_POS, axis=1)
    return float(np.min(ranges)), float(np.max(ranges))


def percentile_bounds(values, low=10.0, high=90.0):
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, low)), float(np.percentile(array, high))


def binomial_confidence_interval(flags, z=1.96):
    n = len(flags)
    if n == 0:
        return 0.0, 0.0
    p_hat = float(np.mean(flags))
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))
    return clamp(center - margin, 0.0, 1.0), clamp(center + margin, 0.0, 1.0)


def metric_summary(values):
    mean_value = float(np.mean(values))
    p10_value, p90_value = percentile_bounds(values, low=10.0, high=90.0)
    return mean_value, p10_value, p90_value


def save_collision_csv(summary, output_path):
    with open(output_path, "w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "experiment",
                "packet_profile",
                "packet_bytes",
                "encrypted",
                "mitigation",
                "trials",
                "duration_s",
                "collision_threshold_m",
                "safe_separation_m",
                "cruise_speed_knots",
                "collision_free_rate",
                "collision_free_ci_low",
                "collision_free_ci_high",
                "mean_min_separation_m",
                "p05_min_separation_m",
                "p10_min_separation_m",
                "p90_min_separation_m",
                "mean_gossip_delivery_rate",
                "mean_gossip_loss_rate",
                "p10_gossip_loss_rate",
                "p90_gossip_loss_rate",
                "mean_peer_age_s",
                "p10_peer_age_s",
                "p90_peer_age_s",
                "mean_gossip_latency_ms",
                "p10_gossip_latency_ms",
                "p90_gossip_latency_ms",
                "mean_gossip_bitrate_bps",
                "p10_gossip_bitrate_bps",
                "p90_gossip_bitrate_bps",
                "operator_to_drone_freq_khz",
                "drone_to_drone_freq_khz",
            ],
        )
        writer.writeheader()
        for packet_name, packet_summary in summary.items():
            for mitigation_name, metrics in packet_summary["results"].items():
                writer.writerow(
                    {
                        "experiment": "collision_only_patrol",
                        "packet_profile": packet_name,
                        "packet_bytes": packet_summary["packet_bytes"],
                        "encrypted": packet_summary["encrypted"],
                        "mitigation": mitigation_name,
                        "trials": packet_summary["trials"],
                        "duration_s": COLLISION_SIM_DURATION_S,
                        "collision_threshold_m": COLLISION_RADIUS_M,
                        "safe_separation_m": SAFE_SEPARATION_M,
                        "cruise_speed_knots": CRUISE_SPEED_MPS / KNOT_TO_MPS,
                        "collision_free_rate": metrics["collision_free_rate"],
                        "collision_free_ci_low": metrics["collision_free_ci_low"],
                        "collision_free_ci_high": metrics["collision_free_ci_high"],
                        "mean_min_separation_m": metrics["mean_min_separation_m"],
                        "p05_min_separation_m": metrics["p05_min_separation_m"],
                        "p10_min_separation_m": metrics["p10_min_separation_m"],
                        "p90_min_separation_m": metrics["p90_min_separation_m"],
                        "mean_gossip_delivery_rate": metrics["mean_gossip_delivery_rate"],
                        "mean_gossip_loss_rate": metrics["mean_gossip_loss_rate"],
                        "p10_gossip_loss_rate": metrics["p10_gossip_loss_rate"],
                        "p90_gossip_loss_rate": metrics["p90_gossip_loss_rate"],
                        "mean_peer_age_s": metrics["mean_peer_age_s"],
                        "p10_peer_age_s": metrics["p10_peer_age_s"],
                        "p90_peer_age_s": metrics["p90_peer_age_s"],
                        "mean_gossip_latency_ms": metrics["mean_gossip_latency_ms"],
                        "p10_gossip_latency_ms": metrics["p10_gossip_latency_ms"],
                        "p90_gossip_latency_ms": metrics["p90_gossip_latency_ms"],
                        "mean_gossip_bitrate_bps": metrics["mean_gossip_bitrate_bps"],
                        "p10_gossip_bitrate_bps": metrics["p10_gossip_bitrate_bps"],
                        "p90_gossip_bitrate_bps": metrics["p90_gossip_bitrate_bps"],
                        "operator_to_drone_freq_khz": OPERATOR_TO_DRONE_FREQ_KHZ,
                        "drone_to_drone_freq_khz": DRONE_TO_DRONE_FREQ_KHZ,
                    }
                )


def save_mission_csv(summary, output_path):
    min_range_m, max_range_m = operator_range_bounds()
    with open(output_path, "w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "packet_profile",
                "packet_bytes",
                "encrypted",
                "mitigation",
                "trials",
                "mission_success_rate",
                "mission_success_ci_low",
                "mission_success_ci_high",
                "collision_free_rate",
                "collision_free_ci_low",
                "collision_free_ci_high",
                "mean_completed_waypoints",
                "p10_completed_waypoints",
                "p90_completed_waypoints",
                "mean_total_time_s",
                "p10_total_time_s",
                "p90_total_time_s",
                "mean_min_separation_m",
                "p10_min_separation_m",
                "p90_min_separation_m",
                "mean_command_loss_rate",
                "p10_command_loss_rate",
                "p90_command_loss_rate",
                "mean_arrival_loss_rate",
                "p10_arrival_loss_rate",
                "p90_arrival_loss_rate",
                "mean_command_latency_ms",
                "p10_command_latency_ms",
                "p90_command_latency_ms",
                "mean_arrival_latency_ms",
                "p10_arrival_latency_ms",
                "p90_arrival_latency_ms",
                "mean_retries_per_waypoint",
                "p10_retries_per_waypoint",
                "p90_retries_per_waypoint",
                "mean_snr_db",
                "p10_snr_db",
                "p90_snr_db",
                "mean_gossip_delivery_rate",
                "mean_gossip_loss_rate",
                "p10_gossip_loss_rate",
                "p90_gossip_loss_rate",
                "mean_peer_age_s",
                "p10_peer_age_s",
                "p90_peer_age_s",
                "mean_gossip_latency_ms",
                "p10_gossip_latency_ms",
                "p90_gossip_latency_ms",
                "mean_gossip_bitrate_bps",
                "p10_gossip_bitrate_bps",
                "p90_gossip_bitrate_bps",
                "operator_range_min_m",
                "operator_range_max_m",
                "operator_to_drone_freq_khz",
                "drone_to_drone_freq_khz",
                "mission_shadow_mean_db_min",
                "mission_shadow_mean_db_max",
            ],
        )
        writer.writeheader()
        for packet_name, packet_summary in summary.items():
            for mitigation_name, metrics in packet_summary["results"].items():
                writer.writerow(
                    {
                        "packet_profile": packet_name,
                        "packet_bytes": packet_summary["packet_bytes"],
                        "encrypted": packet_summary["encrypted"],
                        "mitigation": mitigation_name,
                        "trials": packet_summary["trials"],
                        "mission_success_rate": metrics["mission_success_rate"],
                        "mission_success_ci_low": metrics["mission_success_ci_low"],
                        "mission_success_ci_high": metrics["mission_success_ci_high"],
                        "collision_free_rate": metrics["collision_free_rate"],
                        "collision_free_ci_low": metrics["collision_free_ci_low"],
                        "collision_free_ci_high": metrics["collision_free_ci_high"],
                        "mean_completed_waypoints": metrics["mean_completed_waypoints"],
                        "p10_completed_waypoints": metrics["p10_completed_waypoints"],
                        "p90_completed_waypoints": metrics["p90_completed_waypoints"],
                        "mean_total_time_s": metrics["mean_total_time_s"],
                        "p10_total_time_s": metrics["p10_total_time_s"],
                        "p90_total_time_s": metrics["p90_total_time_s"],
                        "mean_min_separation_m": metrics["mean_min_separation_m"],
                        "p10_min_separation_m": metrics["p10_min_separation_m"],
                        "p90_min_separation_m": metrics["p90_min_separation_m"],
                        "mean_command_loss_rate": metrics["mean_command_loss_rate"],
                        "p10_command_loss_rate": metrics["p10_command_loss_rate"],
                        "p90_command_loss_rate": metrics["p90_command_loss_rate"],
                        "mean_arrival_loss_rate": metrics["mean_arrival_loss_rate"],
                        "p10_arrival_loss_rate": metrics["p10_arrival_loss_rate"],
                        "p90_arrival_loss_rate": metrics["p90_arrival_loss_rate"],
                        "mean_command_latency_ms": metrics["mean_command_latency_ms"],
                        "p10_command_latency_ms": metrics["p10_command_latency_ms"],
                        "p90_command_latency_ms": metrics["p90_command_latency_ms"],
                        "mean_arrival_latency_ms": metrics["mean_arrival_latency_ms"],
                        "p10_arrival_latency_ms": metrics["p10_arrival_latency_ms"],
                        "p90_arrival_latency_ms": metrics["p90_arrival_latency_ms"],
                        "mean_retries_per_waypoint": metrics["mean_retries_per_waypoint"],
                        "p10_retries_per_waypoint": metrics["p10_retries_per_waypoint"],
                        "p90_retries_per_waypoint": metrics["p90_retries_per_waypoint"],
                        "mean_snr_db": metrics["mean_snr_db"],
                        "p10_snr_db": metrics["p10_snr_db"],
                        "p90_snr_db": metrics["p90_snr_db"],
                        "mean_gossip_delivery_rate": metrics["mean_gossip_delivery_rate"],
                        "mean_gossip_loss_rate": metrics["mean_gossip_loss_rate"],
                        "p10_gossip_loss_rate": metrics["p10_gossip_loss_rate"],
                        "p90_gossip_loss_rate": metrics["p90_gossip_loss_rate"],
                        "mean_peer_age_s": metrics["mean_peer_age_s"],
                        "p10_peer_age_s": metrics["p10_peer_age_s"],
                        "p90_peer_age_s": metrics["p90_peer_age_s"],
                        "mean_gossip_latency_ms": metrics["mean_gossip_latency_ms"],
                        "p10_gossip_latency_ms": metrics["p10_gossip_latency_ms"],
                        "p90_gossip_latency_ms": metrics["p90_gossip_latency_ms"],
                        "mean_gossip_bitrate_bps": metrics["mean_gossip_bitrate_bps"],
                        "p10_gossip_bitrate_bps": metrics["p10_gossip_bitrate_bps"],
                        "p90_gossip_bitrate_bps": metrics["p90_gossip_bitrate_bps"],
                        "operator_range_min_m": min_range_m,
                        "operator_range_max_m": max_range_m,
                        "operator_to_drone_freq_khz": OPERATOR_TO_DRONE_FREQ_KHZ,
                        "drone_to_drone_freq_khz": DRONE_TO_DRONE_FREQ_KHZ,
                        "mission_shadow_mean_db_min": MISSION_SHADOW_MEAN_DB_MIN,
                        "mission_shadow_mean_db_max": MISSION_SHADOW_MEAN_DB_MAX,
                    }
                )


def save_collision_png(summary, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.84, wspace=0.24)

    packet_labels = list(summary.keys())
    mitigation_names = [BASELINE.name, MITIGATED.name]
    colors = {BASELINE.name: "#4A5568", MITIGATED.name: "#2B6CB0"}
    x = np.arange(len(packet_labels))
    width = 0.34

    def metric_values(metric_name, mitigation_name):
        return [summary[label]["results"][mitigation_name][metric_name] for label in packet_labels]

    def ci_error(rate_name, low_name, high_name, mitigation_name):
        means = np.array(metric_values(rate_name, mitigation_name))
        lows = np.array(metric_values(low_name, mitigation_name))
        highs = np.array(metric_values(high_name, mitigation_name))
        return np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])

    axes[0].bar(
        x - width / 2.0,
        metric_values("collision_free_rate", BASELINE.name),
        color=colors[BASELINE.name],
        width=width,
        yerr=ci_error("collision_free_rate", "collision_free_ci_low", "collision_free_ci_high", BASELINE.name),
        capsize=6,
        ecolor="#1A202C",
    )
    axes[0].bar(
        x + width / 2.0,
        metric_values("collision_free_rate", MITIGATED.name),
        color=colors[MITIGATED.name],
        width=width,
        yerr=ci_error("collision_free_rate", "collision_free_ci_low", "collision_free_ci_high", MITIGATED.name),
        capsize=6,
        ecolor="#1A202C",
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x, packet_labels, rotation=12)
    axes[0].set_ylabel("Fraction of trials")
    axes[0].set_title("Collision-Free Patrol Rate", pad=10)
    for packet_idx, packet_name in enumerate(packet_labels):
        for x_pos, mitigation_name in (
            (packet_idx - width / 2.0, BASELINE.name),
            (packet_idx + width / 2.0, MITIGATED.name),
        ):
            metrics = summary[packet_name]["results"][mitigation_name]
            axes[0].text(
                x_pos,
                min(1.02, metrics["collision_free_rate"] + 0.06),
                f"{metrics['collision_free_rate']:.2f}\n[{metrics['collision_free_ci_low']:.2f}, {metrics['collision_free_ci_high']:.2f}]",
                ha="center",
                va="bottom",
                fontsize=8.6,
            )

    all_min_seps = np.concatenate(
        [
            np.array([result.min_separation_m for result in summary[packet_name]["results"][mitigation_name]["results"]], dtype=float)
            for packet_name in packet_labels
            for mitigation_name in mitigation_names
        ]
    )
    plot_min = float(np.min(all_min_seps))
    plot_max = float(np.max(all_min_seps))
    span = max(plot_max - plot_min, 0.20)
    padding = 0.08 * span + 0.01
    view_min = plot_min - padding
    view_max = plot_max + padding
    axes[1].axvline(COLLISION_RADIUS_M, color="#1A202C", linewidth=1.6, linestyle=":")
    jitter_rng = np.random.default_rng(SEED + 101)
    row_specs = [
        ("32 B clear", BASELINE.name, 0.82),
        ("32 B clear", MITIGATED.name, 0.62),
        ("64 B encrypted", BASELINE.name, 0.38),
        ("64 B encrypted", MITIGATED.name, 0.18),
    ]
    summary_lines = []
    for packet_name, mitigation_name, y_pos in row_specs:
        metrics = summary[packet_name]["results"][mitigation_name]
        min_seps = np.array([result.min_separation_m for result in metrics["results"]], dtype=float)
        axes[1].plot(
            [metrics["p10_min_separation_m"], metrics["p90_min_separation_m"]],
            [y_pos, y_pos],
            color=colors[mitigation_name],
            linewidth=9,
            alpha=0.20,
            solid_capstyle="round",
            zorder=1,
        )
        axes[1].boxplot(
            min_seps,
            vert=False,
            positions=[y_pos],
            widths=0.12,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor=colors[mitigation_name], edgecolor=colors[mitigation_name], linewidth=1.2, alpha=0.20),
            whiskerprops=dict(color=colors[mitigation_name], linewidth=1.2),
            capprops=dict(color=colors[mitigation_name], linewidth=1.2),
            medianprops=dict(color="#1A202C", linewidth=1.4),
            flierprops=dict(marker="o", markersize=0),
        )
        y_jitter = y_pos + jitter_rng.uniform(-0.05, 0.05, size=len(min_seps))
        axes[1].scatter(
            min_seps,
            y_jitter,
            s=24,
            color=colors[mitigation_name],
            edgecolor="white",
            linewidth=0.40,
            alpha=0.82,
            zorder=3,
        )
        summary_lines.append(
            f"{packet_name}, {'Base' if mitigation_name == BASELINE.name else 'Mitigated'}: "
            f"{metrics['mean_min_separation_m']:.1f} m mean | {metrics['p05_min_separation_m']:.1f} m 5th pct"
        )
    axes[1].set_xlim(view_min, view_max)
    axes[1].set_ylim(0.08, 0.92)
    axes[1].set_yticks([0.82, 0.62, 0.38, 0.18])
    axes[1].set_yticklabels(
        ["32 B | No mitigation", "32 B | Hop + redundancy", "64 B | No mitigation", "64 B | Hop + redundancy"]
    )
    axes[1].set_xlabel("Minimum pairwise separation (m)")
    axes[1].set_title("Closest Approach Distribution (zoomed)", pad=10)
    axes[1].text(
        0.02,
        0.95,
        f"Threshold = {COLLISION_RADIUS_M:.0f} m\n"
        + "\n".join(summary_lines),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CBD5E0", alpha=0.95),
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[BASELINE.name]),
        plt.Rectangle((0, 0), 1, 1, color=colors[MITIGATED.name]),
    ]
    fig.legend(
        legend_handles,
        [BASELINE.name, MITIGATED.name],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle("Five-Drone Collision Monte Carlo Over Six Hours | 32 B clear vs 64 B encrypted", fontsize=16, y=0.975)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_mission_metrics_png(summary, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = list(summary.keys())
    x = np.arange(len(labels))
    width = 0.34
    baseline_color = "#4A5568"
    mitigated_color = "#2B6CB0"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.80, wspace=0.22, hspace=0.34)

    def metric_values(metric_name, mitigation_name):
        return [summary[label]["results"][mitigation_name][metric_name] for label in labels]

    def percentile_error(metric_name, low_name, high_name, mitigation_name):
        means = np.array(metric_values(metric_name, mitigation_name))
        lows = np.array(metric_values(low_name, mitigation_name))
        highs = np.array(metric_values(high_name, mitigation_name))
        return np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])

    def ci_error(rate_name, low_name, high_name, mitigation_name):
        means = np.array(metric_values(rate_name, mitigation_name))
        lows = np.array(metric_values(low_name, mitigation_name))
        highs = np.array(metric_values(high_name, mitigation_name))
        return np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])

    axes[0, 0].bar(
        x - width / 2.0,
        metric_values("mission_success_rate", BASELINE.name),
        width=width,
        color=baseline_color,
        yerr=ci_error("mission_success_rate", "mission_success_ci_low", "mission_success_ci_high", BASELINE.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 0].bar(
        x + width / 2.0,
        metric_values("mission_success_rate", MITIGATED.name),
        width=width,
        color=mitigated_color,
        yerr=ci_error("mission_success_rate", "mission_success_ci_low", "mission_success_ci_high", MITIGATED.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_xticks(x, labels, rotation=12)
    axes[0, 0].set_title("Mission Success", pad=10)

    axes[0, 1].bar(
        x - width / 2.0,
        metric_values("mean_command_loss_rate", BASELINE.name),
        width=width,
        color=baseline_color,
        yerr=percentile_error("mean_command_loss_rate", "p10_command_loss_rate", "p90_command_loss_rate", BASELINE.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 1].bar(
        x + width / 2.0,
        metric_values("mean_command_loss_rate", MITIGATED.name),
        width=width,
        color=mitigated_color,
        yerr=percentile_error("mean_command_loss_rate", "p10_command_loss_rate", "p90_command_loss_rate", MITIGATED.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_xticks(x, labels, rotation=12)
    axes[0, 1].set_title("Command Packet Loss", pad=10)

    axes[1, 0].bar(
        x - width / 2.0,
        metric_values("mean_command_latency_ms", BASELINE.name),
        width=width,
        color=baseline_color,
        yerr=percentile_error("mean_command_latency_ms", "p10_command_latency_ms", "p90_command_latency_ms", BASELINE.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 0].bar(
        x + width / 2.0,
        metric_values("mean_command_latency_ms", MITIGATED.name),
        width=width,
        color=mitigated_color,
        yerr=percentile_error("mean_command_latency_ms", "p10_command_latency_ms", "p90_command_latency_ms", MITIGATED.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 0].set_xticks(x, labels, rotation=12)
    axes[1, 0].set_ylabel("Latency (ms)")
    axes[1, 0].set_title("Command Handshake Latency", pad=10)

    axes[1, 1].bar(
        x - width / 2.0,
        metric_values("mean_retries_per_waypoint", BASELINE.name),
        width=width,
        color=baseline_color,
        yerr=percentile_error("mean_retries_per_waypoint", "p10_retries_per_waypoint", "p90_retries_per_waypoint", BASELINE.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 1].bar(
        x + width / 2.0,
        metric_values("mean_retries_per_waypoint", MITIGATED.name),
        width=width,
        color=mitigated_color,
        yerr=percentile_error("mean_retries_per_waypoint", "p10_retries_per_waypoint", "p90_retries_per_waypoint", MITIGATED.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 1].set_xticks(x, labels, rotation=12)
    axes[1, 1].set_title("Retries Per Waypoint", pad=10)

    min_range_m, max_range_m = operator_range_bounds()
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=baseline_color),
        plt.Rectangle((0, 0), 1, 1, color=mitigated_color),
    ]
    fig.legend(
        legend_handles,
        [BASELINE.name, MITIGATED.name],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncols=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Five-Waypoint Mission Monte Carlo | 32 B clear vs 64 B encrypted\n"
        f"Operator baseline {OPERATOR_TO_DRONE_FREQ_KHZ:.0f} kHz | Hop pairs include {OPERATOR_HOP_PAIRS_KHZ[1][0]:.0f}/{OPERATOR_HOP_PAIRS_KHZ[1][1]:.0f} kHz | "
        f"Shadow mean {MISSION_SHADOW_MEAN_DB_MIN:.0f}-{MISSION_SHADOW_MEAN_DB_MAX:.0f} dB | Range {min_range_m:.0f}-{max_range_m:.0f} m",
        fontsize=15,
        y=0.975,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_gossip_metrics_png(summary, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = list(summary.keys())
    x = np.arange(len(labels))
    width = 0.34
    baseline_color = "#4A5568"
    mitigated_color = "#2B6CB0"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.80, wspace=0.24, hspace=0.34)

    def metric_values(metric_name, mitigation_name):
        return [summary[label]["results"][mitigation_name][metric_name] for label in labels]

    def percentile_error(metric_name, low_name, high_name, mitigation_name):
        means = np.array(metric_values(metric_name, mitigation_name))
        lows = np.array(metric_values(low_name, mitigation_name))
        highs = np.array(metric_values(high_name, mitigation_name))
        return np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])

    axes[0, 0].bar(
        x - width / 2.0,
        metric_values("mean_gossip_loss_rate", BASELINE.name),
        width=width,
        color=baseline_color,
        yerr=percentile_error("mean_gossip_loss_rate", "p10_gossip_loss_rate", "p90_gossip_loss_rate", BASELINE.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 0].bar(
        x + width / 2.0,
        metric_values("mean_gossip_loss_rate", MITIGATED.name),
        width=width,
        color=mitigated_color,
        yerr=percentile_error("mean_gossip_loss_rate", "p10_gossip_loss_rate", "p90_gossip_loss_rate", MITIGATED.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_xticks(x, labels, rotation=12)
    axes[0, 0].set_title("Gossip Packet Loss", pad=10)

    axes[0, 1].bar(
        x - width / 2.0,
        metric_values("mean_gossip_latency_ms", BASELINE.name),
        width=width,
        color=baseline_color,
        yerr=percentile_error("mean_gossip_latency_ms", "p10_gossip_latency_ms", "p90_gossip_latency_ms", BASELINE.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 1].bar(
        x + width / 2.0,
        metric_values("mean_gossip_latency_ms", MITIGATED.name),
        width=width,
        color=mitigated_color,
        yerr=percentile_error("mean_gossip_latency_ms", "p10_gossip_latency_ms", "p90_gossip_latency_ms", MITIGATED.name),
        capsize=5,
        ecolor="#1A202C",
    )
    axes[0, 1].set_xticks(x, labels, rotation=12)
    axes[0, 1].set_ylabel("Latency (ms)")
    axes[0, 1].set_title("Gossip One-Way Latency", pad=10)

    baseline_bitrate_kbps = np.array(metric_values("mean_gossip_bitrate_bps", BASELINE.name)) / 1000.0
    mitigated_bitrate_kbps = np.array(metric_values("mean_gossip_bitrate_bps", MITIGATED.name)) / 1000.0
    baseline_bitrate_error = percentile_error(
        "mean_gossip_bitrate_bps", "p10_gossip_bitrate_bps", "p90_gossip_bitrate_bps", BASELINE.name
    ) / 1000.0
    mitigated_bitrate_error = percentile_error(
        "mean_gossip_bitrate_bps", "p10_gossip_bitrate_bps", "p90_gossip_bitrate_bps", MITIGATED.name
    ) / 1000.0
    axes[1, 0].bar(
        x - width / 2.0,
        baseline_bitrate_kbps,
        width=width,
        color=baseline_color,
        yerr=baseline_bitrate_error,
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 0].bar(
        x + width / 2.0,
        mitigated_bitrate_kbps,
        width=width,
        color=mitigated_color,
        yerr=mitigated_bitrate_error,
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 0].set_xticks(x, labels, rotation=12)
    axes[1, 0].set_ylabel("Bitrate (kbps)")
    axes[1, 0].set_title("Aggregate Gossip Bitrate", pad=10)

    baseline_peer_age_ms = np.array(metric_values("mean_peer_age_s", BASELINE.name)) * MS_PER_SECOND
    mitigated_peer_age_ms = np.array(metric_values("mean_peer_age_s", MITIGATED.name)) * MS_PER_SECOND
    baseline_peer_age_error = percentile_error("mean_peer_age_s", "p10_peer_age_s", "p90_peer_age_s", BASELINE.name) * MS_PER_SECOND
    mitigated_peer_age_error = percentile_error("mean_peer_age_s", "p10_peer_age_s", "p90_peer_age_s", MITIGATED.name) * MS_PER_SECOND
    axes[1, 1].bar(
        x - width / 2.0,
        baseline_peer_age_ms,
        width=width,
        color=baseline_color,
        yerr=baseline_peer_age_error,
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 1].bar(
        x + width / 2.0,
        mitigated_peer_age_ms,
        width=width,
        color=mitigated_color,
        yerr=mitigated_peer_age_error,
        capsize=5,
        ecolor="#1A202C",
    )
    axes[1, 1].set_xticks(x, labels, rotation=12)
    axes[1, 1].set_ylabel("Age (ms)")
    axes[1, 1].set_title("Peer-State Age", pad=10)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=baseline_color),
        plt.Rectangle((0, 0), 1, 1, color=mitigated_color),
    ]
    fig.legend(
        legend_handles,
        [BASELINE.name, MITIGATED.name],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncols=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Drone-to-Drone Gossip Metrics | 32 B clear vs 64 B encrypted\n"
        f"Swarm baseline {DRONE_TO_DRONE_FREQ_KHZ:.0f} kHz | Hop pairs include {DRONE_HOP_PAIRS_KHZ[1][0]:.0f}/{DRONE_HOP_PAIRS_KHZ[1][1]:.0f} kHz | "
        "One-way state updates",
        fontsize=15,
        y=0.975,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_sample_tracks(ax, track_samples, center_path, title):
    colors = ["#1F4E79", "#2F855A", "#C05621", "#805AD5", "#C53030"]
    for drone_idx in range(NUM_DRONES):
        ax.plot(track_samples[:, drone_idx, 0], track_samples[:, drone_idx, 1], color=colors[drone_idx], linewidth=1.8, label=f"Drone {drone_idx + 1}")
        ax.scatter(track_samples[0, drone_idx, 0], track_samples[0, drone_idx, 1], color=colors[drone_idx], marker="s", s=20)
        ax.scatter(track_samples[-1, drone_idx, 0], track_samples[-1, drone_idx, 1], color=colors[drone_idx], marker="o", s=24)

    patrol_x = center_path[:, 0]
    patrol_y = center_path[:, 1]
    ax.plot(patrol_x, patrol_y, linestyle="--", color="#1A202C", linewidth=1.2, alpha=0.65, label="Waypoint centers")
    ax.scatter(patrol_x, patrol_y, marker="x", color="#1A202C", s=34)

    x_points = np.concatenate([track_samples[:, :, 0].reshape(-1), patrol_x])
    y_points = np.concatenate([track_samples[:, :, 1].reshape(-1), patrol_y])
    x_pad = 28.0
    y_extent = max(abs(float(np.min(y_points))), abs(float(np.max(y_points))))
    y_pad = max(16.0, 0.18 * max(y_extent, 1.0))
    x_limits = (float(np.min(x_points) - x_pad), float(np.max(x_points) + x_pad))
    y_limits = (-y_extent - y_pad, y_extent + y_pad)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    inset = ax.inset_axes([0.03, 0.06, 0.28, 0.28])
    inset.set_facecolor("#F7FAFC")
    inset.plot(patrol_x, patrol_y, linestyle="--", color="#1A202C", linewidth=0.9, alpha=0.65)
    inset.scatter(patrol_x, patrol_y, marker="x", color="#1A202C", s=16)
    inset.scatter(OPERATOR_POS[0], OPERATOR_POS[1], marker="*", color="#D69E2E", s=60)
    inset.add_patch(plt.Circle((OPERATOR_POS[0], OPERATOR_POS[1]), 1000.0, color="#718096", fill=False, linestyle=":", linewidth=0.9))
    inset.add_patch(plt.Circle((OPERATOR_POS[0], OPERATOR_POS[1]), 2000.0, color="#718096", fill=False, linestyle="--", linewidth=0.9))
    inset.add_patch(
        Rectangle(
            (x_limits[0], y_limits[0]),
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            facecolor="#90CDF4",
            edgecolor="#2B6CB0",
            linewidth=1.0,
            alpha=0.18,
        )
    )
    context_x_min = min(OPERATOR_POS[0] - 250.0, x_limits[0] - 120.0)
    context_x_max = max(float(np.max(patrol_x)) + 140.0, x_limits[1] + 60.0)
    context_y_extent = max(2200.0, abs(y_limits[0]) * 2.2)
    inset.set_xlim(context_x_min, context_x_max)
    inset.set_ylim(-context_y_extent, context_y_extent)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_aspect("equal", adjustable="box")
    for spine in inset.spines.values():
        spine.set_color("#A0AEC0")
        spine.set_linewidth(0.8)

    min_range_m, max_range_m = range_bounds_for_centers(center_path)
    ax.text(
        0.03,
        0.98,
        f"Operator standoff {min_range_m/1000.0:.2f}-{max_range_m/1000.0:.2f} km",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#CBD5E0", alpha=0.92),
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal", adjustable="box")


def save_track_png(collision_summary, mission_summary, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.81, wspace=0.12, hspace=0.22)

    plot_sample_tracks(
        axes[0, 0],
        collision_summary["64 B encrypted"]["results"][BASELINE.name]["example"].track_samples,
        COLLISION_PATROL_CENTERS,
        "Collision-only | 64 B encrypted | No mitigation",
    )
    plot_sample_tracks(
        axes[0, 1],
        collision_summary["64 B encrypted"]["results"][MITIGATED.name]["example"].track_samples,
        COLLISION_PATROL_CENTERS,
        "Collision-only | 64 B encrypted | Hop + redundancy",
    )
    plot_sample_tracks(
        axes[0, 2],
        mission_summary["32 B clear"]["results"][BASELINE.name]["example"].track_samples,
        PATROL_CENTERS,
        "32 B clear | No mitigation",
    )
    plot_sample_tracks(
        axes[1, 0],
        mission_summary["32 B clear"]["results"][MITIGATED.name]["example"].track_samples,
        PATROL_CENTERS,
        "32 B clear | Hop + redundancy",
    )
    plot_sample_tracks(
        axes[1, 1],
        mission_summary["64 B encrypted"]["results"][BASELINE.name]["example"].track_samples,
        PATROL_CENTERS,
        "64 B encrypted | No mitigation",
    )
    plot_sample_tracks(
        axes[1, 2],
        mission_summary["64 B encrypted"]["results"][MITIGATED.name]["example"].track_samples,
        PATROL_CENTERS,
        "64 B encrypted | Hop + redundancy",
    )

    colors = ["#1F4E79", "#2F855A", "#C05621", "#805AD5", "#C53030"]
    handles = [Line2D([0], [0], color=colors[idx], linewidth=1.8) for idx in range(NUM_DRONES)]
    labels = [f"Drone {idx + 1}" for idx in range(NUM_DRONES)]
    handles.extend(
        [
            Line2D([0], [0], color="#1A202C", linestyle="--", linewidth=1.2),
            Line2D([0], [0], marker="*", color="#D69E2E", linestyle="None", markersize=10),
        ]
    )
    labels.extend(["Waypoint centers", "Operator"])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncols=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Five-Drone XY Motion | Collision and Mission Examples\n"
        f"Operator baseline {OPERATOR_TO_DRONE_FREQ_KHZ:.0f} kHz, hop-pair set around 28 kHz | "
        f"Swarm baseline {DRONE_TO_DRONE_FREQ_KHZ:.0f} kHz, hop-pair set around 50 kHz",
        fontsize=16,
        y=0.975,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def print_summary(collision_summary, mission_summary):
    min_range_m, max_range_m = operator_range_bounds()
    print("Five-drone underwater fleet simulation")
    print(f"Operator position: ({OPERATOR_POS[0]:.0f}, {OPERATOR_POS[1]:.0f}, {OPERATOR_POS[2]:.0f}) m")
    print(f"Operator-to-drone geometry envelope from waypoint plan: min {min_range_m:.0f} m, max {max_range_m:.0f} m")
    print(
        f"Configured link frequencies: operator baseline {OPERATOR_TO_DRONE_FREQ_KHZ:.0f} kHz, "
        f"operator hop pairs {OPERATOR_HOP_PAIRS_KHZ}, "
        f"drone baseline {DRONE_TO_DRONE_FREQ_KHZ:.0f} kHz, "
        f"drone hop pairs {DRONE_HOP_PAIRS_KHZ}"
    )
    print(
        f"Swarm coordination: {GOSSIP_SENDERS_PER_STEP} gossip sender(s) per control step at 50 kHz; "
        f"avoidance uses dead-reckoned peer state; base access miss {GOSSIP_ACCESS_MISS_PROB:.2f}; "
        f"prediction horizon {DEAD_RECKONING_HORIZON_S:.0f}s"
    )
    print("Mitigation packet copies are sent simultaneously on the active hop pair, not as serial retransmissions")
    print(f"Mission shadowing: trial mean sampled uniformly from {MISSION_SHADOW_MEAN_DB_MIN:.1f} dB to {MISSION_SHADOW_MEAN_DB_MAX:.1f} dB")
    print()
    print("Experiment 1: collision-only autonomous patrol over 6 hours")
    for packet_name, packet_summary in collision_summary.items():
        print(f"  Packet profile: {packet_name} | trials={packet_summary['trials']}")
        for mitigation_name in [BASELINE.name, MITIGATED.name]:
            metrics = packet_summary["results"][mitigation_name]
            print(
                f"    {mitigation_name:26s}"
                f" collision_free_rate={metrics['collision_free_rate']:.3f}"
                f" ci95=[{metrics['collision_free_ci_low']:.3f},{metrics['collision_free_ci_high']:.3f}]"
                f" mean_min_separation={metrics['mean_min_separation_m']:.1f}m"
                f" p10-p90=[{metrics['p10_min_separation_m']:.1f},{metrics['p90_min_separation_m']:.1f}]m"
                f" p05_min_separation={metrics['p05_min_separation_m']:.1f}m"
                f" gossip_delivery={metrics['mean_gossip_delivery_rate']:.3f}"
                f" gossip_latency={metrics['mean_gossip_latency_ms']:.0f}ms"
                f" mean_peer_age={metrics['mean_peer_age_s']:.1f}s"
            )
    print()
    print("Experiment 2: operator sends 5 sequential waypoints with representative SYN/ACK handshakes")
    for packet_name, packet_summary in mission_summary.items():
        print(f"  Packet profile: {packet_name} | trials={packet_summary['trials']}")
        for mitigation_name, metrics in packet_summary["results"].items():
            print(
                f"    {mitigation_name:26s}"
                f" mission_success={metrics['mission_success_rate']:.3f}"
                f" ci95=[{metrics['mission_success_ci_low']:.3f},{metrics['mission_success_ci_high']:.3f}]"
                f" collision_free={metrics['collision_free_rate']:.3f}"
                f" completed_waypoints={metrics['mean_completed_waypoints']:.2f}"
                f" cmd_loss={metrics['mean_command_loss_rate']:.3f}"
                f" arrival_loss={metrics['mean_arrival_loss_rate']:.3f}"
                f" cmd_rtt={metrics['mean_command_latency_ms']:.0f}ms"
                f" arrival_rtt={metrics['mean_arrival_latency_ms']:.0f}ms"
                f" retries_per_waypoint={metrics['mean_retries_per_waypoint']:.2f}"
                f" gossip_delivery={metrics['mean_gossip_delivery_rate']:.3f}"
                f" gossip_latency={metrics['mean_gossip_latency_ms']:.0f}ms"
                f" mean_peer_age={metrics['mean_peer_age_s']:.1f}s"
            )
    print()
    print("Saved figures:")
    print("  fleet_collision_summary.png")
    print("  fleet_mission_metrics.png")
    print("  fleet_gossip_metrics.png")
    print("  fleet_motion_tracks.png")
    print("Saved CSV:")
    print("  fleet_collision_summary.csv")
    print("  fleet_mission_summary.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Underwater five-drone acoustic Monte Carlo simulator")
    parser.add_argument(
        "--trial-preset",
        choices=sorted(TRIAL_PRESETS.keys()),
        default=None,
        help="Choose lower or higher Monte Carlo counts. If omitted, prompt for 1 or 2.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers for Monte Carlo execution. Defaults to auto.",
    )
    return parser.parse_args()


def resolve_trial_preset(cli_value, input_func=input, stdin_is_tty=None, output_func=print):
    if cli_value is not None:
        return cli_value

    if stdin_is_tty is None:
        stdin_is_tty = sys.stdin.isatty()

    if not stdin_is_tty:
        output_func(f"No interactive input available; defaulting to {DEFAULT_TRIAL_PRESET} preset.")
        return DEFAULT_TRIAL_PRESET

    output_func("Select Monte Carlo trial preset:")
    output_func(
        f"  1. low  ({LOW_COLLISION_TRIALS} collision trials, {LOW_MISSION_TRIALS} mission trials per case)"
    )
    output_func(
        f"  2. high ({HIGH_COLLISION_TRIALS} collision trials, {HIGH_MISSION_TRIALS} mission trials per case)"
    )

    while True:
        selection = input_func("Enter 1 or 2 [1]: ").strip()
        if selection in ("", "1"):
            return "low"
        if selection == "2":
            return "high"
        output_func("Invalid selection. Enter 1 for low or 2 for high.")


def main():
    args = parse_args()
    trial_preset_name = resolve_trial_preset(args.trial_preset)
    trial_preset = TRIAL_PRESETS[trial_preset_name]
    collision_summary = run_collision_monte_carlo(trial_preset.collision_trials, args.workers)
    mission_summary = run_operator_monte_carlo(trial_preset.mission_trials, args.workers)
    output_dir = os.path.dirname(__file__)

    save_collision_png(collision_summary, os.path.join(output_dir, "fleet_collision_summary.png"))
    save_mission_metrics_png(mission_summary, os.path.join(output_dir, "fleet_mission_metrics.png"))
    save_gossip_metrics_png(mission_summary, os.path.join(output_dir, "fleet_gossip_metrics.png"))
    save_track_png(collision_summary, mission_summary, os.path.join(output_dir, "fleet_motion_tracks.png"))
    save_collision_csv(collision_summary, os.path.join(output_dir, "fleet_collision_summary.csv"))
    save_mission_csv(mission_summary, os.path.join(output_dir, "fleet_mission_summary.csv"))
    print_summary(collision_summary, mission_summary)


if __name__ == "__main__":
    main()
