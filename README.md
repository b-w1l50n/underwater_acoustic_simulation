DISCLAIMER: Written with the help of Codex.
# Underwater Acoustic Fleet Simulation

This project simulates a five-drone underwater fleet with one remote human operator.

It is designed to answer three practical questions:

- Is the acoustic link operationally usable?
- What breaks first as the channel degrades?
- Does the overall system still function well enough to complete the mission?

The simulator focuses on two experiments:

- A six-hour collision-endurance patrol for the five-drone swarm.
- A five-waypoint operator-controlled mission with SYN/ACK command and arrival handshakes.

## What The Model Includes

- Five AUVs moving in formation at `3 kn`.
- One fixed operator modem.
- Operator-to-drone acoustic link at `28 kHz`.
- Drone-to-drone gossip / coordination link at `50 kHz`.
- Separate modem rates by link:
  - operator link at `5120 bps`
  - swarm gossip link at `10240 bps`
- Two packet sizes:
  - `32 B clear`
  - `64 B encrypted` (`32 B` packet plus `32 B` encryption overhead)
- Two mitigation modes:
  - `No mitigation`
  - `Frequency hopping + redundancy`

## What Is Modeled Reasonably Well

- Acoustic transmission loss using spreading and Thorp absorption.
- Ambient noise from wind, shipping, and thermal noise.
- Simple surface and bottom multipath penalty.
- Doppler penalty based on radial relative motion.
- Log-normal shadowing.
- Temporary burst-fade states that make nearby packets statistically more likely to experience extra fading for a few seconds.
- Narrowband fades that make hopping and frequency diversity matter.
- Packet airtime and latency from propagation plus transmit time.
- Different airtime on the operator and swarm links because the modem rates are different.
- Packet error probability from an SNR -> BER -> PER approximation.
- Parallel frequency-diverse packet copies for the mitigated case.
- Stop-and-wait command / arrival handshakes with retries and backoff.
- A `15 s` command-request behavior so the fleet asks for a fresh command if one is not received.
- Swarm control that depends on communicated peer state rather than perfect knowledge.
- Dead reckoning on stale peer state using last received velocity and message age.
- Monte Carlo averaging, confidence intervals, and percentile bands.

## What Is Simplified

This is not a waveform-level modem simulation and not a full underwater network stack.

The main simplifications are:

- Track plots are shown in `x/y`, but collision checks and close-range sensing now use full 3D separation.
- The water column is uniform; there is no sound-speed profile, refraction, thermocline, or temperature-gradient model.
- Currents are sampled once per trial and treated as spatially uniform.
- Drone-to-drone gossip uses a compact access-loss model rather than a detailed MAC with queues and scheduling.
- The swarm uses a simple constant-velocity dead-reckoning step, not a full estimator or filter.
- A close-range local sensing layer still exists as a last-ditch safety term.
- The operator mission uses one selected representative drone for each handshake rather than a richer fleet command protocol.
- Obstruction / shadowing by one vehicle physically blocking another is not explicitly modeled.
- Fade correlation is modeled with a lightweight temporary-bias state rather than a full channel-state Markov process.

## Current Collision-Safety Interpretation

The collision threshold in the simulator is `1 m`.

The nominal formation is a compact cross with about `7 m` stand-off from the center vehicle to each side vehicle, so the controller is trying to keep neighbors in roughly the `6-8 m` band.

Each drone also has a last-ditch local sensing layer with a `5 m` proximity range. The swarm controller therefore tries to:

- stay in formation near the `6-8 m` target spacing
- avoid each other using communicated and dead-reckoned peer state inside the broader stand-off zone
- fall back to short-range local sensing inside `5 m`

The collision-endurance patrol uses a gentler loiter path than the operator mission. That is intentional. It is meant to test whether the swarm can remain coherent and avoid collisions over long duration, not whether it can survive the sharpest possible turns for six hours straight.

The collision study now uses the same packet-size sweep as the mission study:

- `32 B clear`
- `64 B encrypted`

That keeps the comparison internally consistent across operator traffic and swarm gossip, even though the realistic deployed case is likely the `64 B encrypted` profile.

## How The Mitigation Works

Mitigation is not modeled as "free SNR gain."

Instead, the simulator does this:

- Frequency hopping chooses a hop pair around the nominal carrier.
- Redundancy sends simultaneous copies on that active hop pair.
- A logical packet succeeds if either copy gets through.
- Packet latency is the earliest successful copy, not the sum of multiple copies.

Transient fading also has short memory:

- each packet gets the usual per-packet fade draw
- a separate low-probability roll can put the link into a temporary degraded state
- while degraded, subsequent packets are biased toward extra fading for roughly `2-5 s`
- after that, the link enters a cooldown window before another temporary burst can start

Current hop-pair sets are:

- Operator link around `28 kHz`: `(24, 28)`, `(26, 30)`, `(28, 32)`
- Swarm link around `50 kHz`: `(46, 50)`, `(48, 52)`, `(50, 54)`

## How The Swarm Gossip Works

Each drone keeps a belief about every other drone:

- last received position
- last received velocity
- age of that information

At each control step:

- drones attempt to gossip state over the `50 kHz` link
- some transmissions are missed due to access loss or channel loss
- successful receptions refresh the receiver's belief state

The controller then uses dead-reckoned peer positions:

- `predicted peer position = last received position + last received velocity * age`

That means the swarm does not steer on perfect truth unless peers are very close and the local safety layer engages.

## Experiments

### 1. Collision-Endurance Patrol

Purpose:

- Measure whether the five-drone swarm stays collision-free over `6 hours`.
- Compare `32 B clear` and `64 B encrypted` gossip traffic under both mitigation modes.

Key outputs:

- collision-free patrol rate
- minimum pairwise separation
- closest-approach distribution
- gossip delivery rate
- gossip packet loss
- gossip one-way latency
- aggregate gossip bitrate
- peer-state age

### 2. Operator-Controlled Five-Waypoint Mission

Purpose:

- Measure whether the mission still completes when operator command / acknowledgment traffic has loss, retries, and latency.

Sequence per waypoint:

- operator sends command SYN
- one drone is chosen to send ACK back
- fleet travels to waypoint
- one drone is chosen to send arrival SYN
- operator sends ACK back
- next waypoint begins

If the fleet does not receive a command for `15 s`, it asks for a command again instead of treating one failed packet as immediate mission failure.

Key outputs:

- mission success rate
- command packet loss
- arrival packet loss
- command handshake latency
- arrival handshake latency
- retries per waypoint
- completed waypoints

## Code Outline

The major sections are:

- Constants and geometry:
  - frequencies, timing, formation offsets, waypoint paths, environments
- Data structures:
  - `AcousticEnvironment`
  - `Mitigation`
  - `PacketProfile`
  - result dataclasses
  - `SwarmBeliefState`
- Acoustic channel model:
  - noise, spreading, absorption, multipath, Doppler, fades
- Packet model:
  - packet copies, hopping, PER, latency
- Swarm-control model:
  - gossip updates
  - dead-reckoned peer avoidance
  - close-range local repulsion
- Experiment runners:
  - collision patrol
  - operator mission
- Monte Carlo aggregation:
  - trial execution
  - confidence intervals
  - percentile summaries
- Output writers:
  - PNG figures
  - CSV summaries

## Running The Simulation

Run the script directly:

```bash
python3 underwater_fleet_sim.py
```

If run interactively, it prompts for a trial preset:

- `1` = low
- `2` = high

You can also pass it explicitly:

```bash
python3 underwater_fleet_sim.py --trial-preset low
python3 underwater_fleet_sim.py --trial-preset high
```

Optional worker control:

```bash
python3 underwater_fleet_sim.py --trial-preset high --workers 1
```

## Generated Outputs

The simulator writes results next to the script:

- `fleet_collision_summary.png`
- `fleet_mission_metrics.png`
- `fleet_gossip_metrics.png`
- `fleet_motion_tracks.png`
- `fleet_collision_summary.csv`
- `fleet_mission_summary.csv`

## How To Read The Results

- If mission success stays high while latency and retries rise, the link is degraded but still operationally usable.
- If packet loss and retries rise enough that completed waypoints drop, the operator link is what is breaking the mission.
- If collision-free patrol rate falls, the swarm-coordination link or control law is not robust enough for the tested formation density and path geometry.
- The `fleet_gossip_metrics.png` figure is the swarm-link companion to the operator mission dashboard: it compares `32 B` vs `64 B` and mitigation vs no mitigation for packet loss, one-way latency, aggregate bitrate, and peer-state age.
- A slightly smaller closest-approach value under mitigation does not automatically mean mitigation is worse. Better gossip can let the swarm hold a tighter intended formation more accurately.
