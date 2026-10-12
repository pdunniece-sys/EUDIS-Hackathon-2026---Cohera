# Hardware

The test platform for Cohera. A Holybro X500 V2 quadcopter carries a
magnetometer over terrain covered by the Tellus survey, so we can record
real flight data and try magnetic-anomaly navigation on it.

The flight sensor is a Matek DroneCAN RM3100, a magneto-inductive
magnetometer. It stands in for the QuSpin QTFM optically pumped
magnetometer that a deployed system would carry. It is far less sensitive,
but it uses the same data path, mounting problems and interference
modelling. That lets us build and test the full pipeline before the
quantum sensor is available.

## Scope

1. **Flying platform.** A stock X500 V2 running ArduCopter, flown manually
   and in the GPS-assisted modes, with full DataFlash logging.
2. **Live RPM telemetry.** Bidirectional DShot from each ESC, so every log
   records per-motor RPM next to the magnetometer data.
3. **Magnetometer payload.** The RM3100 on a non-ferrous mast. It is logged
   by the flight controller but kept out of the EKF, so survey data never
   affects flight safety.
4. **Mag-nav in the loop.** A companion computer matches the field reading
   against the Tellus grid. It sends the resulting position back to the
   autopilot, and an RC switch flips between GNSS and mag-nav in flight.

## Bill of materials

| Part | Item | Notes |
|---|---|---|
| Frame | Holybro X500 V2 kit | 2216 motors (12N14P), 10" props |
| ESCs | Holybro BLHeli_S 20A (EFM8BB21) | To be reflashed to Bluejay for bidirectional DShot |
| Flight controller | Holybro Pixhawk 6C | ArduCopter 4.7.1, `Pixhawk6C-bdshot` build |
| Power | Holybro PM02 power module, 4S 4000 mAh LiPo | Analog voltage and current monitoring |
| GNSS | Holybro u-blox M10 GPS on GPS1 | Puck has an IST8310 compass and the safety switch |
| RC | FlySky FS-i6X transmitter, iA6B receiver | iBUS on RC IN, 10 channels |
| Telemetry | SiK radio pair on TELEM1 | MAVLink2 at 57600 baud |
| Magnetometer | Matek DroneCAN RM3100 on CAN1 | Stand-in for the QuSpin QTFM |
| Remote ID | Dronetag DRI | Arriving |
| Ground station | QGroundControl | Debian, macOS and Android |

## Wiring

| Pixhawk 6C port | Connected to |
|---|---|
| MAIN (I/O PWM OUT) 1–4 | ESCs 1–4, plain PWM (moves to AUX for DShot) |
| RC IN | iA6B iBUS servo port |
| TELEM1 | SiK air radio |
| TELEM2 | Companion computer (planned) |
| GPS1 | M10 GPS, 10-pin lead |
| CAN1 | RM3100 node (planned) |
| POWER1 | PM02 |

### RC channels

| Channel | Control | Function |
|---|---|---|
| 1–4 | Sticks | Roll, pitch, throttle, yaw (Mode 2) |
| 5–6 | Knobs | Unused |
| 7 | SwC | Flight mode: Stabilize, AltHold, Loiter |
| 8 | SwD | RTL |
| 9 | SwA | AutoTune (planned) |
| 10 | SwB | GNSS / mag-nav EKF source switch (planned) |

## Status

**Done**
- X500 V2 built with the Pixhawk 6C, GPS, SiK radio and FlySky receiver.
- Software stack chosen: ArduCopter and QGroundControl. Mission Planner
  was ruled out because it only runs on Windows.
- ArduCopter 4.7.1 flashed, using the `Pixhawk6C-bdshot` build from the
  start, so the DShot upgrade needs no reflash.
- GPS working on GPS1, with the puck compass set as the primary compass.
  The internal compass is disabled.
- Receiver moved from PPM to iBUS. The transmitter switches are enabled and
  mapped to channels 7–10.

**Known issue**
- The GPS-puck compass (IST8310) is intermittent: it went missing on one
  boot and came back after a reseat. If compass pre-arm errors appear,
  check the GPS1 connector first, and secure it if it keeps dropping out.

**Next**
- Finish the calibrations and failsafe tests, then fly first on plain PWM
  with ESC calibration.
- Fit the Dronetag Remote ID module.
- AutoTune on SwA.

## Bidirectional DShot (live RPM telemetry)

**Why:** Motor and ESC current is the biggest source of interference on the
magnetometer. Logging each motor's RPM next to battery current lets us model
and subtract that interference far better than throttle alone would. The
same RPM feed also drives ArduPilot's harmonic notch filter, which gives a
cleaner gyro and a better tune. It also removes the need for ESC calibration.

**Route:** The motors move to the AUX ("FMU PWM OUT") rail on the
`Pixhawk6C-bdshot` build. RC IN and iBUS are not affected. We are avoiding
the MAIN-rail option (`BRD_IO_DSHOT`) because it disables pulse input on RC
IN, which would break iBUS.

**Steps:**
1. Move the four ESC signal leads from MAIN 1–4 to AUX 1–4.
2. Remap the outputs: `SERVO1–4_FUNCTION = 0`, `SERVO9–12_FUNCTION = 33–36`,
   `MOT_PWM_TYPE = 5` (DShot300), `SERVO_BLH_AUTO = 1`.
3. Flash Bluejay onto the BLHeli_S ESCs through the flight controller's
   passthrough using esc-configurator.com. Stock BLHeli_S has no
   bidirectional DShot.
4. Turn on telemetry: `SERVO_BLH_BDMASK = 3840` (SERVO9–12),
   `SERVO_BLH_POLES = 14`.
5. Check that all four motors report RPM. Then enable the harmonic notch
   (`INS_HNTCH_MODE = 3`) and tune it from a hover log.

Per-motor RPM is then recorded in the `ESC` log messages.

## Magnetometer integration

**Stage 1: log only.** Plug the RM3100 into CAN1 and enable DroneCAN
(`CAN_P1_DRIVER = 1`, `CAN_D1_PROTOCOL = 1`). Turn off the RM3100's "use"
flag in the compass list, so ArduPilot still reads and logs it but heading
comes only from the GPS-puck compass. Readings appear in the `MAG` log
messages, in milligauss (1 mG = 100 nT), with GPS-synced timestamps.
Calibration offsets and motor-compensation terms are logged alongside, so
the raw field can be recovered in post.

**Stage 2: interference characterisation.** Tellus anomalies are tens to
hundreds of nT. Motor currents near the arms cause thousands. To deal with
this:
- Mount the sensor on a non-ferrous mast as far from the motors and power
  leads as possible.
- Run `compassmot`.
- Fly hover segments with no translation.
- Model the interference against the per-motor RPM from DShot and against
  battery current.
- Use a ground reference or a nearby observatory (e.g. Valentia) to remove
  diurnal variation.

The analysis lives in [modelling/](../modelling/).

**Stage 3: mag-nav in the loop.** Add a companion computer on TELEM2
(MAVLink2 at 921600 baud). It streams the magnetometer data, runs the
Tellus matcher and sends the position back as `GPS_INPUT`, set up as a
second GPS (type 14). EKF3 source sets switch between real GNSS and mag-nav
on SwB (`RC10_OPTION = 90`). Test the whole loop in ArduPilot SITL before
flying it.

**Stage 4: quantum sensor.** Replace the RM3100 with the QuSpin QTFM. The
logging, compensation and matching pipeline stays the same.
