# Matek CAN-L4-3100 (RM3100) Datasheet

Written 14 Oct 2026 for the Cohera X500 V2 build. Editable copy: https://claude.ai/code/artifact/64e3bdd7-ea15-4446-8987-b0ad60348fb9

## Overview

The Matek CAN-L4-3100 is a PNI RM3100 three-axis magnetometer on a small STM32L431 board that runs ArduPilot's AP_Periph firmware and reports over DroneCAN. On our X500 V2 it plugs into a CAN port on the Pixhawk 6C (ArduCopter 4.7.2) and stands in for a QuSpin optically pumped magnetometer while we build the Tellus map-matching pipeline.

It is a navigation-grade compass, not a survey magnetometer. It is good enough to log interference and exercise the software path end to end, but its noise and the DroneCAN/ArduPilot data path are too coarse for real anomaly matching over one field (see Limitations).

## Specifications

Board figures are from Matek's product listing ([GetFPV](https://www.getfpv.com/mateksys-ap-periph-can-magnetometer-rm3100.html)); sensor figures are PNI's RM3100 numbers as published by [Holybro](https://docs.holybro.com/peripherals/dronecan-rm3100-compass).

**Board**

| Item | Value |
| --- | --- |
| Sensor | PNI RM3100 (magneto-inductive, 3-axis) |
| MCU | STM32L431xC, 256 KB flash |
| Firmware | ArduPilot AP_Periph, MatekL431-Periph or MatekL431-GPS build |
| Bus | DroneCAN (CAN 2.0B, 1 Mbit/s default) |
| Other ports | UART2 (MSP out), UART3 (external GNSS), SWD (SWCLK, SWDIO) |
| Supply | 4.5 to 5.5 V, about 22 mA |
| Operating temperature | -40 to 85 °C |
| Size | 36 × 25 × 10 mm |
| Weight | 6 g |
| In the box | Board, 20 cm JST-GH 4-pin to JST-GH 4-pin cable |
| LEDs | Blue fast blink = booting, blue slow blink = running, red = 3.3 V present |

**RM3100 sensor, by cycle count**

| Cycle count | 50 | 100 | 200 |
| --- | --- | --- | --- |
| Range (µT) | ±800 | ±800 | ±800 |
| Gain (LSB/µT) | 20 | 38 | 75 |
| Sensitivity (nT) | 50 | 26 | 13 |
| Noise (nT) | 30 | 20 | 15 |
| Max single-axis rate (Hz) | 1600 | 850 | 440 |

Divide the single-axis rate by 3 for a full 3-axis sample. ArduPilot's RM3100 driver runs the sensor at 200 cycles, so expect about 13 nT resolution and 15 nT noise at the chip (inferred from the driver, not measured on our unit). For scale, Ireland's total field is about 49,000 nT, well inside the ±800 µT range.

## Connector and wiring

The CAN connector is the standard 4-pin JST-GH used by Pixhawk CAN ports, so the supplied cable plugs straight into **CAN1** on the Pixhawk 6C and also powers the node.

| Pin | Signal |
| --- | --- |
| 1 | 5 V |
| 2 | CAN_H |
| 3 | CAN_L |
| 4 | GND |

This is the Pixhawk/DroneCAN standard order; Matek's listing doesn't print a pinout, so check the silkscreen before making a longer cable. The 6C's own pinout for CAN1 and CAN2 is in [Holybro's Pixhawk 6C docs](https://docs.holybro.com/autopilot/pixhawk-6c/pixhawk-6c-pinout).

- One node on a short cable needs no extra termination. If you daisy-chain more CAN devices later, terminate the far end.
- Leave UART2, UART3 and SWD unconnected. They are only for MSP output, an add-on GPS, or reflashing with an ST-Link.

## Data path and resolution

What reaches our logs is coarser than what the chip measures: about 13 nT at the sensor, 12 to 49 nT on the CAN bus, and 100 nT in the ArduPilot log.

| Stage | Format | Step at Irish field strength |
| --- | --- | --- |
| RM3100 at 200 cycles | 24-bit integer, 75 LSB/µT | about 13 nT |
| DroneCAN message `MagneticFieldStrength2` | three half-precision floats, in gauss | 12 to 49 nT per axis, depending on the axis value |
| ArduPilot `MAG` log message (also MAVLink `SCALED_IMU`) | 16-bit integer, milligauss | 100 nT |

The half-precision step doubles each time an axis passes a power of two in gauss. An axis reading 0.125 to 0.25 G has a 12 nT step, 0.25 to 0.5 G a 24 nT step, and above 0.5 G a 49 nT step. These limits were checked in ArduPilot and DroneCAN source for the project's mag-nav viability note.

- The `MAG` log entry has calibration offsets already applied. It also logs the offsets (`OfsX/Y/Z`) and motor compensation (`MOX/MOY/MOZ`), so you can rebuild the raw reading in post.
- Timestamps are the autopilot's boot time and line up with GPS and battery current in the same log.
- The node's publish rate is a setting in the node's own firmware. Read it in DroneCAN GUI Tool (node parameters) and raise it if the logged rate looks low.
- If we ever need the full sensor resolution, the route is a companion computer reading the RM3100 directly over SPI or I2C, not this board.

## Setup in QGroundControl

The node needs two CAN settings and a reboot; after that it shows up as an extra compass. Parameter names in brackets are only search hints for Vehicle Setup > Parameters.

1. Plug the node into **CAN1** with the supplied cable and power the drone. The blue LED should go from fast to slow blink.
2. Vehicle Setup > Parameters, search **CAN_P1**. Set the CAN1 port's driver to **First driver** (`CAN_P1_DRIVER`).
3. Search **CAN_D1**. Set the first driver's protocol to **DroneCAN** (`CAN_D1_PROTOCOL`).
4. Tools (top right of the Parameters page) > **Reboot Vehicle**.
5. Vehicle Setup > Sensors > **Compass**. A third compass should now be listed alongside the GPS-puck and internal IST8310s.
6. In the compass priority list, keep the **GPS-puck compass first**. Put the RM3100 second and untick **Use Compass** for it, so it is read and logged but the EKF never steers on it. Untick Use Compass for the internal 6C compass too (`COMPASS_USE`, `COMPASS_USE2`, `COMPASS_USE3`, `COMPASS_PRIO1_ID`).
7. Search **COMPASS_AUTODEC** and set auto declination to **Enabled**, as Matek recommends.
8. Run the compass calibration on the same page with all compasses, so the RM3100 gets offsets logged even though it isn't used.
9. Leave motor compensation off for the RM3100. QGC has no compassmot screen and the team has no Windows machine, so motor interference is modelled in post from the logs instead (see Mounting and interference).

For bench and walk-around tests, set Log while disarmed to **Enabled** (`LOG_DISARMED`) and turn it back off before flights so the SD card doesn't fill. Keep the default log bitmask; it already includes compass.

## Mounting and interference

Distance from the motors, ESCs and battery leads matters more than anything else on this list: current near the arms puts hundreds to thousands of nT on the sensor, against Tellus anomalies of tens to hundreds.

- Mount it on a non-ferrous mast (carbon or plastic) above the frame, as far from the power distribution and battery leads as you can. Nylon or brass screws only near the board.
- Align its arrow with the frame's forward direction. If it can't be, set the RM3100's orientation on the Sensors > Compass page to match how it is mounted.
- Keep the CAN cable and other wires twisted and away from the board.
- Log a few minutes of steady hover with no translation at several throttle levels. Regress each axis against battery current (`BAT`) and per-motor RPM (`ESC`, from bidirectional DShot) to build our own interference model.
- Log a static ground reference, or use a geomagnetic observatory such as Valentia, if you need to remove the daily drift of the field.
- Watch board temperature on long sessions. The RM3100 is fairly stable, but the offsets can shift as the node and nearby ESCs warm up.

## Limitations for mag-nav

The CAN-L4-3100 can't fix a position against the Tellus map; it is about 20 to 100 times coarser than the QuSpin it stands in for, so the demo uses a synthetic reading sampled from the map.

| | Matek CAN-L4-3100 (RM3100) | QuSpin (QTFM / MAG-PIE class) |
| --- | --- | --- |
| Measures | 3 vector axes | Total field (scalar), plus vector on some models |
| Sensor noise | about 15 nT | about 1 nT or better |
| Usable step in our logs | 100 nT (ArduPilot log) | Set by our own logger |
| Heading error on total field | Tens of nT after calibration | Small; scalar sensors don't depend on axis calibration |
| Interface | DroneCAN into the autopilot | Serial to a companion computer |
| Cost and weight | Low cost, 6 g | Much higher on both |

- Tellus anomalies over a typical 200 to 300 m field span only tens of nT, below what this data path resolves.
- What it is good for: proving the CAN wiring and logging, characterising motor and ESC interference against current and RPM, and giving a realistic noisy vector input to the matcher.

## Sources

- [MATEKSYS AP_Periph CAN Magnetometer - RM3100 (GetFPV listing, Matek's spec text)](https://www.getfpv.com/mateksys-ap-periph-can-magnetometer-rm3100.html): board, power, size, weight, LEDs, firmware, ArduPilot settings.
- [Holybro DroneCAN RM3100 Compass](https://docs.holybro.com/peripherals/dronecan-rm3100-compass): PNI RM3100 range, gain, sensitivity, noise and sample rate by cycle count.
- [Holybro Pixhawk 6C pinout](https://docs.holybro.com/autopilot/pixhawk-6c/pixhawk-6c-pinout): CAN1/CAN2 connector pins.

Not yet measured on our unit: the node's publish rate, its actual noise on the mast, and whether the board has a CAN termination resistor.
