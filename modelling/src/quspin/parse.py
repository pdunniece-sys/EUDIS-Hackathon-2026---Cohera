"""Read a QuSpin QTFM raw serial log into arrays you can work with.

    from quspin import read_log

    log = read_log()            # the file in data/QuSpin/
    log.summary()
    flight = log.airborne()     # just the part where it was flying

What the raw file looks like
----------------------------
The magnetometer writes one line per sample down a serial port, and the
logger saves those lines exactly as they arrived. There is no header row
and no separators - the fields are marked by punctuation:

    223637639!53098.490_@066>222476s114a17.09b-47.85c-1025.88
        |         |        |     |    |   |
        |         |        |     |    |   +-- one auxiliary reading
        |         |        |     |    +------ signal level
        |         |        |     +----------- a second clock, milliseconds
        |         |        +----------------- sample counter, 0-249
        |         +-------------------------- TOTAL FIELD in nT
        +------------------------------------ timestamp in microseconds

The counter runs 0 to 249 and wraps, which is how you know the sampling
rate is 250 Hz: it resets once a second.

Only one auxiliary reading fits on each line, so the sensor rotates through
four of them in turn. Each therefore arrives at a quarter of the sample
rate, 62.5 Hz, on its own timebase:

    a b c   accelerometer      milli-g   (1000 = one g)
    i j k   gyroscope          degrees per second
    x y z   3-axis magnetometer microtesla
    t       sensor temperature  degrees C

Mixed into the same stream, once a second, are the position lines:

    999443896,GNSSFIX,440776320,-798143744,2026,06,08,19,53,52,276,4
        |        |        |          |       |                  |
        |        |        |          |       +-- Y M D h m s UTC |
        |        |        |          +---------- longitude x 10^7 |
        |        |        +--------------------- latitude  x 10^7 |
        |        +------------------------------ marker           |
        +--------------------------------------- same clock, us   |
                                     altitude above the ellipsoid -+

And occasionally a status line such as "+Done" or "#Auto Start Active".

What this parser changes, and what it leaves alone
--------------------------------------------------
Changed:
  - timestamps are converted from microseconds to seconds from the first
    sample, because absolute microseconds since power-on are unwieldy
  - a handful of lines print their timestamp as 0 rather than a real value.
    Their field reading is fine, so the timestamp is rebuilt by interpolating
    between the neighbours rather than throwing the sample away
  - latitude and longitude are divided by 10^7 to give degrees
  - the four auxiliary channels are separated out, each keeping its own
    timestamps rather than being forced onto the magnetometer's

Left alone:
  - no filtering, no smoothing, no gap filling, no outlier removal
  - the field values are exactly as recorded, in nT

A few things worth knowing before you use the numbers:
  - the field is TOTAL field, not an anomaly. Earth's background is still in
    it, which is why it reads about 53,000 nT rather than about zero
  - GNSS altitude is height above the WGS84 ellipsoid, not above sea level.
    In southern Ontario the two differ by roughly 36 m
  - the 3-axis magnetometer does NOT agree with the scalar reading (it is
    about 37 uT against 53 uT). It is an uncalibrated auxiliary sensor, useful
    for attitude, not as a measurement of the field
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "data" / "QuSpin" / "T2M0-00N6_Data2.txt"

SAMPLE_RE = re.compile(r"^(\d+)!([\d.]+)_@(\d+)>(\d+)s(\d+)(.*)$")
AUX_RE = re.compile(r"([a-z])(-?[\d.]+)")

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Channel:
    """One auxiliary channel on its own timebase.

    values is (n,) for temperature, (n, 3) for the three-axis channels.
    """

    time_s: np.ndarray
    values: np.ndarray
    name: str
    units: str

    def at(self, times_s: np.ndarray) -> np.ndarray:
        """Linearly interpolate this channel onto other timestamps."""
        times_s = np.asarray(times_s, dtype=float)
        if self.values.ndim == 1:
            return np.interp(times_s, self.time_s, self.values)
        columns = [np.interp(times_s, self.time_s, self.values[:, i]) for i in range(3)]
        return np.column_stack(columns)

    def __repr__(self) -> str:
        return f"<Channel {self.name} {self.values.shape} {self.units}>"


@dataclass
class Fixes:
    """The GNSS positions, one per second."""

    time_s: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray       # above the WGS84 ellipsoid
    utc: list[dt.datetime]

    def __len__(self) -> int:
        return len(self.time_s)

    def local_xy(self, origin: tuple[float, float] | None = None):
        """Positions as metres east and north of a local origin.

        Latitude and longitude are awkward for distances, so this flattens
        them onto a local plane. Accurate to well under a metre over a few
        kilometres, which is all this survey covers. Not for long distances.
        """
        lat0, lon0 = origin if origin else (self.lat.mean(), self.lon.mean())
        east = np.radians(self.lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
        north = np.radians(self.lat - lat0) * EARTH_RADIUS_M
        return east, north

    def speed_mps(self) -> np.ndarray:
        """Ground speed between consecutive fixes. One shorter than the fixes."""
        east, north = self.local_xy()
        return np.hypot(np.diff(east), np.diff(north)) / np.diff(self.time_s)

    def heading_deg(self) -> np.ndarray:
        """Direction of travel, degrees clockwise from north. One shorter."""
        east, north = self.local_xy()
        return np.degrees(np.arctan2(np.diff(east), np.diff(north))) % 360


@dataclass
class QuSpinLog:
    """Everything in one log file, split into its separate streams."""

    time_s: np.ndarray          # magnetometer timestamps, 250 Hz
    field_nt: np.ndarray        # total field
    signal: np.ndarray          # sensor signal level
    accel: Channel
    gyro: Channel
    mag_vector: Channel
    temperature: Channel
    fixes: Fixes
    status: list[str] = dc_field(default_factory=list)
    source: Path | None = None
    repaired_timestamps: int = 0

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def rate_hz(self) -> float:
        return len(self.time_s) / self.duration_s

    def position_at(self, times_s: np.ndarray):
        """Latitude, longitude and altitude interpolated onto given times.

        GNSS arrives once a second and the magnetometer runs at 250 Hz, so
        every magnetic sample between two fixes gets a position interpolated
        between them. Over a second at 5 m/s the drone moves about 5 m, so
        treat these as good to a few metres, not centimetres.
        """
        times_s = np.asarray(times_s, dtype=float)
        return (
            np.interp(times_s, self.fixes.time_s, self.fixes.lat),
            np.interp(times_s, self.fixes.time_s, self.fixes.lon),
            np.interp(times_s, self.fixes.time_s, self.fixes.alt_m),
        )

    def field_at_fixes(self) -> np.ndarray:
        """The field reading at the moment of each GNSS fix."""
        return np.interp(self.fixes.time_s, self.time_s, self.field_nt)

    def smoothed_field(self, cutoff_hz: float = 1.0) -> np.ndarray:
        """The field with the high-frequency noise filtered out.

        The raw 250 Hz trace carries about 5 nT of noise, and almost all of it
        sits at exactly 60 Hz - North American mains, picked up from power
        lines near the survey. It is environmental, not the aircraft: it does
        not correlate with heading or rotation rate.

        Geology cannot change faster than the aircraft flies over it. At 6 m/s
        with 50 m wide anomalies, everything real is below about 0.1 Hz, so a
        1 Hz cutoff throws away the interference and keeps the signal. This
        drops the noise from roughly 5 nT to well under 1 nT.
        """
        from scipy.signal import butter, filtfilt

        nyquist = self.rate_hz / 2
        if cutoff_hz >= nyquist:
            raise ValueError(f"cutoff must be below {nyquist:.0f} Hz")
        b, a = butter(4, cutoff_hz / nyquist, btype="low")
        return filtfilt(b, a, self.field_nt)

    def trim(self, start_s: float, end_s: float) -> "QuSpinLog":
        """A copy covering only the given time window."""
        keep = (self.time_s >= start_s) & (self.time_s <= end_s)
        fix_keep = (self.fixes.time_s >= start_s) & (self.fixes.time_s <= end_s)

        def cut(channel: Channel) -> Channel:
            k = (channel.time_s >= start_s) & (channel.time_s <= end_s)
            return Channel(channel.time_s[k], channel.values[k], channel.name, channel.units)

        return QuSpinLog(
            time_s=self.time_s[keep],
            field_nt=self.field_nt[keep],
            signal=self.signal[keep],
            accel=cut(self.accel),
            gyro=cut(self.gyro),
            mag_vector=cut(self.mag_vector),
            temperature=cut(self.temperature),
            fixes=Fixes(
                self.fixes.time_s[fix_keep],
                self.fixes.lat[fix_keep],
                self.fixes.lon[fix_keep],
                self.fixes.alt_m[fix_keep],
                [u for u, k in zip(self.fixes.utc, fix_keep) if k],
            ),
            status=self.status,
            source=self.source,
            repaired_timestamps=self.repaired_timestamps,
        )

    def airborne(self, climb_m: float = 10.0) -> "QuSpinLog":
        """A copy covering only the flight, with ground time cut off.

        Takeoff is found from the GNSS altitude: the ground level is taken as
        the lowest altitude seen, and the drone counts as flying once it is
        climb_m above that. This also removes the sensor's startup transients,
        which happen while it is still sitting on the ground.
        """
        ground = np.percentile(self.fixes.alt_m, 5)
        flying = self.fixes.alt_m > ground + climb_m
        if not flying.any():
            raise ValueError("No fixes are more than "
                             f"{climb_m} m above ground level - nothing looks airborne.")
        return self.trim(self.fixes.time_s[flying][0], self.fixes.time_s[flying][-1])

    def summary(self) -> None:
        """Print what is in this log."""
        print(f"{self.source.name if self.source else 'log'}\n")
        print("magnetometer")
        print(f"  samples        {len(self.field_nt):,}")
        print(f"  duration       {self.duration_s:.1f} s  ({self.duration_s/60:.1f} min)")
        print(f"  rate           {self.rate_hz:.1f} Hz")
        print(f"  total field    {self.field_nt.min():,.1f} to {self.field_nt.max():,.1f} nT")
        print(f"  mean / std     {self.field_nt.mean():,.1f} / {self.field_nt.std():.1f} nT")
        print(f"  signal level   {self.signal.min()} to {self.signal.max()}")
        if self.repaired_timestamps:
            print(f"  repaired       {self.repaired_timestamps} missing timestamps")

        print("\nauxiliary channels")
        for channel in (self.accel, self.gyro, self.mag_vector, self.temperature):
            rate = len(channel.time_s) / self.duration_s
            lo, hi = np.nanmin(channel.values), np.nanmax(channel.values)
            print(f"  {channel.name:12s} {len(channel.time_s):>7,} samples "
                  f"({rate:.1f} Hz)  {lo:9.2f} to {hi:9.2f} {channel.units}")

        print("\nposition")
        if len(self.fixes):
            east, north = self.fixes.local_xy()
            print(f"  fixes          {len(self.fixes):,}  "
                  f"({len(self.fixes)/self.duration_s:.2f} Hz)")
            print(f"  utc            {self.fixes.utc[0]}  to  {self.fixes.utc[-1]}")
            print(f"  latitude       {self.fixes.lat.min():.6f} to {self.fixes.lat.max():.6f}")
            print(f"  longitude      {self.fixes.lon.min():.6f} to {self.fixes.lon.max():.6f}")
            print(f"  area covered   {east.max()-east.min():.0f} m east-west"
                  f"  x  {north.max()-north.min():.0f} m north-south")
            print(f"  track length   {np.hypot(np.diff(east), np.diff(north)).sum()/1000:.2f} km")
            print(f"  altitude       {self.fixes.alt_m.min():.0f} to {self.fixes.alt_m.max():.0f} m"
                  f" above the ellipsoid")
            speed = self.fixes.speed_mps()
            print(f"  speed          {np.median(speed):.1f} m/s median, {speed.max():.1f} max")
        else:
            print("  no GNSS fixes in this window")

        if self.status:
            print("\nstatus lines")
            for line in dict.fromkeys(self.status):
                print(f"  {line}")


def read_log(path: str | Path = DEFAULT_LOG) -> QuSpinLog:
    """Parse a QuSpin serial log. See the module docstring for the format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            f"Put the QuSpin .txt log in {DEFAULT_LOG.parent}, or pass its path."
        )

    times, fields, signals = [], [], []
    aux: dict[str, list[tuple[int, float]]] = {}
    fix_rows, status = [], []

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "GNSSFIX" in line:
                fix_rows.append(line.split(","))
                continue

            match = SAMPLE_RE.match(line)
            if match is None:
                status.append(line)
                continue

            times.append(int(match.group(1)))
            fields.append(float(match.group(2)))
            signals.append(int(match.group(5)))
            for tag, value in AUX_RE.findall(match.group(6)):
                aux.setdefault(tag, []).append((len(fields) - 1, float(value)))

    if not fields:
        raise ValueError(f"{path} contains no magnetometer samples - is it a QuSpin log?")

    time_us = np.array(times, dtype=float)
    field_nt = np.array(fields)
    signal = np.array(signals)

    # A few lines print their timestamp as 0. The reading itself is good, so
    # rebuild the timestamp from the samples either side instead of discarding.
    missing = time_us == 0
    repaired = int(missing.sum())
    if repaired:
        index = np.arange(len(time_us))
        time_us[missing] = np.interp(index[missing], index[~missing], time_us[~missing])

    start_us = time_us[0]
    time_s = (time_us - start_us) / 1e6

    def build(tags: str, name: str, units: str) -> Channel:
        """Collect one auxiliary channel, using the timestamps of its own lines."""
        first = aux.get(tags[0], [])
        rows = np.array([i for i, _ in first], dtype=int)
        if len(tags) == 1:
            values = np.array([v for _, v in first])
        else:
            values = np.column_stack([[v for _, v in aux[t]] for t in tags])
        return Channel(time_s[rows], values, name, units)

    log = QuSpinLog(
        time_s=time_s,
        field_nt=field_nt,
        signal=signal,
        accel=build("abc", "accelerometer", "mg"),
        gyro=build("ijk", "gyroscope", "deg/s"),
        mag_vector=build("xyz", "mag vector", "uT"),
        temperature=build("t", "temperature", "C"),
        fixes=_read_fixes(fix_rows, start_us),
        status=status,
        source=path,
        repaired_timestamps=repaired,
    )
    return log


def _read_fixes(rows: list[list[str]], start_us: float) -> Fixes:
    """Turn the GNSSFIX lines into arrays, on the same clock as the samples."""
    if not rows:
        return Fixes(np.array([]), np.array([]), np.array([]), np.array([]), [])

    time_s, lat, lon, alt, utc = [], [], [], [], []
    for row in rows:
        try:
            time_s.append((int(row[0]) - start_us) / 1e6)
            lat.append(int(row[2]) / 1e7)      # stored as degrees x 10^7
            lon.append(int(row[3]) / 1e7)
            utc.append(dt.datetime(*(int(row[i]) for i in range(4, 10))))
            alt.append(float(row[10]))
        except (ValueError, IndexError):
            continue  # a truncated line, e.g. if logging stopped mid-write

    return Fixes(np.array(time_s), np.array(lat), np.array(lon), np.array(alt), utc)
