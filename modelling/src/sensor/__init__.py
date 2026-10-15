"""What a magnetometer on the drone would actually read.

    from sensor import FieldModel, ScalarOPM, VectorMagnetometer
    from tellus import MagneticGrid

    field = FieldModel(grid=MagneticGrid("a7_tmi"))
    opm = ScalarOPM()
    readings = opm.emulate(truth, field, seed=0)
    anomaly = opm.to_anomaly(readings, truth, field)   # what the filter compares

Two sensors: a QTFM-like scalar OPM (the quantum sensor we are modelling) and
an RM3100-like vector magnetometer (the cheap one already on the aircraft,
here so the comparison can be made rather than asserted).

Read field.py first - it explains how the four parts of the field are
assembled, and why they are added as vectors before the magnitude is taken.
"""

from sensor.field import DiurnalModel, FieldModel, NoDiurnal
from sensor.interference import (
    ArrayInterference,
    NoInterference,
    ParametricInterference,
)
from sensor.opm import ScalarOPM, default_heading_error
from sensor.timing import decimate
from sensor.types import FieldSamples, Readings
from sensor.vector import VectorMagnetometer

__all__ = [
    "FieldModel", "DiurnalModel", "NoDiurnal",
    "ScalarOPM", "VectorMagnetometer", "default_heading_error",
    "NoInterference", "ParametricInterference", "ArrayInterference",
    "FieldSamples", "Readings", "decimate",
]
