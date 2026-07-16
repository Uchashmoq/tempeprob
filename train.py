"""Low-level Python bridge to R's ``ensembleMOS`` package.

Forecast and observation preprocessing intentionally does not live here.  The
caller is responsible for constructing the R ``ensembleData`` object expected
by ``ensembleMOS::ensembleMOS``.
"""

from typing import Any

from rpy2 import rinterface
from rpy2.robjects.packages import importr


def ensemble_mos(
    ensemble_data: Any,
    training_days: int,
    *,
    consecutive: bool = False,
    dates: Any = rinterface.NULL,
    control: Any | None = None,
    warm_start: bool = False,
    model: str = "normal",
    exchangeable: Any = rinterface.NULL,
) -> Any:
    """Call ``ensembleMOS::ensembleMOS`` and return its raw R result.

    ``ensemble_data`` must be an R object inheriting from ``ensembleData``.
    No Python/R data-frame conversion or weather-data preprocessing is done in
    this low-level wrapper.  ``model="normal"`` is the usual EMOS model for
    temperature forecasts.
    """
    ensemble_mos_package = importr("ensembleMOS")
    arguments = {
        "trainingDays": training_days,
        "consecutive": consecutive,
        "dates": dates,
        "warmStart": warm_start,
        "model": model,
        "exchangeable": exchangeable,
    }
    if control is not None and control is not rinterface.NULL:
        arguments["control"] = control

    return ensemble_mos_package.ensembleMOS(ensemble_data, **arguments)
