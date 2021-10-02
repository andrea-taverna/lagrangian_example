import pandas as pd
import mizani.formatters
from plotnine import *

from generic.series_dict import series_dict_to_array


def bounds(kpis: pd.DataFrame) -> ggplot:
    data = pd.melt(
        kpis[
            [
                "iteration",
                "dual_algorithm",
                "best_dual_bound",
                "best_primal_bound",
                "best_master_bound",
            ]
        ],
        id_vars=["iteration", "dual_algorithm"],
    )

    p = (
        ggplot(data, aes(x="iteration", y="value"))
        + geom_line(aes(color="variable"), size=1)
        + scale_color_discrete(
            breaks=["best_dual_bound", "best_primal_bound", "best_master_bound"],
            labels=["Best Dual", "Best Primal", "Best Master Bound"],
        )
        + ggtitle("Bounds")
    )

    return p


def gap(kpis: pd.DataFrame) -> ggplot:
    data = pd.melt(
        kpis[["iteration", "dual_algorithm", "optimality_gap"]],
        id_vars=["iteration", "dual_algorithm"],
    )

    p = (
        ggplot(data, aes(x="iteration", y="value"))
        + geom_line(aes(color="variable"), size=1)
        + scale_y_continuous(labels=mizani.formatters.percent_format())
        + scale_color_discrete(guide=False)
        + ggtitle("Optimality gap %")
    )

    return p


def infeasibility(kpis: pd.DataFrame) -> ggplot:
    kpis = kpis.copy()
    kpis["v_infeasibilities"] = kpis["relaxation_primal"].transform(
        lambda h: series_dict_to_array(**h.infeasibilities)
    )

    infeasibility = kpis[["iteration", "v_infeasibilities"]]
    infeasibility["infeasibility"] = infeasibility["v_infeasibilities"].map(lambda v: v.sum())

    return ggplot(infeasibility, aes("iteration", "infeasibility")) + geom_step(size=1.5) \
        + labs(x="Iteration #", y="Total infeasibility [MWh]") + ggtitle("Infeasibility by iteration")