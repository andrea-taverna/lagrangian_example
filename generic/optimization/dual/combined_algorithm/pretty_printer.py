import collections

from generic.optimization.dual.combined_algorithm.kpi_collector import KpiCollector

columns_format = collections.OrderedDict([
    ("iteration", ("It.", ">4d", ">4")),
    ("dual_algorithm", ("Dual Algorithm", ">18", ">18")),
    ("dual_bound", ("Current Dual", ">15.5g", ">15")),
    #("primal_bound", ("Current Primal", ">15.5g", ">15")),
    ("master_bound", ("Current Master", ">18.5g", ">18")),
    ("best_dual_bound", ("Best Dual", ">15.5g", ">15")),
    ("best_primal_bound", ("Best Primal", ">15.5g", ">15")),
    ("best_master_bound", ("Best Master", ">18.5g", ">18")),
    ("optimality_gap", ("Optimality Gap%", ">15.2%", ">15")),
    ("cutting_plane_gap", ("CP Gap%", ">15.2%", ">15")),
])


def header():
    formats = "".join(["|{:" + fmt[2] + "}" for fmt in columns_format.values()])
    return formats.format(*[fmt[0] for fmt in columns_format.values()])


def row(kpi_collector: KpiCollector):
    formats = "".join(["|{:" +  fmt[1] + "}" for fmt in columns_format.values()])
    return formats.format(*[kpi_collector.rows[k][-1] for k in columns_format.keys()])
