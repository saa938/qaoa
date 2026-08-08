"""
One figure for Ian pulling the shell results together.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import maqaoa_core as M

DATA = "../ashaydata/AshayMAQAOAData"
OUT = "shell_summary.png"
FOLDER = {10: "AshayGlobalMinimums_0", 11: "AshayGlobalMinimums_1",
          12: "AshayGlobalMinimums_2", 13: "AshayGlobalMinimums_3"}

def radii(P):
    G = M.geodesic_vec(P, np.zeros(P.shape[1]))
    return np.sqrt((G ** 2).sum(1))

# Radius against edge count, with the half-integer levels the squared radius lands on.
def panel_edges(a, shell):
    a.scatter([s["m"] for s in shell], [s["r_min_after_slide"] for s in shell],
              s=70, color="tab:blue", zorder=3)
    for s in shell:
        a.annotate(str(s["row"]), (s["m"], s["r_min_after_slide"]),
                   textcoords="offset points", xytext=(6, 4), fontsize=8)
    for k, c in [(11.5, "tab:grey"), (12.0, "tab:green"), (14.0, "tab:orange")]:
        a.axhline(np.sqrt(k) * np.pi / 4, ls="--", lw=1, color=c,
                  label=r"$r^2=%.1f(\pi/4)^2$" % k)
    a.set_xlabel("number of edges m")
    a.set_ylabel("inner shell radius")
    a.set_title("Shell radius is flat in edge count (p=1, unit weights)")
    a.set_ylim(2.3, 3.2)

# My shell against the radius distribution of his harvest, on the two graphs where
# his global optimizer never gets down to my radius at all.
def panel_harvest(a, shell):
    byrow = {s["row"]: s for s in shell}
    for row, col in [(11, "tab:red"), (12, "tab:purple")]:
        d = np.load(DATA + "/" + FOLDER[row] + "/er_graph_minima.npz", allow_pickle=True)
        rr = radii(M.wrap_pi(d["minima"]))
        a.hist(rr, bins=50, alpha=0.45, color=col,
               label="Ian row %d (n=%d)" % (row, len(rr)))
        a.axvline(byrow[row]["r_min_after_slide"], color=col, lw=2.2,
                  label="my shell row %d = %.3f" % (row, byrow[row]["r_min_after_slide"]))
    a.set_xlabel("geodesic radius from origin")
    a.set_ylabel("count")
    a.set_title("His harvest never reaches the inner shell (rows 11, 12)")

def panel_layers_weights(a, lw):
    for row, col in [(10, "tab:blue"), (13, "tab:green")]:
        L = [x for x in lw if x["kind"] == "layers" and x["row"] == row]
        if L:
            a.plot([x["p"] for x in L], [x["shell_radius"] for x in L],
                   "o-", color=col, label="row %d, layers" % row)
        Wt = [x for x in lw if x["kind"] == "weighted" and x["row"] == row]
        if Wt:
            a.scatter([1.15] * len(Wt), [x["shell_radius"] for x in Wt],
                      marker="x", s=70, color=col, label="row %d, random weights" % row)
    a.set_xlabel("layers p   (weighted points offset for visibility)")
    a.set_ylabel("shell radius")
    a.set_xticks([1, 2])
    a.set_title("Radius does not grow with layers or with weights")

def panel_degeneracy(a, deg):
    xs = np.arange(len(deg))
    a.bar(xs - 0.2, [d.get("n_distinct_at_shell", 0) for d in deg], 0.4,
          label="minima at shell radius", color="tab:blue")
    a.bar(xs + 0.2, [d.get("n_tied_at_max_fraction", 0) for d in deg], 0.4,
          label="still tied after max positive fraction", color="tab:red")
    a.set_xticks(xs)
    a.set_xticklabels(["row %d" % d["row"] for d in deg])
    a.set_ylabel("count")
    a.set_title("Largest positive fraction only halves the set, never picks one")

def main():
    shell = json.load(open("shell_radius.json"))
    deg = json.load(open("shell_degeneracy.json"))
    lw = json.load(open("layers_weights.json"))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    panel_edges(ax[0, 0], shell)
    panel_harvest(ax[0, 1], shell)
    panel_layers_weights(ax[1, 0], lw)
    panel_degeneracy(ax[1, 1], deg)
    for a in ax.ravel():
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    print("wrote %s" % OUT)

if __name__ == "__main__":
    main()
