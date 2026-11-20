"""Reproduce the stratified summaries from unit, prediction and QC frequency tables."""
from pathlib import Path
import numpy as np
import pandas as pd


def weighted_quantiles(values, weights, probabilities):
    """Left inverse of the area-weighted empirical distribution function."""
    order = np.argsort(values, kind="stable")
    x = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(w)) and np.all(w > 0)
    return x[np.searchsorted(np.cumsum(w), np.array(probabilities) * w.sum(), side="left")]


def weighted_fit(x, y, w):
    x, y, w = map(np.asarray, (x, y, w))
    xbar, ybar = np.average(x, weights=w), np.average(y, weights=w)
    xc, yc = x - xbar, y - ybar
    xx, xy, yy = (np.sum(w*z) for z in (xc*xc, xc*yc, yc*yc))
    slope = xy / xx
    return xy / np.sqrt(xx * yy), slope, ybar - slope * xbar


def derive(units, oof, native, out):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    rows = []
    for scale, cells in units.groupby("grid_m"):
        area = cells.A_g
        for d in ["gain", "loss"]:
            x, y = cells["P3_" + d], cells["P5_" + d]
            pos3, pos5 = x > 1e-10, y > 1e-10
            assert (x - y).min() > -1e-10 and not ((~pos3) & pos5).any()
            both = pos3 & pos5
            r, slope, intercept = weighted_fit(x[both], y[both], area[both])
            removed = area * (x - y)
            row = dict(grid_m=scale, direction=d, cells=len(cells),
                       r_all=weighted_fit(x, y, area)[0], r_both_positive=r,
                       slope_both_positive=slope, intercept_both_positive=intercept,
                       removed_km2=removed.sum()/1e6,
                       removed_in_both_positive_km2=removed[both].sum()/1e6,
                       pct_removed_in_both_positive=100*removed[both].sum()/removed.sum())
            for name, regime in [("both_zero", ~pos3 & ~pos5), ("only_three", pos3 & ~pos5), ("both_positive", both)]:
                row[name + "_n"] = int(regime.sum())
                row[name + "_pct_cells"] = 100 * regime.mean()
                row[name + "_pct_area"] = 100 * area[regime].sum() / area.sum()
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "zero_regimes_correlations.csv", index=False)
    rows = []
    for window, hist in native.groupby("window", sort=True):
        q1, med, q3 = weighted_quantiles(hist.qc_percent, hist.area_m2, [.25,.5,.75])
        rows.append(dict(window=window,n_native_pixels=int(hist.native_pixels.sum()),
                         support_km2=hist.area_m2.sum()/1e6,
                         weighted_mean_pct=np.average(hist.qc_percent,weights=hist.area_m2),
                         q25_pct=q1,median_pct=med,q75_pct=q3,iqr_pct_points=q3-q1))
    pd.DataFrame(rows).to_csv(out / "native_qc_distribution.csv", index=False)
    cells = units.loc[units.grid_m == 1000, ["unit_id","A_g","Y1","QC1"]].copy()
    cut = weighted_quantiles(cells.QC1, cells.A_g, [.5])[0]
    cells["qc_group"] = np.where(cells.QC1 <= cut,"lower","higher")
    p = oof.loc[(oof.group=="main") & (oof.model.isin(["M0","Mnet","M3","M5"])) &
                (oof.kset.isin(["LM","B"])), ["unit_id","family","model","pred"]]
    p = p.merge(cells,on="unit_id",validate="many_to_one")
    assert len(p) == len(cells)*8 and not p.duplicated(["family","model","unit_id"]).any()
    rows = []
    for (quality,family), z in p.groupby(["qc_group","family"]):
        scores = {m:np.sqrt(np.average((a.Y1-a.pred)**2, weights=a.A_g)) for m,a in z.groupby("model")}
        u = z[z.model=="M3"]
        rows.append(dict(later_QC_group=quality,family=family,n_cells=len(u),support_km2=u.A_g.sum()/1e6,
                         QC1_weighted_median_split_pct=cut,QC1_min_pct=u.QC1.min(),QC1_max_pct=u.QC1.max(),
                         **{"RMSE_"+k:v for k,v in scores.items()},delta_M3_minus_M5=scores["M3"]-scores["M5"]))
    pd.DataFrame(rows).to_csv(out / "main_qc_group_errors.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    derive(pd.read_csv(args.data / "units.csv"), pd.read_csv(args.data / "oof_predictions.csv"),
           pd.read_csv(args.data / "native_qc_frequencies.csv"), args.output)
