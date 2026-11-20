"""Recalculate pooled errors, fixed-prediction block intervals and QC comparisons."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from core_evidence import derive


def read(name):
    return pd.read_csv(data / (name + ".csv"), float_precision="round_trip")


def rmse(frame):
    return np.sqrt(np.average((frame.Y1 - frame.pred)**2, weights=frame.A_g))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    data = Path(__file__).resolve().parents[1] / "data"
    units, roles, pred = read("units"), read("fold_roles"), read("oof_predictions")
    held = roles.loc[roles.role=="validation",["unit_id","fold","group"]]
    assert not held.duplicated(["unit_id","group"]).any()
    assert not pred.duplicated(["unit_id","group","family","kset","model"]).any()
    fit = pred.merge(held,on=["unit_id","group","fold"],validate="many_to_one")
    fit = fit.merge(units[["unit_id","Y1","A_g"]],on="unit_id",validate="many_to_one")
    assert len(fit)==len(pred) and np.isfinite(fit[["Y1","pred","A_g"]]).all().all() and (fit.A_g>0).all()
    factor = ["group","family","kset","model"]
    scores = fit.groupby(factor).apply(rmse, include_groups=False).rename("rmse").reset_index()
    scores.to_csv(out/"all_pooled_scores.csv",index=False)
    cmp = read("published_scores").merge(scores,on=factor,validate="one_to_one")
    assert len(cmp)==48
    differences = {"scores48":float(abs(cmp.rmse_x-cmp.rmse_y).max())}
    basis = scores.loc[(scores.group=="main") & (scores.family=="GAM")].pivot(index="kset",columns="model",values="rmse")
    E = pd.DataFrame({"rmse_E":basis.E,"rmse_M3":basis.M3,"delta_E_minus_M3":basis.E-basis.M3}).reset_index()
    E.to_csv(out/"elevation_scores.csv",index=False)
    e = E.merge(read("published_E_scores"),on="kset",validate="one_to_one")
    differences["E6"]=max(float(abs(e[col+"_x"]-e[col+"_y"]).max()) for col in ["rmse_E","rmse_M3","delta_E_minus_M3"])
    assert max(differences.values()) < 5.1e-7
    paired=fit.loc[(fit.group=="main") & fit.model.isin(["M3","M5"]) & fit.kset.isin(["LM","B"])].copy()
    paired["sse"] = paired.A_g*(paired.Y1-paired.pred)**2
    actual=read("stability_block_sums")
    draws=[]
    block_diff=[]
    for length,group in [(8000,"main"),(4000,"block4km")]:
        blk = roles.loc[(roles.group==group)&(roles.role=="validation"),["unit_id","block"]]
        mapped = paired.merge(blk,on="unit_id",validate="many_to_one")
        area = mapped.drop_duplicates("unit_id").groupby("block").A_g.sum()
        sums = mapped.groupby(["block","family","model"]).sse.sum().unstack(["family","model"])
        frozen = actual.loc[actual.partition_m==length].sort_values("draw_index").set_index("block")
        assert set(frozen.index)==set(sums.index) and set(area.index)==set(sums.index)
        a = area.loc[frozen.index].to_numpy()
        assert np.allclose(a,frozen.a_b,rtol=1e-12,atol=1e-7)
        m=len(a)
        rng=np.random.Generator(np.random.PCG64(20261103))
        mult=np.stack([np.bincount(rng.integers(0,m,size=m),minlength=m) for _ in range(5000)])
        mass = mult@a
        assert (mult.sum(axis=1)==m).all() and (mass>0).all()
        for family in ["LM","GAM"]:
            s3 = sums[(family,"M3")].loc[frozen.index].to_numpy()
            s5 = sums[(family,"M5")].loc[frozen.index].to_numpy()
            for model,vec in [(3,s3),(5,s5)]:
                old=frozen[f"{family}_s{model}"].to_numpy()
                block_diff.append(float(np.max(abs(vec-old)/np.maximum(1,old))))
            delta = np.sqrt((mult@s3)/mass)-np.sqrt((mult@s5)/mass)
            lo,med,hi = np.quantile(delta,[.025,.5,.975])
            draws.append(dict(partition_m=length,family=family,blocks=m,q2_5=lo,median=med,q97_5=hi,
                              positive_draws=int(sum(delta>0)),replicates=len(delta)))
    res=pd.DataFrame(draws)
    res.to_csv(out/"conditional_block_quantiles.csv",index=False)
    check=res.merge(read("published_stability"),on=["partition_m","family","blocks","positive_draws","replicates"],validate="one_to_one")
    assert len(check)==4
    differences["conditional_quantiles"]=max(float(abs(check[v+"_x"]-check[v+"_y"]).max()) for v in ["q2_5","median","q97_5"])
    differences["max_relative_block_SSE"]=max(block_diff)
    assert differences["conditional_quantiles"]<1e-9 and max(block_diff)<1e-9
    a = read("aligned_predictions")
    assert not a.duplicated(["unit_id","family","model"]).any()
    ac = read("aligned_covariates")
    assert ac.unit_id.is_unique and set(ac.unit_id)==set(units.unit_id)
    for name in ["P3_gain","P3_loss","P5_gain","P5_loss","forest_net","tmean_change_c","precip_change_mm"]:
        assert np.isfinite(ac[name]).all()
    ax = a.merge(held,on=["unit_id","fold","group"],validate="many_to_one").merge(
        units[["unit_id","grid_m","Y1","A_g"]],on=["unit_id","grid_m"],validate="many_to_one")
    assert len(ax)==len(a)==24948
    aligned_scores = ax.groupby(["grid_m","family","model"]).apply(rmse,include_groups=False).rename("RMSE").reset_index()
    aligned_scores.to_csv(out/"aligned_scores.csv",index=False)
    axc = aligned_scores.merge(read("published_aligned_scores"),on=["grid_m","family","model"],validate="one_to_one")
    differences["aligned_scores12"]=float(abs(axc.RMSE_x-axc.RMSE_y).max())
    assert len(axc)==12 and differences["aligned_scores12"]<1e-9
    derive(units,pred,read("native_qc_frequencies"),out)
    for name in ["zero_regimes_correlations","native_qc_distribution","main_qc_group_errors"]:
        expected,calculated=read(name),pd.read_csv(out/(name+".csv"))
        pd.testing.assert_frame_equal(expected,calculated,atol=1e-8,rtol=1e-11)
    print(json.dumps({"verified":True,"units":len(units),"role_rows":len(roles),"predictions":len(pred),
                      "pooled_scores":len(scores),"aligned_predictions":len(a),"aligned_scores":len(axc),
                      "conditional_ranges":len(check),"errors":differences},indent=2))
