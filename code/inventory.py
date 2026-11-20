"""Regenerate mapped directional footprints and excluded-event areas from class histories."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(hist, data, out):
    x = np.array([list(s) for s in hist.classes_2000_to_2025],dtype=np.uint8)
    area = hist.county_area_m2.to_numpy()/1e6
    n = hist.native_cells.to_numpy()
    flags={3:np.zeros((len(x),2),bool),5:np.zeros((len(x),2),bool)}
    events=[]
    for t in range(5,22):
        a,b=x[:,t-1],x[:,t]
        transition=(a!=b)
        controls={}
        for k,flag in flags.items():
            left=(x[:,t-k:t]==a[:,None]).all(axis=1)
            right=(x[:,t:t+k]==b[:,None]).all(axis=1)
            controls[k]=[left,right]
            event=transition & left & right
            flag[:,0]|=event&(b==2)
            flag[:,1]|=event&(a==2)
        left,right=controls[5]
        base=transition & controls[3][0]&controls[3][1]
        for d,m in [("gain",b==2),("loss",a==2)]:
            for category,mask in [("source_only",~left&right),("destination_only",left&~right),("both_sides",~left&~right)]:
                cells=base&m&mask
                events.append((d,category,n[cells].sum(),area[cells].sum()))
    rows=[]
    for k,flag in flags.items():
        g,l=flag.T
        rows.append(dict(rule="three_year_K3" if k==3 else "five_year_K5",gain_km2=area[g].sum(),loss_km2=area[l].sum(),
                         both_km2=area[g&l].sum(),union_km2=area[g|l].sum(),gain_units=n[g].sum(),loss_units=n[l].sum(),both_units=n[g&l].sum()))
    res=pd.DataFrame(rows)
    out.mkdir(exist_ok=True,parents=True)
    res.to_csv(out/"county_footprints.csv",index=False)
    expect=pd.read_csv(data/"direction_footprint.csv").set_index("rule").loc[res.rule]
    pd.testing.assert_frame_equal(expect.reset_index(),res,atol=1e-8,rtol=1e-12,check_dtype=False)
    e=pd.DataFrame(events,columns=["direction","category","n_events","area_km2"]).groupby(["direction","category"]).sum()
    e.to_csv(out/"excluded_event_areas.csv")
    old=pd.read_csv(data/"event_exclusion_cause_summary.csv").set_index(["direction","category"]).sort_index()
    pd.testing.assert_frame_equal(e,old[e.columns],atol=1e-8,rtol=1e-12,check_dtype=False)
    return {"clcd_cells":int(n.sum()),"unique_sequences":len(x),"excluded_events":int(e.n_events.sum())}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=Path(__file__).resolve().parents[1]/"data"
    h=pd.read_csv(data/"class_histories.csv",dtype={"classes_2000_to_2025":str})
    print(summarize(h,data,args.output))
