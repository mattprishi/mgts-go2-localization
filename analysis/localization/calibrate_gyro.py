"""Estimate the rigid gyro axes and bias using the separate mapping drive."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def calibrate():
    folder=Path('runs/imu_comparison')
    with np.load(folder/'mapping.npz') as d:
        it, od=d['imu_ts'], d['odom_ts']
        stamps=np.arange(max(it[0],od[0]), min(it[-1],od[-1]), 72_000_000)
        o=Slerp((od-od[0])*1e-9,Rotation.from_quat(d['quats']))((stamps-od[0])*1e-9)
        ref=(o[:-1].inv()*o[1:]).as_rotvec()/0.072
        gyro=np.array([d['rates'][i:j].mean(axis=0) for i,j in zip(np.searchsorted(it,stamps[:-1]),np.searchsorted(it,stamps[1:]))])
    split=len(ref)*2//3
    x,y=gyro[:split],ref[:split]
    r,_=Rotation.align_vectors(y-y.mean(0),x-x.mean(0))
    bias=x.mean(0)-r.inv().apply(y.mean(0))
    pred=r.apply(gyro-bias)
    report=dict(body_from_imu=r.as_matrix().tolist(),bias=bias.tolist(),
        mapping_fit_windows=split,mapping_check_windows=len(ref)-split,
        mapping_check_correlations=[np.corrcoef(pred[split:,i],ref[split:,i])[0,1] for i in range(3)],
        mapping_check_rmse_rad_s=np.mean((pred[split:]-ref[split:])**2,axis=0)**.5)
    report['mapping_check_rmse_rad_s']=report['mapping_check_rmse_rad_s'].tolist()
    (folder/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__': calibrate()
