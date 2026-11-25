"""Offline: candidate counts per reference type, empty scene and at the three world boxes.

Compares straight-line waypoints with the file under test. Synthetic points, stationary pose:
says nothing about tracking or real scans. Run from the workspace root.
"""
import importlib.util,numpy as np,math,copy
from scipy.interpolate import CubicSpline
spec=importlib.util.spec_from_file_location('fp','src/frenet_planner_optimized.py');fp=importlib.util.module_from_spec(spec);spec.loader.exec_module(fp)
xy=np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv',delimiter=',')[:,:2]
print('dup end',np.hypot(*(xy[0]-xy[-1])),'zero segs',(np.hypot(*np.diff(xy,axis=0).T)<1e-6).sum())
orig=fp.resample_closed
def spline(xy,step):
    xy=np.asarray(xy,float)[:,:2]
    if np.hypot(*(xy[0]-xy[-1]))<1e-6: xy=xy[:-1]
    c=np.vstack([xy,xy[:1]]);s=np.r_[0,np.cumsum(np.hypot(*np.diff(c,axis=0).T))]
    cs=CubicSpline(s,c,bc_type='periodic');n=max(int(round(s[-1]/step)),8);sn=np.linspace(0,s[-1],n,endpoint=False)
    p=cs(sn);d1=cs(sn,1);d2=cs(sn,2);ka=(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.hypot(*d1.T)**3
    spline.ka=ka;return p,s[-1]
OBS=[(5,-.0878,.15,.4),(-3,3.054,.15,.4),(-8,1.145,.15,.5)]
def box_pts():
    out=[]
    for cx,cy,hx,hy in OBS:
        for t in np.arange(-hx,hx+1e-6,.025):
            out+= [(cx+t,cy-hy),(cx+t,cy+hy)]
        for t in np.arange(-hy,hy+1e-6,.025):
            out+= [(cx-hx,cy+t),(cx+hx,cy+t)]
    return np.array(out)
B=box_pts()
def run(name,fn):
    fp.resample_closed=fn
    P=fp.FrenetPlanner(xy)
    n_free=[];rows=[]
    for i in range(0,P.n,10):
        pose=(P.ref[i,0],P.ref[i,1],P.theta[i])
        best,c,_=P.lattice(i,0.,0.,pose,np.zeros((0,2)),free=False,block_s=1.0)
        n_free.append(len(c))
    print(name,'empty scene: windows with no candidate',round(np.mean(np.array(n_free)==0),3),'of',len(n_free),'median cand',np.median(n_free))
    for k,(cx,cy,hx,hy) in enumerate(OBS):
        j=int(np.argmin(np.hypot(*(P.ref-(cx,cy)).T)))
        res=[]
        for back in (3.,2.,1.5,1.):
            i=(j-int(back/P.ds))%P.n
            pose=(P.ref[i,0],P.ref[i,1],P.theta[i])
            pts=fp.to_base(B,pose);r=np.hypot(*pts.T);pts=pts[(r<5)&(r>.12)]
            best,c,d1=P.lattice(i,0.,0.,pose,pts,free=False,block_s=back-.3)
            res.append(f'{back}m:n={len(c)},d1={d1:+.2f},rej={P.rej["curv"]}/{P.rej["back"]}/{P.rej["coll"]}')
        print(' ',name,'o%d'%(k+1),' '.join(res))
run('polyline',orig)
run('spline',spline)
ka=spline.ka;print('spline analytic |k| med',round(np.median(abs(ka)),3),'p95',round(np.percentile(abs(ka),95),3),'max',round(abs(ka).max(),3),'limit',round(.9*fp.MAX_CURV,3))
P=fp.FrenetPlanner(xy);po,_=orig(xy,.05)
print('max distance spline ref to polyline',round(max(np.hypot(*(po-p).T).min() for p in P.ref),4))
