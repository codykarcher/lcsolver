import numpy as np
from scipy.optimize import least_squares
X=np.load('tp_X.npy'); e=np.load('tp_e.npy')
lamt,lams,eo,es,clhf,bhb,bohf,dz,rcls,rclt,CL=[X[:,i] for i in range(11)]
gs=lams*rcls; gt=lamt*rclt
F=np.stack([gs,gt,eo,es,clhf,bhb,dz,CL],axis=1)
lF=np.log(F); ld=np.log(1.0/e-1.0)
def model(p):
    c1,c2=p[0],p[9]
    e1=p[1:9]; e2=p[10:18]
    t1=c1+lF@e1; t2=c2+lF@e2
    return np.logaddexp(t1,t2)
def resid(p): return model(p)-ld
p0=np.zeros(18)
p0[0]=-4.2; p0[1:9]=[-1.4,-1.0,0.5,-1.1,0.05,-0.09,0.11,-0.08]
p0[9]=-8.0; p0[10:18]=[0.0,2.0,0,0,0,0,0,0]
r=least_squares(resid,p0,max_nfev=20000)
d=np.exp(model(r.x)); efit=1/(1+d)
err=np.abs(efit-e)/e
print("2-term posynomial: max %.2f%%  rms %.2f%%"%(100*err.max(),100*(err**2).mean()**0.5))
names=['gam_s','gam_t','eta_o','eta_s','CLhfrac','bh_b','dz','CL']
print("K1 = %.6g"%np.exp(r.x[0])); [print(f"  {n:8s} {c:+.4f}") for n,c in zip(names,r.x[1:9])]
print("K2 = %.6g"%np.exp(r.x[9])); [print(f"  {n:8s} {c:+.4f}") for n,c in zip(names,r.x[10:18])]
np.save('tp_coef.npy',r.x)
