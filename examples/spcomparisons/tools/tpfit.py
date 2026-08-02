import numpy as np, subprocess
rng=np.random.default_rng(7)
n=420
# ranges: lamt, lams, eta_o, eta_s, CLhfrac(=CLsurfsp2/CL, SIGNED->use magnitude dn), bh/b, boh/bh, dz(m), rcls, rclt, CL
lo=np.array([0.10,0.60,0.06,0.26,0.002,0.30,0.05,0.5,1.00,0.70,0.40])
hi=np.array([0.40,1.00,0.16,0.40,0.030,0.45,0.40,5.0,1.30,1.20,0.80])
X=lo+ (hi-lo)*rng.uniform(size=(n,len(lo)))
lines=[]
S=120.0
for r in X:
    lamt,lams,eo,es,clhf,bhb,bohf,dz,rcls,rclt,CL=r
    b=np.sqrt(S*10.0)  # AR 10 baseline; e is scale-invariant given ratios
    bs=es*b; bo=eo*b; bh=bhb*b; boh=bohf*bh
    CLh=-clhf*CL  # download
    lines.append(f"{S} {b} {b} {bs} {bo} 0.0 {lams} {lamt} {rcls} {rclt} -0.3 {CL} {CLh} {bh} {boh} {dz} 0.30")
p=subprocess.run(["./tpsweep"],input="\n".join(lines)+"\n",capture_output=True,text=True)
out=np.array([[float(v) for v in l.split()] for l in p.stdout.strip().splitlines()])
e=out[:,0]
ok=(e>0.5)&(e<1.0)
X,e=X[ok],e[ok]
print(f"{ok.sum()} valid points, e range {e.min():.4f}-{e.max():.4f}")
# monomial fit: log(1/e-1) = c0 + sum ai log xi   (xi all positive)
Y=np.log(1.0/e-1.0)
feat=np.log(X)
A=np.hstack([np.ones((len(e),1)),feat])
coef,res,_,_=np.linalg.lstsq(A,Y,rcond=None)
pred=A@coef
efit=1.0/(1.0+np.exp(pred))
err=np.abs(efit-e)/e
print("monomial fit: max err %.3f%%  rms %.3f%%"%(100*err.max(),100*(err**2).mean()**0.5))
names=['lamt','lams','eta_o','eta_s','CLhfrac','bh/b','boh/bh','dz','rcls','rclt','CL']
print("c0 = %.5f"%coef[0])
for nm,c in zip(names,coef[1:]): print(f"  {nm:8s} {c:+.5f}")
np.save('tp_X.npy',X); np.save('tp_e.npy',e)
