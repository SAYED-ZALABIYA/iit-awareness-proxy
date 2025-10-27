# iit_integration_proxy_demo.py  — CLEAN

import math, random, argparse, csv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ========= Utils =========
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def make_xor(n=512, noise=0.05, seed=0):
    rng=np.random.RandomState(seed)
    X=rng.rand(n,2); y=((X[:,0]>.5) ^ (X[:,1]>.5)).astype(np.int64)
    X=X+rng.normal(0,noise,X.shape); X=(X-X.mean(0))/X.std(0)
    return X.astype(np.float32), y

def make_two_moons(n=1000, noise=0.15, seed=0):
    rng=np.random.RandomState(seed)
    ang=rng.rand(n)*math.pi; r=1.0+rng.normal(0,noise,n)
    x1,y1=r*np.cos(ang), r*np.sin(ang)
    x2,y2=1.0-r*np.cos(ang)+0.5, -r*np.sin(ang)
    X=np.vstack([np.stack([x1,y1],1), np.stack([x2,y2],1)]).astype(np.float32)
    y=np.array([0]*n+[1]*n, dtype=np.int64)
    X=(X-X.mean(0))/X.std(0); return X,y

def stable_cov(X,eps=1e-6):
    Xc=X-X.mean(0,keepdims=True); C=(Xc.T@Xc)/max(1,X.shape[0]-1)
    return C+np.eye(C.shape[0])*eps

def gaussian_total_correlation(X,eps=1e-6):
    C=stable_cov(X,eps); D=np.diag(np.diag(C))
    s1,d1=np.linalg.slogdet(D); s2,d2=np.linalg.slogdet(C)
    if s1<=0 or s2<=0: return 0.0
    return float(max(0.0,0.5*(d1-d2)))

# ========= Model =========
class SparseLinear(nn.Module):
    def __init__(self, in_f, out_f, p_connect=1.0, seed=0):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_f,in_f))
        self.bias=nn.Parameter(torch.zeros(out_f))
        nn.init.xavier_uniform_(self.weight)
        g=torch.Generator().manual_seed(seed)
        self.register_buffer("mask",(torch.rand(out_f,in_f,generator=g)<p_connect).float())
    def forward(self,x): return F.linear(x,self.weight*self.mask,self.bias)

class TinyMLP(nn.Module):
    def __init__(self,in_dim=2,widths:List[int]=[8,8],p_connect=1.0,residual=False,seed=0):
        super().__init__()
        self.blocks=nn.ModuleList()
        dims=[in_dim]+widths
        for i in range(len(widths)):
            self.blocks.append(SparseLinear(dims[i],dims[i+1],p_connect,seed+i))
            self.blocks.append(nn.Tanh())
        self.out=SparseLinear(widths[-1],2,1.0,seed+999)
        self.residual=residual
    def forward(self,x,collect=False):
        acts=[]; h=x
        for m in self.blocks:
            if isinstance(m,SparseLinear):
                z=m(h); 
                if self.residual and z.shape[-1]==h.shape[-1]: z=z+h
                h=z
            else:
                h=m(h); 
                if collect: acts.append(h)
        logits=self.out(h)
        return (logits,acts) if collect else logits

# ========= Train/Eval =========
@dataclass
class RunCfg:
    dataset:str="xor"; n_samples:int=1024; noise:float=0.1; seed:int=0
    widths:Tuple[int,int]=(8,8); p_connect:float=1.0; residual:bool=False
    batch:int=128; lr:float=1e-2; epochs:int=40

def make_dataset(cfg:RunCfg):
    if cfg.dataset=="xor": X,y=make_xor(cfg.n_samples,cfg.noise,cfg.seed)
    else: X,y=make_two_moons(cfg.n_samples//2,cfg.noise,cfg.seed)
    n=len(X); idx=np.arange(n); np.random.RandomState(cfg.seed).shuffle(idx)
    tr=int(.8*n); id_tr,id_te=idx[:tr],idx[tr:]
    return (torch.from_numpy(X[id_tr]),torch.from_numpy(y[id_tr])),(torch.from_numpy(X[id_te]),torch.from_numpy(y[id_te]))

def eval_acc(model,loader,dev):
    model.eval(); c=t=0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(dev); yb=yb.to(dev)
            c+=(model(xb).argmax(1)==yb).sum().item(); t+=yb.numel()
    return c/max(1,t)

def run_once(cfg:RunCfg):
    set_seed(cfg.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (Xtr,ytr),(Xte,yte)=make_dataset(cfg)
    tr=DataLoader(TensorDataset(Xtr,ytr),batch_size=cfg.batch,shuffle=True)
    te=DataLoader(TensorDataset(Xte,yte),batch_size=cfg.batch)
    net=TinyMLP(2,list(cfg.widths),cfg.p_connect,cfg.residual,cfg.seed).to(dev)
    opt=torch.optim.Adam(net.parameters(), lr=cfg.lr)
    logs=[]
    for ep in range(cfg.epochs):
        net.train()
        for xb,yb in tr:
            xb=xb.to(dev); yb=yb.to(dev)
            logits,_=net(xb,collect=True)
            loss=F.cross_entropy(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        acc=eval_acc(net,te,dev)
        with torch.no_grad():
            acts_all=[]
            for xb,yb in te:
                _,acts=net(xb.to(dev),collect=True)
                if not acts_all: acts_all=[a.cpu().numpy() for a in acts]
                else:
                    for i,a in enumerate(acts): acts_all[i]=np.vstack([acts_all[i],a.cpu().numpy()])
        tc=[gaussian_total_correlation(a) for a in acts_all]
        logs.append({"epoch":ep+1,"acc":acc,"tc_sum":float(np.sum(tc)),**{f"tc_l{i+1}":v for i,v in enumerate(tc)}})
    return logs

# ========= Main =========
def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["xor","moons"],default="xor")
    ap.add_argument("--widths",nargs="+",type=int,default=[8,8])
    ap.add_argument("--p_connect",type=float,default=1.0)
    ap.add_argument("--residual",type=int,default=0)
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--grid",type=int,default=0)
    ap.add_argument("--fast",type=int,default=1, help="grid خفيفة إذا =1")
    args=ap.parse_args([] if argv is None else argv)

    fields=["dataset","widths","p_connect","residual","seed","epoch","acc","tc_sum","tc_l1","tc_l2"]
    cfgs=[]

    if args.grid:
        if args.fast:
            seeds=[0,1]; widths=[(4,4),(8,8),(16,16)]
            pcon=[0.4,0.8,1.0]; residuals=[0,1]; datasets=[args.dataset]
        else:
            seeds=list(range(5)); widths=[(4,4),(8,8),(16,16),(32,32)]
            pcon=[0.2,0.4,0.6,0.8,1.0]; residuals=[0,1]; datasets=["xor","moons"]
        for ds in datasets:
            for s in seeds:
                for w in widths:
                    for p in pcon:
                        for r in residuals:
                            cfgs.append(RunCfg(dataset=ds,widths=w,p_connect=p,residual=bool(r),seed=s,epochs=args.epochs))
    else:
        cfgs=[RunCfg(dataset=args.dataset,widths=tuple(args.widths),p_connect=args.p_connect,
                     residual=bool(args.residual),seed=args.seed,epochs=args.epochs)]

    with open("results.csv","w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for cfg in cfgs:
            for row in run_once(cfg):
                wr.writerow({"dataset":cfg.dataset,"widths":str(cfg.widths),"p_connect":cfg.p_connect,
                             "residual":cfg.residual,"seed":cfg.seed,**row})
    print(f"Wrote results.csv with {len(cfgs)} run(s).")

if __name__=="__main__":
    main()
