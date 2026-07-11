import torch
import dataset as D
from train import (make_batch, integrate_batch, init_optics, eval_loss,
                   TrainConfig, _zero_residual_output, GrayBoxUDE)
from mechanistic_ode import MechanisticODE, MeasurementModel
from loss_functions import kinetiflow_loss

data=D.load(); tr,cal,te=D.grouped_split(data)
# check actual operating ranges quickly
print('C_f0 range', float(data['C_f0'].min()), float(data['C_f0'].max()))
cfg=TrainConfig(); cfg.residual_scale=5e-4
torch.manual_seed(0)
core=MechanisticODE.identifiable(
    res_mean=(4.0,0.025,450.0,30.0,55.0),
    res_scale=(5.0,0.025,450.0,3.0,8.0))
_zero_residual_output(core)
model=GrayBoxUDE(core, cfg.residual_scale); meas=MeasurementModel()
init_optics(model,meas,tr,cfg)
res_params=list(model.residual.parameters())
opt=torch.optim.Adam([{'params':res_params,'lr':1e-2},{'params':[meas.alpha,meas.beta],'lr':0.3}])
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=8,min_lr=1e-4)
z0,t,Itar,Ta,RH=make_batch(tr,cfg.L0)
c0=eval_loss(model,meas,cal,cfg)[1]['obs']; best=c0
print('cal0=%.3f'%c0,flush=True)
for ep in range(150):
    opt.zero_grad()
    z,Ip=integrate_batch(model,meas,z0,t,Ta,RH,cfg,use_adjoint=True)
    tot,parts=kinetiflow_loss(Ip,Itar,z,model.B_max,lam=cfg.lam); tot.backward()
    torch.nn.utils.clip_grad_norm_(res_params,1.0); opt.step()
    ct,cp,_=eval_loss(model,meas,cal,cfg); sched.step(ct); best=min(best,cp['obs'])
    if ep%15==0 or ep==149: print('ep%3d train=%.3f cal=%.3f best=%.3f lr=%.1e'%(ep,float(parts['obs']),cp['obs'],best,opt.param_groups[0]['lr']),flush=True)
print('==> cal0=%.3f best=%.3f decrease=%.1f%%'%(c0,best,100*(c0-best)/c0))
