"""Geometry-conditioned linear thermal response for native-grid mean evolution."""
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from SciML.models.v12.learning_coordinate_operator import interpolate


class KernelMeanReadout(nn.Module):
    kind='thermal_kernel'
    latent_width=128

    def __init__(self,rank=3,samples=12,reference_temperature=293.):
        super().__init__();self.rank=rank;self.samples=samples;self.reference_temperature=reference_temperature
        shapes={'thermal_scale':(210,),'geometry_mean':(4,),'geometry_scale':(4,),
                'training_thermal':(samples,210),'training_geometry':(samples,4),'dual':(samples,rank),'basis':(rank,50)}
        for name,shape in shapes.items():self.register_buffer(name,torch.zeros(shape,dtype=torch.float64))
        self.thermal_scale.fill_(1);self.geometry_scale.fill_(1)
        self.register_buffer('time_grid',torch.linspace(0,1,50,dtype=torch.float64))

    def features(self,initial,control,state):
        if initial.shape!=(len(control),2,101) or control.shape!=(len(initial),9) or state is None or state.shape!=(len(initial),515):
            raise ValueError('Thermal response requires fields, controls and predicted thermal context')
        thermal=torch.cat((initial.flatten(1),state[:,406:414]),dim=1)-self.reference_temperature
        geometry=control[:,[0,1,3,4]]
        if not torch.isfinite(thermal).all() or not torch.isfinite(geometry).all():raise ValueError('Nonfinite thermal inputs')
        return thermal,geometry

    @staticmethod
    def kernel(thermal,geometry,other_thermal,other_geometry):
        distance=(geometry[:,None]-other_geometry[None]).square().mean(-1)
        return torch.exp(-.5*distance)*(thermal@other_thermal.T/thermal.shape[1])

    @threadpool_limits.wrap(limits=1)
    def fit(self,initial,control,target,alpha=.1,state=None,latent=None):
        thermal,geometry=self.features(initial,control,state)
        if len(thermal)!=self.samples or self.samples<=self.rank or target.shape!=(self.samples,50) or not torch.isfinite(target).all() or alpha<=0:
            raise ValueError('Invalid response supervision')
        y=target.detach().clone();y[:,0]=0.
        self.thermal_scale.copy_(thermal.std(0,unbiased=False).clamp_min(.1))
        self.geometry_mean.copy_(geometry.mean(0))
        floor=torch.tensor([.003,1.5e-5,.1,.4],dtype=torch.float64)
        self.geometry_scale.copy_(torch.maximum(geometry.std(0,unbiased=False),floor))
        self.training_thermal.copy_(thermal/self.thermal_scale)
        self.training_geometry.copy_((geometry-self.geometry_mean)/self.geometry_scale)
        _,_,vh=np.linalg.svd(y.detach().cpu().numpy(),full_matrices=False)
        self.basis.copy_(torch.tensor(vh[:self.rank],dtype=torch.float64))
        codes=y@self.basis.T
        covariance=self.kernel(self.training_thermal,self.training_geometry,self.training_thermal,self.training_geometry)
        self.dual.copy_(torch.linalg.solve(covariance+alpha*torch.eye(self.samples,dtype=torch.float64),codes))
        return {'samples':self.samples,'thermal_features':210,'geometry_features':4,'rank':self.rank,'alpha':alpha,
            'reference_temperature_K':self.reference_temperature,
            'training_basis_rmse_K':float(((codes@self.basis-y)[:,1:]).square().mean().sqrt())}

    def forward(self,initial,control,times,state=None,latent=None):
        if times.ndim!=1 or not len(times) or not torch.isfinite(times).all() or torch.any((times<0)|(times>1)):
            raise ValueError('Time queries must be finite within [0,1]')
        thermal,geometry=self.features(initial,control,state)
        k=self.kernel(thermal/self.thermal_scale,(geometry-self.geometry_mean)/self.geometry_scale,
                      self.training_thermal,self.training_geometry)
        profile=(k@self.dual)@self.basis
        profile=torch.cat((torch.zeros_like(profile[:,:1]),profile[:,1:]),dim=1)
        return interpolate(profile,self.time_grid,times)
