#coding:utf8
import visdom
import time
import numpy as np
import torch

class Visualizer(object):
    def denormalize(self,x_hat,mean=None,std=None):
        x = x_hat.clone().detach().cpu() 
        if mean != None and std != None:
            mean = torch.tensor(mean).reshape(3, 1, 1)
            std = torch.tensor(std).reshape(3, 1, 1)
            x = x * std + mean

        x = x.mul_(255).add_(0.5).clamp_(0, 255)
        return x.detach()

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = 1
        self.log_text = ''

    def reinit(self,env='default',**kwargs):
        self.vis = visdom.Visdom(env=env,**kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y,**kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot_curves(self, d, iters, title='loss', xlabel='iters', ylabel='accuracy'):
        name = list(d.keys())
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title,
                      opts=dict(legend=name, title = title, xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')
        self.index = iters

    def img(self, name, img_, mean=None, std=None, **kwargs):
        img = self.denormalize(img_,mean,std)
        self.vis.images(img.numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )
 
    def log(self,info,win='log_text'):
        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(self.log_text,win)   

    def __getattr__(self, name):
        return getattr(self.vis, name)

