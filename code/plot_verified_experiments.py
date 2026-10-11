"""Native PDF figures and TeX tables for the verified main-text experiments."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from run_polynomial_experiments import problem,reference,predict,grid,hj
from refit_selected_kernel import features
DATA=ROOT/'results/accuracy';OUT=ROOT/'figures/native';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#9B3D25','#226794','#25826A']

def save(fig,name):
 for artist in fig.findobj():
  if hasattr(artist,'set_rasterized'):artist.set_rasterized(False)
 fig.savefig(OUT/name,bbox_inches='tight',metadata={'Creator':'plot_verified_experiments.py','CreationDate':None,'ModDate':None});plt.close(fig)

def texnum(x):
 if x==0:return '$0$'
 if .001<=abs(x)<1000:return f'${x:.4g}$'
 a,e=f'{x:.2e}'.split('e');return '$'+a+r'\times10^{'+str(int(e))+'}$'

def hausdorff(v,t,x,levels,band):
 out=[]
 for level in levels:
  a=x[abs(v-level)<=band];b=x[abs(t-level)<=band]
  if len(a) and len(b):out.append(max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0]))
 return float(np.mean(out)) if out else None

def main():
 rows=json.loads((DATA/'polynomial_experiments.json').read_text())['records'];stable=json.loads((DATA/'example3_stable.json').read_text())['records'];extra=[]
 # Two independent model-selection/accuracy plots, showing all candidates.
 for kind,filename in [('validation','Figure_1.pdf'),('accuracy','Figure_2.pdf')]:
  fig,axes=plt.subplots(1,2,figsize=(10,3.65),layout='constrained')
  for ax,name,title in zip(axes,['maier_stein','double_well'],['Maier–Stein','Double well']):
   p0,P,b=problem(name)
   for color,row in zip(COLORS,[r for r in rows if r['problem']==name]):
    model=np.load(DATA/f'{name}_polynomial_seed{row["seed"]}.npz');points=model['display'];truth=reference(name,points)
    values=[c['validation_hj_rms'] if kind=='validation' else float(np.sqrt(np.mean((predict(points,p0,P,c['degree'],np.array(c['coefficients']))[0]-truth)**2))) for c in row['candidates']]
    ax.semilogy(range(2,7),values,'o-',color=color,label=f'Seed {row["seed"]}')
   if kind=='validation':ax.axhline(1e-6,color='gray',ls='--',lw=1,label='Selection tolerance')
   ax.set(xticks=range(2,7),xlabel='Polynomial kernel degree',ylabel='Validation HJ RMS' if kind=='validation' else 'Potential RMSE',title=title);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
  save(fig,filename)
 # Maier surface and residual plots at declared physical coordinates.
 r=rows[0];p0,P,b=problem('maier_stein');coef=np.array(r['candidates'][r['chosen_degree']-2]['coefficients']);points=grid([-4,-3],[0,3],81);X=points[:,0].reshape(81,81);Y=points[:,1].reshape(81,81)
 fit,g=predict(points,p0,P,r['chosen_degree'],coef);ric,rg=predict(points,p0,P,2,np.array([]));truth=reference('maier_stein',points)
 fig=plt.figure(figsize=(11,3.8),layout='constrained')
 for k,(v,title) in enumerate([(truth,'Analytic potential'),(ric,'Riccati quadratic'),(fit,'Degree-four kernel fit')]):
  ax=fig.add_subplot(1,3,k+1,projection='3d');ax.plot_surface(X,Y,v.reshape(81,81),cmap='viridis',rstride=2,cstride=2,linewidth=0);ax.set(xlabel='$x_1$',ylabel='$x_2$',zlabel='$V$',title=title);ax.view_init(25,-125);ax.tick_params(labelsize=8)
 save(fig,'Figure_3.pdf')
 fig,axes=plt.subplots(2,2,figsize=(9,6.7),layout='constrained')
 for row,(v,gg,title) in enumerate([(ric,rg,'Riccati'),(fit,g,'Degree-four fit')]):
  for col,(z,metric) in enumerate([(abs(v-truth),'Absolute potential error'),(abs(hj(b(points),gg)),'Absolute HJ residual')]):
   ax=axes[row,col];cf=ax.contourf(X,Y,z.reshape(81,81),levels=np.linspace(0,max(float(max(z)),1e-16),17),cmap='magma');ax.set(xlabel='$x_1$',ylabel='$x_2$',title=f'{title}: {metric}');cb=fig.colorbar(cf,ax=ax);cb.formatter.set_powerlimits((-2,3));cb.formatter.set_useMathText(True);cb.update_ticks()
 save(fig,'Figure_4_residuals.pdf')
 # Example 2 metrics and actual discrete level-set comparison, all seeds.
 for row in rows:
  name=row['problem'];x0,P,b=problem(name);c=np.array(row['candidates'][row['chosen_degree']-2]['coefficients']);a=np.load(DATA/f'{name}_polynomial_seed{row["seed"]}.npz');test=a['test'];v,g=predict(test,x0,P,row['chosen_degree'],c);rv,rg=predict(test,x0,P,2,np.array([]));truth=reference(name,test)
  extra.append(dict(problem=name,seed=row['seed'],riccati_basin_rmse=float(np.sqrt(np.mean((rv-truth)**2))),riccati_basin_relative_l2=float(np.linalg.norm(rv-truth)/np.linalg.norm(truth))))
  if name=='double_well':
   hp=grid([.05,-1.5],[1.5,1.5],101);hv=predict(hp,x0,P,row['chosen_degree'],c)[0];hr=predict(hp,x0,P,2,np.array([]))[0];ht=reference(name,hp)
   extra[-1].update(levelset_grid=101,levels=[.1,.2,.3,.4],band=.02,fit_hausdorff=hausdorff(hv,ht,hp,[.1,.2,.3,.4],.02),riccati_hausdorff=hausdorff(hr,ht,hp,[.1,.2,.3,.4],.02))
 dw=[r for r in rows if r['problem']=='double_well'];de=[r for r in extra if r['problem']=='double_well'];body=[]
 for label,rmse,rel,hd in [('Riccati',de[0]['riccati_basin_rmse'],de[0]['riccati_basin_relative_l2'],de[0]['riccati_hausdorff']),('Polynomial kernel ($r=4$)',dw[0]['independent_basin_metrics']['rmse'],dw[0]['independent_basin_metrics']['relative_l2'],de[0]['fit_hausdorff'])]:body.append(label+' & '+' & '.join(map(texnum,[rmse,rel,hd]))+r' \\')
 (ROOT/'results/example2_verified_table.tex').write_text(r'''\begin{table}[htbp]\centering
\caption{Double-well reconstruction, seed 42. Potential errors use 10,000 independent points in $[0.05,1.5]\times[-1.5,1.5]$. The level-set distance uses a separate $101^2$ grid, levels $0.1,0.2,0.3,0.4$, and band width $0.02$.}\label{tab:example2_errors}
\begin{tabular}{lccc}\toprule Method & RMSE & Relative $L^2$ & Mean $d_H$\\\midrule
'''+ '\n'.join(body)+r'\bottomrule\end{tabular}\end{table}'+'\n')
 # Example 3 exact grid baseline and stable fits.
 e=np.load(DATA/'example3_stable_N200.npz');pts=e['points'];P=e['P'];truth=pts[:,0]**2+.5*np.sum(pts[:,1:]**2,axis=1);ric=.5*np.einsum('ni,ij,nj->n',pts,P,pts);rr=stable[2];err=ric-truth;ricmetrics=[float(np.sqrt(np.mean(err**2))),float(max(abs(err))),float(np.linalg.norm(err)/np.linalg.norm(truth))]
 lines=['Riccati & '+' & '.join(map(texnum,ricmetrics))+r' \\',r'Anchored kernel refit ($r=2$) & '+' & '.join(texnum(rr['grid_metrics'][k]) for k in ['rmse','max_abs','relative_l2'])+r' \\']
 (ROOT/'results/example3_verified_table.tex').write_text(r'''\begin{table}[htbp]\centering
\caption{Multiplicative-noise example: seed 42, $N=200$, and the fixed $20^3$ evaluation grid on $[-2,2]^3$.}\label{tab:example3_comparison}
\begin{tabular}{lccc}\toprule Method & RMSE & Maximum error & Relative $L^2$\\\midrule
'''+ '\n'.join(lines)+r'\bottomrule\end{tabular}\end{table}'+'\n')
 fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');counts=[r['n'] for r in stable]
 for ax,key,label in [(axes[0],'rmse','Potential RMSE'),(axes[1],'hj_mean_abs','Mean absolute HJ residual')]:
  ax.loglog(counts,[r['grid_metrics'][key] for r in stable],'o-',color=COLORS[1],label='Fixed grid');ax.loglog(counts,[r['independent_test_metrics'][key] for r in stable],'s--',color=COLORS[2],label='Independent random test');ax.set(xlabel='Collocation count $N$',ylabel=label);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
 save(fig,'convergence_study_example3.pdf')
 # Evaluate actual zero slices independently of the even 20-point metric grid.
 coords=np.linspace(-2,2,81);X,Y=np.meshgrid(coords,coords);c=e['coefficients'];fields=[]
 for fixed in [2,1,0]:
  pp=np.zeros((X.size,3));free=[i for i in range(3) if i!=fixed];pp[:,free[0]]=X.ravel();pp[:,free[1]]=Y.ravel();t=pp[:,0]**2+.5*np.sum(pp[:,1:]**2,axis=1);rv=.5*np.einsum('ni,ij,nj->n',pp,P,pp);fv=features(pp)[0]@c;fields.append((t,rv,fv))
 for error,filename in [(False,'quasi_potential_slices_sparse.pdf'),(True,'residuals_example3.pdf')]:
  nrows=2 if error else 3;fig,axes=plt.subplots(nrows,3,figsize=(10,6.1 if error else 8.1),layout='constrained')
  for row in range(nrows):
   index=row+1 if error else row;vmax=max(max(abs(f[index]-f[0])) for f in fields) if error else 8.;vmax=max(vmax,1e-16)
   for col,(t,rv,fv) in enumerate(fields):
    vals=[t,rv,fv];z=abs(vals[index]-t) if error else vals[index];ax=axes[row,col];cf=ax.contourf(X,Y,z.reshape(81,81),levels=np.linspace(0,vmax,17),cmap='magma' if error else 'viridis');title=['Reference','Riccati','Anchored kernel'][index];fixed=[3,2,1][col];free=[i for i in [1,2,3] if i!=fixed];ax.set(title=title+f': $x_{fixed}=0$',xlabel=f'$x_{free[0]}$',ylabel=f'$x_{free[1]}$',aspect='equal')
   cb=fig.colorbar(cf,ax=axes[row,:],label='Absolute error' if error else 'Potential',fraction=.026,pad=.02);cb.formatter.set_powerlimits((-2,3));cb.formatter.set_useMathText(True);cb.update_ticks()
  save(fig,filename)
 (DATA/'main_text_metrics.json').write_text(json.dumps(dict(two_dimensional=extra,riccati_example3=ricmetrics,description='Metrics calculated only after model selection; no reference-based amplitude rescaling'),indent=2)+'\n')
 print('Generated seven verified vector PDFs and two TeX tables.')
if __name__=='__main__':main()
