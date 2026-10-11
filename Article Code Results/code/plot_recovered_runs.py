"""Generate seven native-vector figure counterparts from saved rerun arrays.

Reported curves in Figures 1--2 are explicitly transcribed from the manuscript,
not recomputed or digitized. All other plotted values come from replay_*.py.
No raster image is read or embedded. Failed optimizations remain labeled.
"""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
DATA=ROOT/'results/reproduction';OUT=ROOT/'figures/audit'
OUT.mkdir(parents=True,exist_ok=True)
BLUE='#246B9C';RED='#B44336';GRAY='#545B66'

def save(fig,name):
    # Matplotlib rasterizes colorbars by default above a small threshold.
    for artist in fig.findobj():
        if hasattr(artist,'set_rasterized'):artist.set_rasterized(False)
    fig.savefig(OUT/name,bbox_inches='tight',metadata={'Creator':'plot_recovered_runs.py','CreationDate':None,'ModDate':None})
    plt.close(fig)

def main():
    m=json.loads((DATA/'maier_replay.json').read_text())['records'];a=np.load(DATA/'maier_replay.npz')
    lam=np.array([r['sparsity'] for r in m]);x=np.arange(len(lam));labels=[f'{l:g}' for l in lam]
    reported=[7,7,7,7,7,3,3]
    for name,key,known,ylabel in [('Figure_1.pdf','active',reported,'Active kernels after thresholding'),('Figure_2.pdf','rmse',[30.4]*5+[30.6,58.1],'Potential RMSE')]:
        fig,ax=plt.subplots(figsize=(7.3,3.4),layout='constrained')
        ax.plot(x,known,'o--',color=BLUE,label='Reported summary'+(' (rounded)' if key=='rmse' else ''))
        ax.plot(x,[r[key] for r in m],'x-',color=RED,ms=8,label='Rerun: all optimizer runs failed')
        ax.set(xticks=x,xticklabels=labels,xlabel=r'Sparsity parameter $\lambda_1$ (discrete settings)',ylabel=ylabel)
        if key=='active':ax.set(yticks=range(1,8),ylim=(.5,8.8))
        else:ax.set_ylim(0,110)
        ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.2)
        save(fig,name)
    X,Y=a['X'],a['Y'];V=a['potentials'];truth=a['truth']
    fig=plt.figure(figsize=(10,9.5),layout='constrained')
    for i in range(8):
        ax=fig.add_subplot(3,3,i+1,projection='3d')
        ax.plot_surface(X,Y,V[i] if i<7 else truth,cmap='viridis',linewidth=0,antialiased=True)
        ax.set_title((rf'$\lambda_1={lam[i]:g}$: failed solve' if i<7 else 'Analytic reference'),fontsize=12)
        ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.tick_params(labelsize=10)
        ax.set_zlabel('$V$',labelpad=0)
    fig.suptitle('Maier–Stein: returned fields are failure diagnostics',fontsize=13)
    save(fig,'Figure_3.pdf')
    residual=np.abs(V-truth);ric=2*((X+1)**2+Y**2);residual=np.concatenate([residual,np.abs(ric-truth)[None]])
    fig,axes=plt.subplots(3,3,figsize=(10,8.3),layout='constrained');levels=np.linspace(0,residual.max(),16)
    for i,ax in enumerate(axes.flat):
        if i==8:ax.axis('off');continue
        cf=ax.contourf(X,Y,residual[i],levels=levels,cmap='magma')
        ax.plot(-1,0,'w+',ms=7)
        title=rf'$\lambda_1={lam[i]:g}$; failed' if i<7 else 'Riccati baseline'
        ax.set_title(title+f'\nRMSE = {np.sqrt(np.mean(residual[i]**2)):.3f}',fontsize=11)
        ax.set(xlabel='$x_1$',ylabel='$x_2$')
    fig.colorbar(cf,ax=axes,label='Absolute error',shrink=.9)
    fig.suptitle('Maier–Stein errors on the recovered 10 × 10 display grid',fontsize=12)
    save(fig,'Figure_4_residuals.pdf')
    e=json.loads((DATA/'example3_replay.json').read_text());rows=e['records']
    n=np.array([r['n'] for r in rows]);err=np.array([r['rmse'] for r in rows]);rate=-np.polyfit(np.log(n),np.log(err),1)[0]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax in axes:
        ax.plot(n,err,'o-',color=BLUE,label='Hybrid rerun')
        ax.plot(n,[r['riccati_rmse'] for r in rows],'s--',color='#B97919',label='Riccati')
        bad=np.array([not all(q['success'] for q in r['optimizers']) for r in rows])
        ax.plot(n[bad],err[bad],'x',color=RED,ms=10,mew=1.4,label='At least one pass hit limit')
        ax.set(yscale='log',xlabel='Collocation points $N$',ylabel='RMSE');ax.grid(alpha=.2);ax.legend(fontsize=10,loc='center left')
    axes[0].set_xscale('log');axes[0].set_title(f'Empirical fit exponent: {rate:.3f}')
    axes[1].set_title('One fixed seed; nonmonotone errors')
    save(fig,'convergence_study_example3.pdf')
    grid=np.load(DATA/'example3_grid.npz');base=np.load(DATA/'example3_baselines.npz');hyb=np.load(DATA/'example3_N400.npz')
    fields=[grid['truth'],grid['riccati'],base['quadratic'],base['polynomial'],hyb['values']]
    names=['Analytic reference','Riccati','Optimized quadratic','Polynomial degrees 2–6','Hybrid (N=400)']
    coords=grid['coordinates'];idx=np.argmin(abs(coords));slice_value=coords[idx];A,B=np.meshgrid(coords,coords)
    def slices(v):
        z=v.reshape(20,20,20)
        return [z[:,:,idx].T,z[:,idx,:].T,z[idx,:,:].T]
    fig,axes=plt.subplots(5,3,figsize=(9,12.7),layout='constrained')
    levels=np.linspace(min(v.min() for v in fields),max(v.max() for v in fields),16)
    for row,(name,field) in enumerate(zip(names,fields)):
        for col,z in enumerate(slices(field)):
            ax=axes[row,col];cf=ax.contourf(A,B,z,levels=levels,cmap='viridis')
            fixed=[3,2,1][col];axis_labels=[(1,2),(1,3),(2,3)][col]
            ax.set_title(name+'\n'+rf'$x_{fixed}={slice_value:.5f}$',fontsize=10)
            ax.set(xlabel=f'$x_{axis_labels[0]}$',ylabel=f'$x_{axis_labels[1]}$',aspect='equal')
    fig.colorbar(cf,ax=axes,label='Potential',shrink=.85,fraction=.025)
    save(fig,'quasi_potential_slices_sparse.pdf')
    fig,axes=plt.subplots(4,3,figsize=(9.5,10.5),layout='constrained')
    for row,(name,field) in enumerate(zip(names[1:],fields[1:])):
        error=np.abs(field-grid['truth']);sliced=slices(error);vmax=max(float(error.max()),1e-16)
        levels=np.linspace(0,vmax,13)
        for col,z in enumerate(sliced):
            ax=axes[row,col];cf=ax.contourf(A,B,z,levels=levels,cmap='magma')
            fixed=[3,2,1][col];axis_labels=[(1,2),(1,3),(2,3)][col]
            ax.set_title(name+'\n'+rf'$x_{fixed}={slice_value:.5f}$',fontsize=10)
            ax.set(xlabel=f'$x_{axis_labels[0]}$',ylabel=f'$x_{axis_labels[1]}$',aspect='equal')
        cb=fig.colorbar(cf,ax=axes[row,:],label='Absolute error',fraction=.03,pad=.025)
        cb.formatter.set_powerlimits((-2,3));cb.formatter.set_useMathText(True);cb.update_ticks()
    fig.suptitle('Common analytic reference; separate color scale for each method',fontsize=11)
    save(fig,'residuals_example3.pdf')
    print('Generated seven native PDFs in',OUT)

if __name__=='__main__':main()
