"""Build manuscript tables from saved, unrounded results. Run after experiments."""
import json
from pathlib import Path
from statistics import mean, stdev, median
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/experiments.json').read_text())
names={'maier_stein':'1: MS, $d=2$','double_well':'2: DW, $d=2$','multiplicative':'3: Mult., $d=3$','rotational_8d':'4: Rot., $d=8$'}
methods={'riccati':'Riccati','direct':'Direct','koopman':'Koopman'}
def fmt(v):
 if abs(v)<.001 or abs(v)>=100:
  a,b=f'{v:.2e}'.split('e');return a+r'\!\times\!10^{'+str(int(b))+'}'
 return f'{v:.3g}'
def stats(rows,k):
 return r'\shortstack{$'+fmt(mean(r[k] for r in rows))+r'$\\{\scriptsize$\pm '+fmt(stdev(r[k] for r in rows))+r'$}}'
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Test errors: mean $\pm$ sample standard deviation over three seeds. No reference-based rescaling is applied. MS, DW, Mult., and Rot. identify the four test systems.}\label{tab:accuracy}',r'\begin{tabular}{@{}llccc@{}}',r'\toprule',r'Example & Model & RMSE & Relative $\ell^2$ error & HJ RMS\\',r'\midrule']
for name,label in names.items():
 for j,(method,display) in enumerate(methods.items()):
  rows=[r for r in data['runs'] if r['problem']==name and r['method']==method]
  lines.append((label if j==0 else '')+' & '+display+' & '+' & '.join(stats(rows,k) for k in ['rmse','rel_l2','hj_rms'])+r'\\')
 if name!='rotational_8d':lines.append(r'\addlinespace')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(ROOT/'results/accuracy_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Median measured time in milliseconds. Fit includes the three-bandwidth search and validation; test is value and gradient prediction on 4096 points. Problem setup, imports, plots, and error calculation are excluded.}\label{tab:timing}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r' & \multicolumn{2}{c}{Direct} & \multicolumn{2}{c}{Koopman} & Riccati\\',r'Example & Fit & Test & Fit & Test & Test\\',r'\midrule']
for name,label in names.items():
 vals=[]
 for method in ['direct','koopman','riccati']:
  rows=[r for r in data['runs'] if r['problem']==name and r['method']==method]
  for key in (['train_seconds','test_seconds'] if method!='riccati' else ['test_seconds']):
   vals.append(f'{1000*median(r[key] for r in rows):.2f}')
 lines.append(label+' & '+' & '.join(vals)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(ROOT/'results/timing_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Dense Gaussian Gram microbenchmark: median milliseconds after one warm-up, with storage for the Gram matrix alone. MiB denotes $2^{20}$ bytes. This is separate from the reduced-center fits.}\label{tab:dense}',r'\begin{tabular}{@{}rrrrr@{}}',r'\toprule',r'$d$ & $N$ & Assembly (ms) & Cholesky (ms) & Gram (MiB)\\',r'\midrule']
for r in data['dense_costs']:
 lines.append(f"{r['d']} & {r['N']} & {1000*r['assembly_seconds']:.3f} & {1000*r['factor_seconds']:.3f} & {r['gram_MiB']:.3f}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(ROOT/'results/dense_table.tex').write_text('\n'.join(lines)+'\n')
print('Wrote three tables from experiments.json')
