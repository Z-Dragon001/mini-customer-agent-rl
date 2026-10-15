"""Plot complete recorded development evaluations; no model or API calls."""
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'experiments/retail-grpo-repair'


def main():
    rows=list(csv.DictReader((OUT/'results.csv').open()))
    baseline=next(row for row in rows if row['run']=='sft-dev42' and row['tasks']=='15')
    upper=max(40,10*math.ceil((max(100*float(row['success_rate']) for row in rows
                                 if row['split']=='dev' and row['tasks']=='15')+5)/10))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    names=[('outcome','A: outcome','#0072B2','o'),('kl','B: + KL','#E69F00','s'),
           ('shaped','C: + process penalties','#CC79A7','^'),
           ('consistent','D: + numerical fixes','#009E73','D'),
           ('progress','E: + business progress','#D55E00','P'),
           ('attention','F: + attention mask','#882255','X')]
    for key,label,color,marker in names:
        points=sorted([row for row in rows if row['run']==key and row['split']=='dev'
                       and row['seed']=='42' and row['tasks']=='15'],key=lambda row:int(row['round']))
        if points:
            ax.plot([int(row['round']) for row in points],
                    [100*float(row['success_rate']) for row in points],
                    marker=marker,color=color,label=label,linewidth=1.6,markersize=7,alpha=.85)
    ax.axhline(100*float(baseline['success_rate']),color='#444444',linestyle='--',label='SFT baseline')
    ax.set(xlabel='Sampled groups (4 trajectories each)',ylabel='Official task success (%)',
           title='Retail development: 15 tasks, evaluation seed 42',xticks=[8,16,24],ylim=(0,upper))
    ax.grid(axis='y',alpha=.18)
    ax.legend(frameon=False,ncol=2,loc='upper right')
    fig.savefig(OUT/'development_curve.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
