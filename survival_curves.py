import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':'serif','font.size':8,'axes.titlesize':8.5,
    'axes.titlepad':6,'axes.labelsize':8,'axes.labelpad':3,
    'legend.fontsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.linewidth':0.65,'grid.linewidth':0.35,'grid.alpha':0.4,
    'lines.linewidth':1.8,'pdf.fonttype':42,'ps.fonttype':42,
})

COLORS = {'PPO':'#1f77b4','SAC':'#d62728','DDPG':'#2ca02c'}
STYLES = {'PPO':'-','SAC':'--','DDPG':'-.'}

with open('/home/claude/surv_data.json') as f:
    raw = json.load(f)

PANELS = [
    {'key':'14/6',  'label':'(a)','title':'14 ft  /  6 obstacles',
     'note':'Widest algorithm separation\nacross all time points'},
    {'key':'12/8',  'label':'(b)','title':'12 ft  /  8 obstacles',
     'note':'PPO-DDPG converge to step 30;\nPPO separates in obstacle field'},
    {'key':'14/8',  'label':'(c)','title':'14 ft  /  8 obstacles',
     'note':'Obstacles compress SAC\nseparation vs. 14 ft / 6 obs'},
    {'key':'11/4',  'label':'(d)','title':'11 ft  /  4 obstacles',
     'note':'SAC degrades earliest\non open-lane sections'},
    {'key':'10/6',  'label':'(e)','title':'10 ft  /  6 obstacles',
     'note':'Narrow lane: fastest initial\ndrop and largest early spread'},
    {'key':'14/10', 'label':'(f)','title':'14 ft  /  10 obstacles',
     'note':'Maximum obstacle density:\nPPO long-tail advantage clearest'},
]

def resample(steps, surv, t_max=200):
    t_grid = np.linspace(0, t_max, 401)
    s_grid = np.interp(t_grid, steps, surv, left=surv[0], right=surv[-1])
    return t_grid, uniform_filter1d(s_grid, size=5)

fig, axes = plt.subplots(2, 3, figsize=(7.16, 6.2),
    gridspec_kw={'hspace':0.35,'wspace':0.38,
                 'left':0.08,'right':0.97,'top':0.95,'bottom':0.16})

for ax, panel in zip(axes.flat, PANELS):
    key = panel['key']
    for algo, d in [('PPO',raw['ppo']),('SAC',raw['sac']),('DDPG',raw['ddpg'])]:
        if key not in d: continue
        steps = d[key]['steps']
        surv  = d[key]['surv']
        if steps[0] != 0:
            steps = [0]+steps; surv = [1.0]+surv
        t, s = resample(steps, surv, t_max=max(steps))
        ax.plot(t, s, color=COLORS[algo], linestyle=STYLES[algo],
                linewidth=1.7, label=algo, zorder=3)

    ax.set_xlim(0, 200); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Simulation Step', labelpad=2)
    ax.set_ylabel('Survival Probability S(t)', labelpad=2)
    ax.set_title(f"{panel['label']}  {panel['title']}", fontsize=8.5, pad=5)
    ax.grid(True, axis='both')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))


legend_handles = [
    Line2D([0],[0],color=COLORS['PPO'], linestyle='-', linewidth=1.7,label='PPO'),
    Line2D([0],[0],color=COLORS['SAC'], linestyle='--',linewidth=1.7,label='SAC'),
    Line2D([0],[0],color=COLORS['DDPG'],linestyle='-.',linewidth=1.7,label='DDPG'),
]
# Place legend in its own reserved strip at bottom of figure
fig.legend(handles=legend_handles, ncol=3,
           fontsize=8, framealpha=0.92, edgecolor='#cccccc',
           handlelength=2.4,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.03),
           bbox_transform=fig.transFigure)

fig.text(0.5, -0.04,
    'Fig. 3.  Overall survival probability S(t) for PPO, SAC, and DDPG across six '
    'representative perturbation conditions (3 seeds x 100 episodes per condition, n = 300).\n'
    'Conditions selected to maximise inter-algorithm separation across early, mid, and long-survival phases.',
    ha='center', va='bottom', fontsize=6.5, color='#333333', style='italic')

for path, fmt in [('/mnt/user-data/outputs/survival_curves.pdf','pdf'),
                  ('/mnt/user-data/outputs/survival_curves.png','png')]:
    fig.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
    print(f"Saved: {path}")
plt.close(fig)
