"""
training_curves.py
==================
Generates a 2x2 publication-quality figure of training diagnostics for
PPO, SAC, and DDPG controllers trained with 300 000-step curriculum learning.

Usage
-----
    python training_curves.py \
        --ppo  ppo_training_result  \
        --sac  sac_training_result  \
        --ddpg ddpg_training_result \
        --out  training_curves

Output
------
    training_curves.pdf   (vector, IEEE submission ready)
    training_curves.png   (300 dpi raster preview)

Figure layout  (2 rows x 2 columns)
------------------------------------
  Row 1, Col 1  |  Row 1, Col 2
  --------------+----------------
  Row 2, Col 1  |  Row 2, Col 2

  (a) Episode Length  -- ep_len_mean for PPO, SAC, DDPG vs timesteps
  (b) Episode Reward  -- ep_rew_mean for PPO, SAC, DDPG vs timesteps
  (c) PPO Diagnostics -- explained_variance + learning_rate vs timesteps
  (d) Off-Policy Loss -- actor & critic loss for SAC and DDPG vs timesteps
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# Global style constants  (IEEE two-column, serif body text)
# =============================================================================

FIG_W  = 7.8    # inches -- slightly wider to give twinx labels room
FIG_H  = 6.2    # inches

# Algorithm colours (colour-blind safe)
COLOR = {'PPO': '#1f77b4', 'SAC': '#d62728', 'DDPG': '#2ca02c'}

# Curriculum stage shading
STAGE_XMIN   = [      0,  80_000, 180_000]
STAGE_XMAX   = [ 80_000, 180_000, 310_000]
STAGE_FILL   = ['#eef2ff', '#fff5ee', '#efffee']
STAGE_LABELS = ['Stage 1 (Easy)', 'Stage 2 (Medium)', 'Stage 3 (Hard)']

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 8,
    'axes.titlesize'   : 8.5,
    'axes.titlepad'    : 7,        # extra clearance between title and axes top
    'axes.labelsize'   : 8,
    'axes.labelpad'    : 4,
    'legend.fontsize'  : 7,
    'legend.borderpad' : 0.4,
    'legend.handlelength': 1.8,
    'xtick.labelsize'  : 7.5,
    'ytick.labelsize'  : 7.5,
    'axes.linewidth'   : 0.65,
    'grid.linewidth'   : 0.35,
    'grid.alpha'       : 0.4,
    'lines.linewidth'  : 1.6,
    'pdf.fonttype'     : 42,       # embed fonts for IEEE submission
    'ps.fonttype'      : 42,
})


# =============================================================================
# Parsers
# =============================================================================

def parse_ppo(filepath):
    """
    Parse a PPO SB3 training log.
    Returns list of dicts: ts, ep_len, ep_rew, ev (explained_variance), lr.
    """
    with open(filepath, encoding='utf-8', errors='replace') as f:
        content = f.read()

    pat = re.compile(
        r'ep_len_mean\s*\|\s*([\d.]+).*?'
        r'ep_rew_mean\s*\|\s*([-\d.]+).*?'
        r'total_timesteps\s*\|\s*([\d.]+).*?'
        r'explained_variance\s*\|\s*([-\d.e]+).*?'
        r'learning_rate\s*\|\s*([\de.-]+)',
        re.DOTALL
    )
    records = []
    for m in pat.finditer(content):
        el, er, ts, ev, lr = m.groups()
        records.append({'ts': int(float(ts)), 'ep_len': float(el),
                        'ep_rew': float(er), 'ev': float(ev), 'lr': float(lr)})
    if not records:
        raise ValueError(f"No PPO log blocks found in '{filepath}'.")
    return records


def parse_off_policy(filepath, algo):
    """
    Parse a SAC or DDPG SB3 training log.

    SAC/DDPG reset the internal timestep counter at each curriculum stage
    transition (a new model object is created).  This function detects those
    resets and rebuilds a monotonically increasing cumulative timestep axis
    so all three stages are correctly positioned on a 0-300k plot.

    Returns list of dicts: ts (cumulative), ep_len, ep_rew,
                           actor_loss, critic_loss.
    """
    with open(filepath, encoding='utf-8', errors='replace') as f:
        content = f.read()

    pat = re.compile(
        r'ep_len_mean\s*\|\s*([\d.]+).*?'
        r'ep_rew_mean\s*\|\s*([-\d.]+).*?'
        r'total_timesteps\s*\|\s*([\d.]+).*?'
        r'actor_loss\s*\|\s*([-\d.]+).*?'
        r'critic_loss\s*\|\s*([\d.]+)',
        re.DOTALL
    )
    raw = []
    for m in pat.finditer(content):
        el, er, ts, al, cl = m.groups()
        raw.append({'ts': int(float(ts)), 'ep_len': float(el),
                    'ep_rew': float(er), 'actor_loss': float(al),
                    'critic_loss': float(cl)})
    if not raw:
        raise ValueError(f"No {algo} log blocks found in '{filepath}'.")

    # Detect counter resets (ts drops by >1000) and accumulate an offset
    offset  = 0
    prev_ts = 0
    out     = []
    for entry in raw:
        ts = entry['ts']
        if ts < prev_ts - 1000:
            offset += prev_ts
        rec      = dict(entry)
        rec['ts'] = offset + ts
        out.append(rec)
        prev_ts = ts
    return out


# =============================================================================
# Helpers
# =============================================================================

def subsample(data, n=300):
    """Uniformly thin a list to at most n entries."""
    if len(data) <= n:
        return data
    idx = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return [data[i] for i in idx]


def smooth(values, window=13):
    """Gaussian-weighted moving average for display clarity."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    k = np.exp(-0.5 * ((np.arange(window) - window // 2) / (window / 4)) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(arr, window // 2, mode='edge'), k, mode='valid')


def draw_stages(ax):
    """
    Draw three shaded curriculum stage regions with bottom-aligned labels.
    Dashed vertical lines mark stage boundaries.
    Labels are placed near the bottom of each region to avoid title overlap.
    """
    for xmin, xmax, fc, lbl in zip(STAGE_XMIN, STAGE_XMAX, STAGE_FILL, STAGE_LABELS):
        ax.axvspan(xmin, xmax, color=fc, alpha=0.5, zorder=0, linewidth=0)
        if xmin > 0:                          # boundary line at each stage start
            ax.axvline(xmin, color='#909090', linewidth=0.65, linestyle='--', zorder=1)
        # Label anchored at y=0.04 in axes-fraction coordinates
        ax.text((xmin + xmax) / 2, 0.04, lbl,
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=5.8,
                color='#505050', style='italic',
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))


def fmt_xticks(ax):
    """Format x-axis ticks as 0, 50k, 100k ..."""
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: '0' if x == 0 else f'{int(x/1000)}k'))
    ax.set_xlim(0, 310_000)


def panel_label(ax, txt):
    """Bold panel identifier e.g. (a) in top-left corner."""
    ax.text(0.015, 0.975, txt, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.9))


# =============================================================================
# Panel (a) -- Episode Length
# =============================================================================

def panel_episode_length(ax, ppo, sac, ddpg):
    draw_stages(ax)
    for label, data in [('PPO', ppo), ('SAC', sac), ('DDPG', ddpg)]:
        ts  = np.array([d['ts']     for d in data])
        y   = np.array([d['ep_len'] for d in data])
        # Smoothed line only -- raw trace removed for readability
        ax.plot(ts, smooth(y), color=COLOR[label], linewidth=1.8,
                label=label, zorder=3)

    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Mean Episode Length (steps)')
    ax.set_title('(a)  Mean Episode Length')
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y')
    fmt_xticks(ax)
    ax.legend(loc='upper right', framealpha=0.90, edgecolor='#d0d0d0',
              borderaxespad=0.5)
    panel_label(ax, '(a)')


# =============================================================================
# Panel (b) -- Episode Reward
# =============================================================================

def panel_episode_reward(ax, ppo, sac, ddpg):
    draw_stages(ax)
    for label, data in [('PPO', ppo), ('SAC', sac), ('DDPG', ddpg)]:
        ts = np.array([d['ts']     for d in data])
        y  = np.array([d['ep_rew'] for d in data])
        ax.plot(ts, smooth(y), color=COLOR[label], linewidth=1.8,
                label=label, zorder=3)

    ax.axhline(0, color='#b0b0b0', linewidth=0.55, linestyle=':', zorder=2)
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('(b)  Mean Episode Reward')
    ax.grid(True, axis='y')
    fmt_xticks(ax)
    ax.legend(loc='upper right', framealpha=0.90, edgecolor='#d0d0d0',
              borderaxespad=0.5)
    panel_label(ax, '(b)')


# =============================================================================
# Panel (c) -- PPO Value Quality + Learning Rate
# =============================================================================

def panel_ppo_diagnostics(ax, ppo):
    draw_stages(ax)

    ts = np.array([d['ts'] for d in ppo])
    ev = np.array([d['ev'] for d in ppo])
    lr = np.array([d['lr'] for d in ppo])

    # --- Explained variance (left y-axis, indigo) ---
    C_EV = '#4040C0'
    ax.plot(ts, smooth(ev, window=7), color=C_EV, linewidth=1.8,
            label='Explained Variance', zorder=3)
    ax.axhline(0, color='#b8b8b8', linewidth=0.55, linestyle=':', zorder=2)
    ax.set_ylabel('Explained Variance', color=C_EV, labelpad=3)
    ax.tick_params(axis='y', labelcolor=C_EV)
    ax.set_ylim(-0.15, 0.80)
    ax.set_xlabel('Training Timesteps')
    ax.set_title('(c)  PPO: Value Quality and Learning Rate Decay')
    ax.grid(True, axis='y', alpha=0.3)
    fmt_xticks(ax)

    # --- Learning rate (right y-axis, orange) ---
    # LR range: 5.04e-5 to 2.94e-4  =>  multiply by 1e3 gives 0.050 to 0.294
    C_LR = '#D06000'
    ax2 = ax.twinx()
    ax2.plot(ts, lr * 1e3, color=C_LR, linewidth=1.6,
             linestyle='--', label='Learning Rate', zorder=3)
    ax2.set_ylabel('LR  (\u00d710\u207b\u00b3)', color=C_LR, labelpad=3)
    ax2.tick_params(axis='y', labelcolor=C_LR)
    ax2.set_ylim(0.0, 0.40)          # 0.050-0.294 clearly visible
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Combined legend placed at centre-right to avoid EV curve and stage labels
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='center right',
              framealpha=0.92, edgecolor='#d0d0d0', borderaxespad=0.5)
    panel_label(ax, '(c)')


# =============================================================================
# Panel (d) -- Off-Policy Losses (SAC + DDPG)
# =============================================================================

def panel_off_policy_losses(ax, sac, ddpg):
    draw_stages(ax)

    sac_ts  = np.array([d['ts'] for d in sac])
    ddpg_ts = np.array([d['ts'] for d in ddpg])

    sac_al  = np.array([d['actor_loss']  for d in sac])
    ddpg_al = np.array([d['actor_loss']  for d in ddpg])
    sac_cl  = np.array([d['critic_loss'] for d in sac])
    ddpg_cl = np.array([d['critic_loss'] for d in ddpg])

    # Clip critic loss at 95th percentile to suppress spikes
    sac_cl  = np.clip(sac_cl,  None, np.percentile(sac_cl,  95))
    ddpg_cl = np.clip(ddpg_cl, None, np.percentile(ddpg_cl, 95))

    # --- Actor losses (left y-axis) ---
    C_SA = COLOR['SAC']
    C_DA = COLOR['DDPG']
    ax.plot(sac_ts,  smooth(sac_al),  color=C_SA, linewidth=1.8,
            linestyle='-',  label='SAC Actor Loss',  zorder=3)
    ax.plot(ddpg_ts, smooth(ddpg_al), color=C_DA, linewidth=1.8,
            linestyle='-',  label='DDPG Actor Loss', zorder=3)
    ax.axhline(0, color='#b0b0b0', linewidth=0.55, linestyle=':', zorder=2)
    ax.set_ylabel('Actor Loss  (\u2193 = better)', labelpad=3)
    ax.set_xlabel('Training Timesteps')
    ax.set_title('(d)  Off-Policy Actor and Critic Losses (SAC & DDPG)')
    ax.grid(True, axis='y', alpha=0.3)
    fmt_xticks(ax)

    # --- Critic losses (right y-axis, dashed, muted colours) ---
    C_SC = '#C06080'   # muted rose
    C_DC = '#806040'   # muted brown
    ax2 = ax.twinx()
    ax2.plot(sac_ts,  smooth(sac_cl),  color=C_SC, linewidth=1.4,
             linestyle='--', label='SAC Critic Loss',  zorder=3)
    ax2.plot(ddpg_ts, smooth(ddpg_cl), color=C_DC, linewidth=1.4,
             linestyle='--', label='DDPG Critic Loss', zorder=3)
    ax2.set_ylabel('Critic Loss  (95th pct. clip)', labelpad=3)
    ax2.tick_params(axis='y')

    # Legend: 4 entries in 2 columns placed at upper right (losses start high
    # then settle -- bottom-left stays clear of the data)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2,
              loc='upper right', ncol=2,
              framealpha=0.92, edgecolor='#d0d0d0',
              borderaxespad=0.5, fontsize=6.8)
    panel_label(ax, '(d)')


# =============================================================================
# Figure assembly
# =============================================================================

def build_figure(ppo_file, sac_file, ddpg_file, out_stem='training_curves'):
    print(f"  Parsing PPO  : {ppo_file}")
    ppo_raw  = parse_ppo(ppo_file)

    print(f"  Parsing SAC  : {sac_file}")
    sac_raw  = parse_off_policy(sac_file,  'SAC')

    print(f"  Parsing DDPG : {ddpg_file}")
    ddpg_raw = parse_off_policy(ddpg_file, 'DDPG')

    ppo  = ppo_raw
    sac  = subsample(sac_raw,  300)
    ddpg = subsample(ddpg_raw, 300)
    print(f"  Points -- PPO:{len(ppo)}  SAC:{len(sac)}  DDPG:{len(ddpg)}")

    # Use gridspec_kw to add explicit horizontal and vertical spacing
    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={
            'hspace': 0.52,   # vertical gap between rows
            'wspace': 0.52,   # wider column gap so twinx labels don't collide
            'left'  : 0.07,   # tighter left margin
            'right' : 0.94,   # tighter right margin
            'top'   : 0.95,
            'bottom': 0.09,
        }
    )
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    panel_episode_length   (ax_a, ppo, sac, ddpg)
    panel_episode_reward   (ax_b, ppo, sac, ddpg)
    panel_ppo_diagnostics  (ax_c, ppo)
    panel_off_policy_losses(ax_d, sac, ddpg)

    # Figure caption
    fig.text(
        0.5, 0.018,
        'Fig. 2.  Training diagnostics for PPO, SAC, and DDPG over 300\u2009000-step curriculum\n'
        '(Stage\u00a01 Easy \u2192 Stage\u00a02 Medium \u2192 Stage\u00a03 Hard). '
        'Dashed lines mark stage transitions. Curves are Gaussian-smoothed.',
        ha='center', va='top', fontsize=6.5,
        color='#333333', style='italic', linespacing=1.5
    )

    pdf_path = f'{out_stem}.pdf'
    png_path = f'{out_stem}.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f"  Saved -> {pdf_path}")
    print(f"  Saved -> {png_path}")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate 2x2 training-curves figure for the ANNSIM paper.')
    parser.add_argument('--ppo',  required=True)
    parser.add_argument('--sac',  required=True)
    parser.add_argument('--ddpg', required=True)
    parser.add_argument('--out',  default='training_curves')
    args = parser.parse_args()
    print('\n  Building training curves figure ...')
    build_figure(args.ppo, args.sac, args.ddpg, args.out)
    print('  Done.\n')


if __name__ == '__main__':
    main()
