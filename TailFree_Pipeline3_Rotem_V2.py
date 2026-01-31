from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_rel, ttest_ind

plt.close('all')
data_dir_ctrl = r'Z:\Ella\control'
dir_list_ctrl = glob(data_dir_ctrl + '\\F*')

data_dir_pos = r'Z:\Ella\positive'
dir_list_pos = glob(data_dir_pos + '\\F*')

groups = [dir_list_ctrl, dir_list_pos]
groups_labels = ['ChR2 neg', 'ChR2 pos']
colors = ['k', 'c']

# --- Options ---
INCLUDE_DURING = False  # << toggle during ON/OFF
REMOVE_OUTLIERS = True  # as before
GROUP_TO_FILTER = groups_labels[1]  # which group to filter when removing outliers
OUTLIER_THRESH = 3.5  # MAD threshold

FPS = 400.0
RECORDING_DURATION_SEC = 60.0  # used for frequency (bouts/sec)


# ---------- Helpers ----------
def get_indices(swim_info):
    start_keys = ['startSwimInd', 'start_swim_ind', 'swimStartInd', 'startInd']
    end_keys = ['endSwimInd', 'end_swim_ind', 'swimEndInd', 'stopSwimInd', 'endInd']
    start = None;
    end = None
    for k in start_keys:
        if k in swim_info:
            start = np.asarray(swim_info[k], dtype=float);
            break
    for k in end_keys:
        if k in swim_info:
            end = np.asarray(swim_info[k], dtype=float);
            break
    if start is None or end is None:
        raise KeyError("start/end swim indices not found in swim_info")
    return start, end


def avg_bout_duration_seconds(swim_info, fps=FPS):
    start, end = get_indices(swim_info)
    dur_frames = end - start
    dur_frames = dur_frames[np.isfinite(dur_frames) & (dur_frames > 0)]
    if dur_frames.size == 0: return np.nan
    return float(np.nanmean(dur_frames) / fps)


def mad_outlier_mask(x, thresh=OUTLIER_THRESH):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0: return np.zeros_like(x, dtype=bool)
    xf = x[finite]
    med = np.median(xf)
    mad = np.median(np.abs(xf - med))
    if mad == 0: return finite
    mz = 0.6745 * (x - med) / mad
    keep = (np.abs(mz) <= thresh) & finite
    return keep


def apply_group_pairwise_outlier_filter(ydata_dict, group_label, active_conds):
    """
    If REMOVE_OUTLIERS and label==GROUP_TO_FILTER:
      - compute MAD outliers independently per active condition
      - drop any fish that is an outlier in ANY active condition (pairwise removal across all)
    """
    out = {g: {c: np.asarray(ydata_dict[g][c], dtype=float) for c in ydata_dict[g]} for g in ydata_dict}
    if not (REMOVE_OUTLIERS and group_label == GROUP_TO_FILTER):
        return out

    # Intersect keep masks across active conditions
    keep_masks = []
    for cond in active_conds:
        vals = np.asarray(ydata_dict[group_label][cond], dtype=float)
        keep_masks.append(mad_outlier_mask(vals))
    keep_pair = np.logical_and.reduce(keep_masks)

    removed = int(np.count_nonzero(~keep_pair))
    print(f"\n[Outlier removal ON] Group '{group_label}': removed {removed} fish (pairs across {active_conds}).")

    for cond in out[group_label]:
        out[group_label][cond] = np.asarray(ydata_dict[group_label][cond], dtype=float)[keep_pair]
    return out


# ---------- Containers (supporting during) ----------
conds_all = ['before', 'during', 'after'] if INCLUDE_DURING else ['before', 'after']

all_data_counts = {label: {c: [] for c in conds_all} for label in groups_labels}
all_data_dur = {label: {c: [] for c in conds_all} for label in groups_labels}

# ---------- Collect data per fish ----------
for group, label in zip(groups, groups_labels):
    # Per-condition lists
    counts = {c: [] for c in conds_all}
    durs = {c: [] for c in conds_all}

    for directory in group:
        # Load required files
        swim_files = {
            'before': directory + '\\swim_info_before.npy',
            'after': directory + '\\swim_info_after.npy'
        }
        if INCLUDE_DURING:
            swim_files['during'] = directory + '\\swim_info_during.npy'

        swim_info = {}
        for c in conds_all:
            swim_info[c] = np.load(swim_files[c], allow_pickle=True)[()]
            print(swim_files[c])
        # Counts and durations by condition
        for c in conds_all:
            s, _ = get_indices(swim_info[c])
            counts[c].append(len(s))
            durs[c].append(avg_bout_duration_seconds(swim_info[c], fps=FPS))

    # Save
    for c in conds_all:
        all_data_counts[label][c] = np.asarray(counts[c], dtype=float)
        all_data_dur[label][c] = np.asarray(durs[c], dtype=float)

# ---------- Figure & axes ----------
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

# Anchors for conditions (centered with small gaps)
if INCLUDE_DURING:
    anchors = {'before': 0.0, 'during': 0.4, 'after': 0.8}
else:
    anchors = {'before': 0.0, 'after': 0.8}

# Group offsets (left/right)
offsets = {groups_labels[0]: -0.12, groups_labels[1]: +0.12}

# Build x_positions mapping
x_positions = {lab: {c: anchors[c] + offsets[lab] for c in anchors.keys()} for lab in groups_labels}


def plot_data(ax, ydata_dict, ylabel, title):
    """Scatter, sequential lines, mean±SEM, and stats with low bars. Respects 'during' and outlier removal for second group."""
    active_conds = list(anchors.keys())

    # Apply outlier filtering only to chosen group across ALL active conditions
    ydat = apply_group_pairwise_outlier_filter(ydata_dict, GROUP_TO_FILTER, active_conds)

    # Determine y-range from all active conditions
    all_vals = []
    for g in groups_labels:
        for cond in active_conds:
            v = np.asarray(ydat[g][cond], dtype=float)
            if v.size: all_vals.append(v[np.isfinite(v)])
    if all_vals:
        all_vals = np.concatenate(all_vals) if len(all_vals) else np.array([0.0])
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    else:
        ymin, ymax = 0.0, 1.0
    yrng = max(1e-9, ymax - ymin)

    # Plot
    for label, color in zip(groups_labels, colors):
        # Scatter points per condition
        per_cond = {c: np.asarray(ydat[label][c], dtype=float) for c in active_conds}
        masks = {c: np.isfinite(per_cond[c]) for c in active_conds}

        for c in active_conds:
            ax.scatter([x_positions[label][c]] * masks[c].sum(), per_cond[c][masks[c]],
                       color=color, alpha=0.3, s=35)

        # Connect sequential segments (before→during→after if available)
        # Connect only for fish indices that have finite values for both endpoints of a segment
        if INCLUDE_DURING:
            # segment 1: before→during
            b, d = per_cond['before'], per_cond['during']
            mseg = np.isfinite(b) & np.isfinite(d)
            for y1, y2 in zip(b[mseg], d[mseg]):
                ax.plot([x_positions[label]['before'], x_positions[label]['during']],
                        [y1, y2], color=color, alpha=0.2, lw=1.5)
            # segment 2: during→after
            d, a = per_cond['during'], per_cond['after']
            mseg = np.isfinite(d) & np.isfinite(a)
            for y1, y2 in zip(d[mseg], a[mseg]):
                ax.plot([x_positions[label]['during'], x_positions[label]['after']],
                        [y1, y2], color=color, alpha=0.2, lw=1.5)
        else:
            b, a = per_cond['before'], per_cond['after']
            mseg = np.isfinite(b) & np.isfinite(a)
            for y1, y2 in zip(b[mseg], a[mseg]):
                ax.plot([x_positions[label]['before'], x_positions[label]['after']],
                        [y1, y2], color=color, alpha=0.2, lw=1.5)

        # Mean ± SEM per condition
        for c in active_conds:
            vals = per_cond[c][np.isfinite(per_cond[c])]
            if vals.size:
                mean = float(np.nanmean(vals))
                sem = float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                ax.errorbar(x_positions[label][c], mean, yerr=sem, fmt='o',
                            color=color, markersize=12, capsize=4)

    # Axis formatting
    ax.set_xticks([anchors[c] for c in active_conds])
    ax.set_xticklabels([c.capitalize() for c in active_conds], fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title)
    ax.set_xlim([-0.4, 1.2])

    # --- Stats (low bars) ---
    def add_sig(ax_, x1, x2, pval, row=0):
        if not (np.isfinite(pval) and pval < 0.05): return
        bar_h = 0.02 * yrng
        y = (ymax + 0.01 * yrng) + row * (0.03 * yrng)
        ax_.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color='k', lw=1)
        stars = '***' if pval < 1e-3 else ('**' if pval < 1e-2 else '*')
        ax_.text((x1 + x2) / 2, y + bar_h * 1.2, stars, ha='center', va='bottom')

    # Within-group paired tests (adjacent comparisons)
    r = 0
    if INCLUDE_DURING:
        pairs_within = [('before', 'during'), ('during', 'after')]
    else:
        pairs_within = [('before', 'after')]

    for label in groups_labels:
        for (c1, c2) in pairs_within:
            v1 = np.asarray(ydat[label][c1], dtype=float)
            v2 = np.asarray(ydat[label][c2], dtype=float)
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() >= 2:
                _, p = ttest_rel(v1[mask], v2[mask])
                add_sig(ax, x_positions[label][c1], x_positions[label][c2], p, row=r)
                r += 1

    # Between-group tests for each active condition
    for c in active_conds:
        g1 = np.asarray(ydat[groups_labels[0]][c], dtype=float)
        g2 = np.asarray(ydat[groups_labels[1]][c], dtype=float)
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        if (g1.size >= 2) and (g2.size >= 2):
            _, p = ttest_ind(g1, g2, equal_var=False)
            add_sig(ax, x_positions[groups_labels[0]][c], x_positions[groups_labels[1]][c], p, row=r)
            r += 1


# --- Subplot 1: Number of bouts ---
plot_data(axes[0], all_data_counts, ylabel="Number of bouts", title="Bout counts")

# --- Subplot 2: Frequency (bouts/sec) ---
freq_data = {lab: {c: all_data_counts[lab][c] / RECORDING_DURATION_SEC for c in conds_all}
             for lab in groups_labels}
plot_data(axes[1], freq_data, ylabel="Frequency (bouts/s)", title="Bout frequency")

# --- Subplot 3: Mean bout duration (s) ---
plot_data(axes[2], all_data_dur, ylabel="Bout duration (s)", title="Mean bout duration")

# --- Legend ---
handles = [plt.Line2D([0], [0], marker='o', color=c, label=lab, markersize=8, lw=0)
           for lab, c in zip(groups_labels, colors)]
axes[0].legend(handles=handles, loc='best')

plt.tight_layout()
plt.show()



