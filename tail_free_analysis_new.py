# -*- coding: utf-8 -*-
# loading data
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, pointbiserialr, ks_2samp, ttest_ind, ttest_rel, f_oneway, chi2_contingency, \
    mannwhitneyu, wilcoxon
import matplotlib.patches as patches
import pingouin as pg
import warnings

warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'Mean of empty slice')

data_path = r'C:\Users\jonathat.WISMAIN\OneDrive - weizmann.ac.il\Desktop\tail-free\big_stress' #TODO - change path

figures_path = join(data_path, 'figures')
plt.close('all')
root_dir = r'Z:\Jonathan\tail_free'
dates = [r'20241103', r'20241117', r'20241124', r'20241202']  # 15 - 5 - 10 experiment
# dates = [ r'20240915', r'20240917', r'20240922', r'20240923', r'20240924'] # 20 - 5 - 25 experiment


dir_list = [join(root_dir, date, file) for date in dates for file in os.listdir(os.path.join(root_dir, date))]
n_fish = len(dir_list) / 3
file_names = [file[:-6] for date in dates for file in os.listdir(os.path.join(root_dir, date))]

# %% create means dataframe (means of parameters for each fish)
params = ['swim_mean_signal', 'swim_max_signal', 'swim_duration', 'swim_LR_balance', 'swim_frequency']
df_means = pd.DataFrame(index=file_names,
                        columns=['fish_name', 'rec_length', 'session', 'group', 'n_bouts',
                                 'swim_mean_signal', 'swim_max_signal', 'swim_duration', 'swim_LR_balance',
                                 'swim_frequency'])
palette = sns.color_palette("tab10")
max_swim_duration = 400
for i, param in enumerate(params):
    before = {'control': [], 'psilocybin': []}
    during = {'control': [], 'psilocybin': []}
    after = {'control': [], 'psilocybin': []}
    print(param)
    for j, directory in enumerate(dir_list):
        filename = file_names[j]
        if filename.split('_')[0][-1] == 'c':
            group = 'control'
        else:
            group = 'psilocybin'
        speed_file = directory + '\\speed_rec.npy'
        swim_file = directory + '\\swim_info.npy'
        speed = np.load(speed_file, allow_pickle=True)[()]

        stim = speed[::400]
        stim[stim > 0] = 1
        swim_info = np.load(swim_file, allow_pickle=True)[()]
        recording_length = len(speed) // 400  # 400fps
        # print(filename, recording_length)
        # include_index = np.where((swim_info['swim_duration'] > 50) & (swim_info['swim_duration'] < max_swim_duration))[0]
        include_index = np.where((swim_info['swim_duration'] > 50) & (swim_info['swim_duration'] < max_swim_duration)
                                 & (swim_info['swim_frequency'] > 0))[0]
        # include_index = np.where((swim_info['swim_duration'] > 50) & (swim_info['swim_frequency'] > 0))[0]                        

        n_bouts = len(include_index)
        # swim_info['bouts_per_min'] = [n_bouts/recording_length*60] * n_bouts
        start_times = swim_info['startSwimInd'] // 400
        stim_vals = stim[start_times.astype(int)]
        swim_info['stim'] = stim_vals
        np.save(join(swim_file), swim_info)
        df_means.iloc[j, :5] = [filename.split('_')[0], recording_length, filename.split('_')[1], group, n_bouts]
        try:
            vals = swim_info[param][include_index]
            val = np.nanmedian(vals)
        except:
            val = swim_info[param]
        df_means.loc[filename, param] = val
        df_means.loc[filename, param] = df_means.loc[filename, param]

        if 'before' in directory:
            before[group].append(val)
        elif 'during' in directory:
            during[group].append(val)
        elif 'after' in directory:
            after[group].append(val)

for i in [1, 4, 5, 6, 7, 8, 9]:
    col = df_means.columns[i]
    df_means[col] = pd.to_numeric(df_means[col], errors='coerce')
df_means = df_means.sort_values(by='fish_name')

df_means['bpm'] = df_means['n_bouts'] / df_means['rec_length']

for session in ['before', 'during', 'after']:
    print(mannwhitneyu(df_means[(df_means['group'] == 'control') & (df_means['session'] == session)]['swim_duration'],
                       df_means[(df_means['group'] == 'psilocybin') & (df_means['session'] == session)][
                           'swim_duration'], nan_policy='omit'))
# %% Plot boxplots for all params
params = ['swim_mean_signal', 'swim_max_signal', 'swim_duration', 'swim_LR_balance', 'swim_frequency', 'n_bouts']

plt.figure(figsize=(12, 8))
palette = {'control': 'gray', 'psilocybin': 'purple'}

for i, param in enumerate(params):
    ax = plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df_means, hue='group', x='session', y=param, palette=palette, ax=ax,
                width=0.4, fliersize=0, linewidth=1.5, boxprops=dict(alpha=0.3))

    sns.stripplot(data=df_means, hue='group', x='session', y=param, palette=palette, dodge=True,
                  alpha=0.8, size=6, jitter=True, ax=ax)

    ctrl_bef = df_means[(df_means['group'] == 'control') & (df_means['session'] == 'before')][param]
    psilo_bef = df_means[(df_means['group'] == 'psilocybin') & (df_means['session'] == 'before')][param]
    ctrl_dur = df_means[(df_means['group'] == 'control') & (df_means['session'] == 'during')][param]
    psilo_dur = df_means[(df_means['group'] == 'psilocybin') & (df_means['session'] == 'during')][param]
    ctrl_aft = df_means[(df_means['group'] == 'control') & (df_means['session'] == 'after')][param]
    psilo_aft = df_means[(df_means['group'] == 'psilocybin') & (df_means['session'] == 'after')][param]

    _, pval_bef = mannwhitneyu(ctrl_bef, psilo_bef, nan_policy='omit')
    _, pval_dur = mannwhitneyu(ctrl_dur, psilo_dur, nan_policy='omit')
    _, pval_aft = mannwhitneyu(ctrl_aft, psilo_aft, nan_policy='omit')

    print(param)
    print('bef', pval_bef)
    print('dur', pval_dur)
    print('aft', pval_aft)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['before (15min)', 'during (5min)', 'after (10min)'], rotation=45)
    ax.set_xlabel('')
    plt.title(param, fontweight='bold', fontsize=13)
    ax.get_legend().remove()

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title='Group', loc='center', bbox_to_anchor=(1.1, 1.2))

plt.tight_layout()
plt.show()


# %% concat all swiminfos in each fish and plot timeseries
def pval2asterik(pvalue, return_pval=False, return_as_str=False, concise=False):
    '''
    Return suitable asteriks based on the pvalue significance.

    Parameters
    ----------
    pvalue : float
        pvalue.
    return_pval : bool, optional
        return non-significant p-value. The default is False.
    return_as_str : bool, optional
        return as str rather than float. The default is False.
    concise : bool, optional
        change all significant values to '*' for a more concise manner. The default is False.

    Returns
    -------
    str
        significance asteriks.

    '''
    if concise:
        if pvalue <= 0.001:
            return '***'
        elif pvalue <= 0.01:
            return '**'
        elif pvalue <= 0.05:
            return '*'
        else:
            return 'ns'
    if pvalue <= 0.00001:
        return '*****'
    if pvalue <= 0.0001:
        return '****'
    if pvalue <= 0.001:
        return '***'
    if pvalue <= 0.01:
        return '**'
    if pvalue <= 0.05:
        return '*'
    if return_pval == True:
        if return_as_str:
            return 'p = {}'.format(round(pvalue, 3))
        return round(pvalue, 3)
    else:
        return 'ns'


fish_names = np.unique([name.split('\\')[-1].split('_')[0] for name in dir_list])
fish_info_d = {fish: None for fish in fish_names}
for i, fish_name in enumerate(fish_names):

    paths = [path for path in dir_list if fish_name in path]
    paths = [paths[i] for i in [1, 2, 0]]  # reorder paths list
    swiminfos = [np.load(path + '/swim_info.npy', allow_pickle=1)[()] for path in paths]
    try:
        for i in range(3):
            del swiminfos[i]['bouts_per_min']
    except:
        print('')
    include_idx = [np.where(swim_info['startSwimInd'] != 0)[0] for swim_info in swiminfos]

    if len(include_idx[1]) == 0:
        include_idx[1] = [0]
    swiminfos = [{k: [v[i] for i in idx] for k, v in d.items()}
                 for d, idx in zip(swiminfos, include_idx)]  # remove the fake bouts
    info_bouts = [len(swiminfo['startSwimInd']) for swiminfo in swiminfos]
    # print(fish_name,info_bouts)
    keys = swiminfos[0].keys()
    merged_swim_info = {key: np.concatenate([d[key] for d in swiminfos]) for key in keys}
    merged_swim_info['swim_start'] = np.concatenate([merged_swim_info['startSwimInd'][:info_bouts[0]],
                                                     merged_swim_info['startSwimInd'][
                                                     info_bouts[0]:info_bouts[1] + info_bouts[0]] + 900 * 400,
                                                     merged_swim_info['startSwimInd'][
                                                     info_bouts[1] + info_bouts[0]:] + 1200 * 400])

    merged_swim_info['swim_end'] = np.concatenate([merged_swim_info['endSwimInd'][:info_bouts[0]],
                                                   merged_swim_info['endSwimInd'][
                                                   info_bouts[0]:info_bouts[1] + info_bouts[0]] + 900 * 400,
                                                   merged_swim_info['endSwimInd'][
                                                   info_bouts[1] + info_bouts[0]:] + 1200 * 400])
    merged_swim_info['swim_start'] = merged_swim_info['swim_start'] // 400
    merged_swim_info['swim_end'] = merged_swim_info['swim_end'] // 400

    fish_info_d[fish_name] = merged_swim_info

for fish_name, swiminfo in fish_info_d.items():
    fish_info_d[fish_name]['fish_name'] = [fish_name] * len(swiminfo['swim_start'])
keys = fish_info_d['f9-c'].keys()
ctrl_dict = {key: val for key, val in fish_info_d.items() if 'c' in key}
ctrl_dict = {key: np.concatenate([d[key] for d in ctrl_dict.values()]) for key in keys}
psilo_dict = {key: val for key, val in fish_info_d.items() if 'p' in key}
psilo_dict = {key: np.concatenate([d[key] for d in psilo_dict.values()]) for key in keys}


def experiment_session(time):
    if time < 900:
        return 'before'
    elif 900 <= time < 1200:
        return 'during'
    elif 1200 <= time < 1800:
        return 'after'


ctrl_df = pd.DataFrame.from_dict(ctrl_dict)
ctrl_df['group'] = len(ctrl_df) * ['control']
ctrl_df['session'] = ctrl_df['swim_start'].apply(experiment_session)
# ctrl_df['stim'] = ctrl_df['swim_start'].apply(stim_value)

psilo_df = pd.DataFrame.from_dict(psilo_dict)
psilo_df['group'] = len(psilo_df) * ['psilocybin']
psilo_df['session'] = psilo_df['swim_start'].apply(experiment_session)
# psilo_df['stim'] = psilo_df['swim_start'].apply(stim_value)

ctrl_df = ctrl_df[(ctrl_df['swim_duration'] > 50) & (ctrl_df['swim_frequency'] > 0)]
psilo_df = psilo_df[(psilo_df['swim_duration'] > 50) & (psilo_df['swim_frequency'] > 0)]
# include_index = np.where((swim_info['swim_duration'] > 50) & (swim_info['swim_duration'] < max_swim_duration)
#                          & (swim_info['swim_frequency'] > 0))[0]

ctrl_df = ctrl_df.iloc[:, 2:]
psilo_df = psilo_df.iloc[:, 2:]

df_data = pd.concat([ctrl_df, psilo_df], ignore_index=True)
import itertools

numeric_cols = ['swim_duration', 'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance', 'swim_frequency']

agg_df = df_data.groupby(['fish_name', 'stim', 'group', 'session'], as_index=False).agg(
    {col: 'mean' for col in numeric_cols})
n_bouts = df_data.groupby(['fish_name', 'stim', 'group', 'session'], as_index=False).size().rename(
    columns={'size': 'n_bouts'})

# Merge back with mean values
agg_df = agg_df.merge(n_bouts, on=['fish_name', 'stim', 'group', 'session'])

sessions = ['before', 'during', 'after']
stims = [0, 1]
fish_names = df_data['fish_name'].unique()

combs = list(itertools.product(sessions, stims))

all_combinations = pd.MultiIndex.from_product(
    [fish_names, stims, sessions],
    names=['fish_name', 'stim', 'session']
).to_frame(index=False)

# Merge with agg_df to ensure all combinations are present
agg_df = all_combinations.merge(agg_df, on=['fish_name', 'stim', 'session'], how='left')

# # Fill missing values: NaN for swim parameters, 0 for n_bouts
agg_df.fillna({'n_bouts': 0}, inplace=True)
agg_df.loc[agg_df['fish_name'].str.contains('c', na=False), 'group'] = 'control'
agg_df.loc[agg_df['fish_name'].str.contains('p', na=False), 'group'] = 'psilocybin'
agg_df['combined_group'] = agg_df['group'] + '_' + agg_df['stim'].astype(str)

df2 = agg_df.copy()

#
df2['bpm'] = np.select(
    [df2['session'] == 'before', df2['session'] == 'during', df2['session'] == 'after'],
    [df2['n_bouts'] / 15, df2['n_bouts'] / 5, df2['n_bouts'] / 10],
    default=df2['n_bouts']  # Keep original value if session is not 'before', 'during', or 'after'
)


def adjust_n_bouts(row):
    if row['session'] == 'before':
        return row['n_bouts'] / 15
    elif row['session'] == 'during':
        return row['n_bouts'] / 5
    elif row['session'] == 'after':
        return row['n_bouts'] / 10
    else:  # Handle cases where the session value is not one of the specified options
        return row['n_bouts']


df2['bpm'] = df2.apply(adjust_n_bouts, axis=1)

print(df2)
# %%

ime_vec = np.arange(1800)  # 1800 seconds
param = 'swim_mean_signal'
from scipy.interpolate import make_smoothing_spline


def plot_param_time(param, fish_info_d):
    units = {'swim_duration': '(ms)', 'swim_max_signal': '', 'swim_mean_signal': '', 'swim_LR_balance': '',
             'swim_frequency': '(Hz)'}
    fig, ax = plt.subplots(figsize=(12, 8))

    time_c = np.zeros((1,))
    vals_c = np.zeros((1,))

    time_p = np.zeros((1,))
    vals_p = np.zeros((1,))

    for fish_name, swiminfo in fish_info_d.items():

        vals = swiminfo[param]
        start_times = swiminfo['swim_start']

        include_index = np.where((swiminfo['swim_duration'] > 100) & (swiminfo['swim_duration'] < max_swim_duration))[0]
        # spont_idx = np.where((swiminfo['stim'] ==0))[0]
        # omr_idx = np.where((swiminfo['stim'] ==1))[0]
        vals, start_times = vals[include_index], start_times[include_index]
        vals = vals[start_times < 1800]
        start_times = start_times[start_times < 1800]

        if 'c' in fish_name:
            time_c = np.append(time_c, start_times)
            vals_c = np.append(vals_c, vals)

        elif 'p' in fish_name:
            time_p = np.append(time_p, start_times)
            vals_p = np.append(vals_p, vals)

        # if fish_name == 'f10-c':
        #     vals_c = vals_c[time_c<1200]
        #     time_c = time_c[time_c<1200]
    time_c = np.delete(time_c, 0)
    vals_c = np.delete(vals_c, 0)
    idx1 = np.argsort(time_c)
    time_c = time_c[idx1]
    vals_c = vals_c[idx1]
    idx1 = np.unique(time_c, return_index=True)[1]
    time_c = time_c[idx1]
    vals_c = vals_c[idx1]

    time_p = np.delete(time_p, 0)
    vals_p = np.delete(vals_p, 0)
    idx2 = np.argsort(time_p)
    time_p = time_p[idx2]
    vals_p = vals_p[idx2]
    idx2 = np.unique(time_p, return_index=True)[1]
    time_p = time_p[idx2]
    vals_p = vals_p[idx2]

    t = np.arange(0, 1800, 30)
    spl1 = make_smoothing_spline(time_c, vals_c, lam=10000000)
    spl2 = make_smoothing_spline(time_p, vals_p, lam=10000000)
    plt.scatter(time_c, vals_c, s=2, color='orange')
    plt.scatter(time_p, vals_p, s=2, color='purple')
    plt.plot(t, spl1(t), color='orange', linewidth=3)
    plt.plot(t, spl2(t), color='purple', linewidth=3)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([ax.get_ylim()[0], max(np.percentile(vals_c, 97), np.percentile(vals_p, 97))])
    if param == 'swim_LR_balance':
        plt.ylim([ax.get_ylim()[0], 1])
    ymin, ymax = ax.get_ylim()
    plt.ylabel(f'{param.replace("_", " ")} {units[param]}', fontsize=12)

    rect = patches.Rectangle((900, ymin), 300, ax.get_ylim()[1] - ax.get_ylim()[0], linewidth=1, edgecolor='cyan',
                             facecolor='cyan', alpha=0.2)
    ax.add_patch(rect)
    plt.text(x=950, y=ymax * 0.95, s='cold stress', color='teal', fontweight='bold', fontsize=11)

    # ast_before = pval2asterik(ks_2samp(vals_c[time_c<900], vals_p[time_p<900])[1])
    # plt.text(x = 400, y = ymax*0.9, s = ast_before, fontsize = 12, fontweight='bold', color='black')
    # ast_during = pval2asterik(ks_2samp(vals_c[np.where(np.logical_and(time_c>=900, time_c<=1200))],
    #                                    vals_p[np.where(np.logical_and(time_p>=900, time_p<=1200))])[1])
    # plt.text(x = 1000, y = ymax*0.9, s = ast_during, fontsize = 12, fontweight='bold', color='black')
    # ast_after = pval2asterik(ks_2samp(vals_c[time_c>1200], vals_p[time_p>1200])[1])
    # plt.text(x = 1500, y = ymax*0.9, s = ast_after, fontsize = 12, fontweight='bold', color='black')

    plt.title(param, fontweight='bold', fontsize=14)
    plt.legend(['control', 'psilocybin'], fontsize=14, loc='upper left', markerscale=2)
    plt.savefig(join(figures_path, f'{param}.jpg'), dpi=1200, bbox_inches='tight')
    plt.show()


params = ['swim_duration', 'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance', 'swim_frequency']
for param in params:
    plot_param_time(param, fish_info_d)


def get_param_vals(fish_info_d, group, param):
    times = np.zeros((1,))
    all_vals = np.zeros((1,))
    fish_info_d = {key: vals for (key, vals) in fish_info_d.items() if group in key}
    for fish_name, swiminfo in fish_info_d.items():
        vals = swiminfo[param]
        start_times = swiminfo['swim_start']

        include_index = np.where((swiminfo['swim_duration'] > 100) & (swiminfo['swim_duration'] < max_swim_duration))[0]

        vals, start_times = vals[include_index], start_times[include_index]
        vals = vals[start_times < 1800]
        start_times = start_times[start_times < 1800]

        times = np.append(times, start_times)
        all_vals = np.append(all_vals, vals)

    times = np.delete(times, 0)
    all_vals = np.delete(all_vals, 0)
    idx1 = np.argsort(times)
    times = times[idx1]
    all_vals = all_vals[idx1]
    idx1 = np.unique(times, return_index=True)[1]
    times = times[idx1]
    all_vals = all_vals[idx1]
    return all_vals, times


param = 'swim_LR_balance'
vals_c, time_c = get_param_vals(fish_info_d, 'c', param)
vals_p, time_p = get_param_vals(fish_info_d, 'p', param)


def get_fish_times(df_data, fish_name):
    fish_times = df_data[df_data['fish_name'] == fish_name]['swim_start']
    return np.array(fish_times)


# ctrl_times = {}
# c_names = [n for n in fish_info_d.keys() if 'c' in n]
# for fish_name in c_names:
#     ctrl_times[fish_name] = get_fish_times(df_data, fish_name)

# psilo_times = {}
# p_names = [n for n in fish_info_d.keys() if 'p' in n]
# for fish_name in p_names:
#     psilo_times[fish_name] = get_fish_times(df_data, fish_name)
# %%
time_c = np.zeros((1,))
time_p = np.zeros((1,))
for fish_name, swiminfo in fish_info_d.items():
    vals = swiminfo['swim_duration']
    start_times = swiminfo['swim_start']
    include_index = np.where((swiminfo['swim_duration'] > 100) & (swiminfo['swim_duration'] < max_swim_duration))[0]
    # include_index = np.where((swim_info['swim_duration'] > 50) & (swim_info['swim_duration'] < max_swim_duration)
    #                          & (swim_info['swim_frequency'] > 0))[0]
    vals, start_times = vals[include_index], start_times[include_index]
    vals = vals[start_times < 1800]
    start_times = start_times[start_times < 1800]
    if 'c' in fish_name:
        time_c = np.append(time_c, start_times)
    elif 'p' in fish_name:
        time_p = np.append(time_p, start_times)

time_c = np.delete(time_c, 0)

idx1 = np.argsort(time_c)
time_c = time_c[idx1]
idx1 = np.unique(time_c, return_index=True)[1]
time_c = time_c[idx1]

time_p = np.delete(time_p, 0)
idx2 = np.argsort(time_p)
time_p = time_p[idx2]
idx2 = np.unique(time_p, return_index=True)[1]
time_p = time_p[idx2]

# %%
# Define time bins
time_bins = np.linspace(0, 1800, 19)  # 6 bins of 300s each
group1_hist, _ = np.histogram(time_c, bins=time_bins)
group2_hist, _ = np.histogram(time_p, bins=time_bins)

heatmap_data = pd.DataFrame({
    "Control": group1_hist,
    "Psilocybin": group2_hist
}, index=[f"{int(time_bins[i])}-{int(time_bins[i + 1])}" for i in range(len(time_bins) - 1)])

# Heatmap plot
plt.figure(figsize=(8, 6))
ax = sns.heatmap(heatmap_data.T, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={'label': 'Swim Event Count'})
plt.title("Swim Events Per Time Bin")
# plt.ylabel("Group")
plt.xlabel("Time Bin (s)")
plt.yticks(fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
rect = plt.Rectangle((9, 0), 3, 18, linewidth=2, edgecolor='blue', linestyle='--', fill=0)
ax.add_patch(rect)
y_labels = ['Control', 'Psilocybin']
y_colors = ['gray', 'purple']
ax.set_yticklabels(y_labels, fontweight='bold')
for tick_label, color in zip(ax.get_yticklabels(), y_colors):
    tick_label.set_color(color)
# plt.savefig(join(figures_path, 'swim_bins.jpg'), dpi=1000)
plt.show()

# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(counts_df.T, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={'label': 'Mean Event Count'})


# plt.title("Swim Events Over Time")
# plt.xlabel("Time Bin (s)")
# y_labels = ['Control', 'Psilocybin']
# y_colors = ['gray', 'purple']
# ax.set_yticklabels(y_labels, fontweight='bold') 
# for tick_label, color in zip(ax.get_yticklabels(), y_colors):
#     tick_label.set_color(color)
# plt.tight_layout()
# # plt.savefig(r'C:\Users\jonathat.WISMAIN\OneDrive - weizmann.ac.il\Desktop\tail-free/swim_event_counts.jpg', dpi=1200, bbox_inches='tight')
# plt.show()
# %%
from matplotlib.lines import Line2D


def plot_param_time(param, fish_info_d_strong, fish_info_d_mild):
    units = {'swim_duration': '(ms)', 'swim_max_signal': '', 'swim_mean_signal': '', 'swim_LR_balance': '',
             'swim_frequency': '(Hz)'}
    fig, ax = plt.subplots(figsize=(12, 8))

    time_c = np.zeros((1,))
    vals_c = np.zeros((1,))

    time_p = np.zeros((1,))
    vals_p = np.zeros((1,))

    for fish_name, swiminfo in fish_info_d_strong.items():
        vals = swiminfo[param]
        start_times = swiminfo['swim_start']
        include_index = np.where((swiminfo['swim_duration'] > 100) & (swiminfo['swim_duration'] < max_swim_duration))[0]
        vals, start_times = vals[include_index], start_times[include_index]
        vals = vals[start_times < 1800]
        start_times = start_times[start_times < 1800]
        if 'c' in fish_name:
            time_c = np.append(time_c, start_times)
            vals_c = np.append(vals_c, vals)
        elif 'p' in fish_name:
            time_p = np.append(time_p, start_times)
            vals_p = np.append(vals_p, vals)

    time_c = np.delete(time_c, 0)
    vals_c = np.delete(vals_c, 0)
    idx1 = np.argsort(time_c)
    time_c = time_c[idx1]
    vals_c = vals_c[idx1]
    idx1 = np.unique(time_c, return_index=True)[1]
    time_c = time_c[idx1]
    vals_c = vals_c[idx1]

    time_p = np.delete(time_p, 0)
    vals_p = np.delete(vals_p, 0)
    idx2 = np.argsort(time_p)
    time_p = time_p[idx2]
    vals_p = vals_p[idx2]
    idx2 = np.unique(time_p, return_index=True)[1]
    time_p = time_p[idx2]
    vals_p = vals_p[idx2]

    t = np.arange(0, 1800, 30)
    spl1 = make_smoothing_spline(time_c, vals_c, lam=1000000)
    spl2 = make_smoothing_spline(time_p, vals_p, lam=1000000)
    plt.scatter(time_c, vals_c, s=2, color='orange', alpha=.3)
    plt.scatter(time_p, vals_p, s=2, color='purple', alpha=.3)
    plt.plot(t, spl1(t), color='orange', linewidth=3)
    plt.plot(t, spl2(t), color='purple', linewidth=3)

    time_c = np.zeros((1,))
    vals_c = np.zeros((1,))

    time_p = np.zeros((1,))
    vals_p = np.zeros((1,))

    for fish_name, swiminfo in fish_info_d_mild.items():
        vals = swiminfo[param]
        start_times = swiminfo['swim_start']
        include_index = np.where((swiminfo['swim_duration'] > 100) & (swiminfo['swim_duration'] < max_swim_duration))[0]
        vals, start_times = vals[include_index], start_times[include_index]
        vals = vals[start_times < 1800]
        start_times = start_times[start_times < 1800]
        if 'c' in fish_name:
            time_c = np.append(time_c, start_times)
            vals_c = np.append(vals_c, vals)
        elif 'p' in fish_name:
            time_p = np.append(time_p, start_times)
            vals_p = np.append(vals_p, vals)

    time_c = np.delete(time_c, 0)
    vals_c = np.delete(vals_c, 0)
    idx1 = np.argsort(time_c)
    time_c = time_c[idx1]
    vals_c = vals_c[idx1]
    idx1 = np.unique(time_c, return_index=True)[1]
    time_c = time_c[idx1]
    vals_c = vals_c[idx1]

    time_p = np.delete(time_p, 0)
    vals_p = np.delete(vals_p, 0)
    idx2 = np.argsort(time_p)
    time_p = time_p[idx2]
    vals_p = vals_p[idx2]
    idx2 = np.unique(time_p, return_index=True)[1]
    time_p = time_p[idx2]
    vals_p = vals_p[idx2]

    t = np.arange(0, 1800, 30)
    spl1 = make_smoothing_spline(time_c, vals_c, lam=1000000)
    spl2 = make_smoothing_spline(time_p, vals_p, lam=1000000)
    plt.scatter(time_c, vals_c, s=2, color='orange', marker='*', alpha=.3)
    plt.scatter(time_p, vals_p, s=2, color='purple', marker='*', alpha=.3)
    plt.plot(t, spl1(t), color='orange', linewidth=3, linestyle='--')
    plt.plot(t, spl2(t), color='purple', linewidth=3, linestyle='--')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([ax.get_ylim()[0], max(np.percentile(vals_c, 97), np.percentile(vals_p, 97))])
    if param == 'swim_LR_balance':
        plt.ylim([0, 1])
    ymin, ymax = ax.get_ylim()
    plt.ylabel(f'{param.replace("_", " ")} {units[param]}', fontsize=12)

    rect = patches.Rectangle((900, ymin), 300, ax.get_ylim()[1] - ax.get_ylim()[0], linewidth=1, edgecolor='cyan',
                             facecolor='cyan', alpha=0.2)
    ax.add_patch(rect)
    plt.text(x=950, y=ymax * 0.95, s='cold stress', color='teal', fontweight='bold', fontsize=11)

    # ast_before = pval2asterik(ks_2samp(vals_c[time_c<900], vals_p[time_p<900])[1])
    # plt.text(x = 400, y = ymax*0.9, s = ast_before, fontsize = 12, fontweight='bold', color='black')
    # ast_during = pval2asterik(ks_2samp(vals_c[np.where(np.logical_and(time_c>=900, time_c<=1200))],
    #                                    vals_p[np.where(np.logical_and(time_p>=900, time_p<=1200))])[1])
    # plt.text(x = 1000, y = ymax*0.9, s = ast_during, fontsize = 12, fontweight='bold', color='black')
    # ast_after = pval2asterik(ks_2samp(vals_c[time_c>1200], vals_p[time_p>1200])[1])
    # plt.text(x = 1500, y = ymax*0.9, s = ast_after, fontsize = 12, fontweight='bold', color='black')

    plt.title(param, fontweight='bold', fontsize=14)

    control = Line2D([], [], color='orange', marker='o', linestyle='None', label='control', markersize=3)
    psilocybin = Line2D([], [], color='purple', marker='o', linestyle='None', label='psilocybin', markersize=3)
    mild_stress = Line2D([], [], color='blue', linestyle='--', linewidth=2, label='mild stress')
    strong_stress = Line2D([], [], color='blue', linestyle='-', linewidth=2, label='strong stress')
    plt.legend(handles=[control, psilocybin, mild_stress, strong_stress], fontsize=14, loc='upper left', markerscale=2)

    plt.savefig(join(r'C:\Users\jonathat.WISMAIN\OneDrive - weizmann.ac.il\Desktop\tail-free', f'combined_{param}.pdf'),
                dpi=1200, bbox_inches='tight')
    plt.show()


rootpath = r'C:\Users\jonathat.WISMAIN\OneDrive - weizmann.ac.il\Desktop\tail-free'
fish_info_d_strong = np.load(join(rootpath, 'big_stress/fish_info_d.npy'), allow_pickle=1)[()]

fish_info_d_mild = np.load(join(rootpath, 'mild_stress/fish_info_d.npy'), allow_pickle=1)[()]

params = ['swim_duration', 'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance', 'swim_frequency']
for param in params:
    plot_param_time(param, fish_info_d_strong, fish_info_d_mild)


# plot_param_time('swim_LR_balance', fish_info_d_strong, fish_info_d_mild)

# %%
def pval2asterik(pvalue, return_pval=False, return_as_str=False, concise=False):
    if concise:
        if pvalue <= 0.001:
            return '***'
        elif pvalue <= 0.01:
            return '**'
        elif pvalue <= 0.05:
            return '*'
        else:
            return 'ns'
    if pvalue <= 0.00001:
        return '*****'
    if pvalue <= 0.0001:
        return '****'
    if pvalue <= 0.001:
        return '***'
    if pvalue <= 0.01:
        return '**'
    if pvalue <= 0.05:
        return '*'
    if return_pval == True:
        if return_as_str:
            return 'p = {}'.format(round(pvalue, 3))
        return round(pvalue, 3)
    else:
        return 'ns'


def get_trial_value(swiminfo, trial):
    '''
    Gets the swiminfo values for the specified trial type

    Parameters
    ----------
    swiminfo : dict
    trial : str
        spont or omr.

    Returns
    -------
    task_dict : dict
        dictionary with keys for parameters and values within.
    mean_dict : dict
        mean values for each parameter for the specified trial type.

    '''
    if trial == 'spont' or trial == 'spontaneous':
        task = 0
    elif trial == 'omr' or trial == 'OMR':
        task = 1
    indices = [i for i, value in enumerate(swiminfo['stim']) if value == task]
    if len(indices) == 0:
        task_dict, mean_dict = {}, {}
        return task_dict, mean_dict
    # print(f'{len(indices)} bouts for {trial}')
    task_dict = {key: np.array([value[i] for i in indices]) for key, value in swiminfo.items()}
    task_dict['n_bouts'] = len(indices)
    # del task_dict['task1']; del task_dict['task2']
    mean_dict = {key: np.nanmean(value) for key, value in task_dict.items()}
    return task_dict, mean_dict


# session = input('Which session?: ')
session = 'after'
cols = ['swim_duration', 'swim_frequency',
        'swim_max_signal', 'swim_LR_balance', 'group', 'trial']
df = pd.DataFrame(columns=cols)
groups = ['control', 'psilocybin']
ctrl = [join(path, 'swim_info.npy') for path in dir_list if path.split("_")[1][-1] == 'c' and session in path]
psilo = [join(path, 'swim_info.npy') for path in dir_list if path.split("_")[1][-1] == 'p' and session in path]

for trial in ['spont', 'omr']:
    for i, group in enumerate([ctrl, psilo]):
        for fish_path in group:
            try:
                fish_name = fish_path.split('\\')[-2][:-6]
            except:
                print(fish_path)
            swiminfo = np.load(fish_path, allow_pickle=True)[()]
            try:
                del swiminfo['bouts_per_min']
            except:
                print('')
            task_dict, mean_dict = get_trial_value(swiminfo, trial)
            df_tmp = df_tmp = pd.DataFrame.from_dict(task_dict)
            df_tmp = df_tmp.drop(columns=[col for col in df_tmp.columns if col not in cols])
            df_tmp['group'] = [groups[i]] * len(df_tmp)
            df_tmp['trial'] = [trial] * len(df_tmp)
            df_tmp['fish_name'] = [fish_name] * len(df_tmp)
            df = pd.concat([df, df_tmp], ignore_index=1)
# df = df[(df['swim_frequency'] > 0) & (df['swim_duration'] >= 30) & (df['swim_duration'] < 400)]
df = df[(df['swim_frequency'] > 0) & (df['swim_duration'] >= 50)]
# df.to_csv(join(files_path, 'df_swim_compare.csv'))

variables = ['swim_max_signal', 'swim_duration', 'swim_frequency', 'swim_LR_balance']

# plt.figure(figsize=(12,4))
x_range = [[0, 80], [0, 600], [10, 50], [0, 1], [0, .15]]
titles = ['Max amplitude', 'Swim duration (ms)', 'Swim frequency (Hz)', 'Swim_LR_balance']

fig, axs = plt.subplots(2, 4, figsize=(14, 6))  # Adjust figsize as needed

handles = []
labels = []
for i in range(4):
    ylims = []

    ax1 = axs[0, i]
    ax2 = axs[1, i]
    variable = variables[i]
    if i < 4:
        bww = 0.2
    elif i == 4:
        bww = 0.1
    data1 = df[(df['trial'] == 'spont') & (df['group'] == 'control')]
    data2 = df[(df['trial'] == 'spont') & (df['group'] == 'psilocybin')]
    stat, pval = ks_2samp(data1[variable], data2[variable])
    pval = pval2asterik(pval)
    sns.kdeplot(data=data1,
                x=variable, bw_method=bww, clip=x_range[i], color='gray', ax=ax1, label='control', linewidth=2)
    sns.kdeplot(data=data2,
                x=variable, bw_method=bww, clip=x_range[i], color='purple', ax=ax1, label='psilocybin', linewidth=2)
    ax1.text(x=ax1.get_xlim()[1] * 0.75, y=ax1.get_ylim()[1] * 0.75, s=pval)
    ylims.append(ax1.get_ylim()[1])
    ax1.set_xlim(x_range[i])
    ax1.set_title(titles[i], fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('spontaneous', fontweight='bold', fontsize=14, color='gray')
    if i != 0:
        ax1.set_ylabel('')

    data1 = df[(df['trial'] == 'omr') & (df['group'] == 'control')]
    data2 = df[(df['trial'] == 'omr') & (df['group'] == 'psilocybin')]
    stat, pval = ks_2samp(data1[variable], data2[variable])
    pval = pval2asterik(pval)

    sns.kdeplot(data=data1,
                x=variable, bw_method=bww, clip=x_range[i], color='gray', ax=ax2, label='control', linewidth=2)
    sns.kdeplot(data=data2,
                x=variable, bw_method=bww, clip=x_range[i], color='purple', ax=ax2, label='psilocybin', linewidth=2)
    ax2.text(x=ax2.get_xlim()[1] * 0.75, y=ax2.get_ylim()[1] * 0.75, s=pval)
    ylims.append(ax2.get_ylim()[1])
    ylim = max(ylims)

    ax1.set_ylim([0, ylim])
    ax2.set_ylim([0, ylim])

    ax2.set_xlim(x_range[i])
    ax2.set_xlabel('')
    ax2.set_ylabel('OMR', fontweight='bold', fontsize=14, color='gray')
    if i != 0:
        ax2.set_ylabel('')

    # Collect handles and labels for the legend
    if i == 0:
        h, l = ax1.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

# Add legend outside the subplots, centered below
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.tight_layout()
plt.savefig(join(r'C:\Users\jonathat.WISMAIN\OneDrive - weizmann.ac.il\Desktop\tail-free',
                 f'group_trial_swim_param_{session}.jpg'), dpi=1500, bbox_inches='tight')
plt.suptitle(f'{session.capitalize()} stress', fontsize=14, fontweight='bold', y=1.05)
plt.show()

# %%
sns.set_style('white')
df_data_bef = df_data[df_data['session'] == 'before']
df_data_dur = df_data[df_data['session'] == 'during']
df_data_aft = df_data[df_data['session'] == 'after']
# mapping = {'before': -1, 'during': 0, 'after': 1}
plt.figure(figsize=(6, 5))
ax = sns.kdeplot(data=df_data_aft, hue='group', y='stim', x='swim_LR_balance', alpha=.6, linewidth=1,
                 palette=['darkgray', 'purple'], levels=8,
                 common_norm=False)

df_data_aft['stim_jitter'] = df_data_aft['stim'] + np.random.normal(scale=0.05, size=len(
    df_data_aft))  # Adjust scale for more/less jitter

for collection in ax.collections:
    collection.set_alpha(0.5)  # Set transparency level for each contour line
    collection.set_linewidths(np.linspace(.5, 2, len(ax.collections)))  # Thicker lines in center, thinner outward

sns.scatterplot(
    data=df_data_aft, x='swim_LR_balance', y='stim_jitter', hue='group',
    palette=['gray', 'purple'], alpha=0.2, edgecolor='none', s=10)

plt.yticks([0, 1], ['Spont', 'OMR'], fontsize=16, rotation=90)
sns.despine(top=True)

# handles, labels = ax.get_legend_handles_labels()
# if handles:
#     # Modify alpha for the legend handles to remove transparency
#     for handle in handles:
#         handle.set_alpha(1)  # Set alpha to 1 to remove transparency
#     ax.legend(handles=handles, labels=labels, title="Group", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.ylabel('')
plt.savefig(join(figures_path, 'swim_lr_stim_after.pdf'), dpi=800, bbox_inches='tight')
plt.show()
# %%
d_df = {}
params = ['swim_mean_signal', 'swim_max_signal', 'swim_duration', 'swim_LR_balance', 'swim_frequency']

for session in ['before', 'during', 'after']:
    df_res = pd.DataFrame(index=params, columns=['spont', 'omr'])
    print(session)
    for param in ['swim_mean_signal', 'swim_max_signal', 'swim_duration', 'swim_LR_balance', 'swim_frequency']:
        print(param)
        ctrl_tmp_spont = ctrl_df[(ctrl_df['stim'] == 0) & (ctrl_df['session'] == session)].groupby('fish_name')[
            param].mean()
        ctrl_tmp_omr = ctrl_df[(ctrl_df['stim'] == 1) & (ctrl_df['session'] == session)].groupby('fish_name')[
            param].mean()
        psilo_tmp_spont = psilo_df[(psilo_df['stim'] == 0) & (psilo_df['session'] == session)].groupby('fish_name')[
            param].mean()
        psilo_tmp_omr = psilo_df[(psilo_df['stim'] == 1) & (psilo_df['session'] == session)].groupby('fish_name')[
            param].mean()

        # df_res.loc[param, 'spont'] = ttest_ind(ctrl_tmp_spont,psilo_tmp_spont)[1]
        # df_res.loc[param, 'omr'] = ttest_ind(ctrl_tmp_omr,psilo_tmp_omr)[1]

        df_res.loc[param, 'c'] = mannwhitneyu(ctrl_tmp_spont, psilo_tmp_spont)[1]
        df_res.loc[param, 'p'] = mannwhitneyu(ctrl_tmp_omr, psilo_tmp_omr)[1]

    d_df[session] = df_res
    print('---')
# %%

ctrl_df_means = df_means[df_means['group'] == 'control']
psilo_df_means = df_means[df_means['group'] == 'psilocybin']

for session in ['before', 'during', 'after']:
    print(session)
    for param in ['n_bouts',
                  'swim_mean_signal', 'swim_max_signal', 'swim_duration',
                  'swim_LR_balance', 'swim_frequency', 'bpm']:
        print(param)
        print(mannwhitneyu(ctrl_df_means[ctrl_df_means['session'] == session][param],
                           psilo_df_means[psilo_df_means['session'] == session][param]), '\n')
    print('---')

# %%

# Aggregate data for plotting (mean)
plot_df = df.groupby(['session', 'group', 'stim'])['n_bouts'].mean().reset_index()

# Convert stim to string for better labeling
plot_df['stim'] = plot_df['stim'].astype(str).replace({'0': 'spont', '1': 'OMR'})

plt.figure(figsize=(8, 6))

# Stripplot overlay (optional, but adds detail)
sns.stripplot(x='session', y='n_bouts', hue='combined_group', data=df[df['stim'] == 0],
              palette={'control_0': 'lightgrey', 'control_1': 'white', 'psilocybin_0': 'violet',
                       'psilocybin_1': 'white'},
              dodge=True, jitter=True, alpha=0.5, size=6, zorder=0)
sns.stripplot(x='session', y='n_bouts', hue='combined_group', data=df[df['stim'] == 1], marker='v',
              palette={'control_0': 'white', 'control_1': 'dimgrey', 'psilocybin_0': 'white', 'psilocybin_1': 'indigo'},
              dodge=True, jitter=True, alpha=0.5, size=6, zorder=0)
# for i in range(12):
#     sns.scatterplot(x = 1, y = plot_df['n_bouts'][0], size = 4, color='green')
# ... (Rest of your plotting code for labels, titles, legend, etc.) ...
plt.xlabel("Session")
plt.ylabel("Number of Bouts")
plt.title("Number of Bouts Across Sessions, Groups, and Stimuli")

# Improve legend
handles, labels = plt.gca().get_legend_handles_labels()
new_handles = handles[:2] + handles[4:]  # Select only line handles
new_labels = labels[:2] + labels[4:]  # Select only group labels
plt.legend(new_handles, new_labels, title="Group", loc="upper left")

plt.tight_layout()
plt.show()

# %%
jitter_strength = 0.07  # Adjust this for more/less jitter

sns.set_style('white')
plot_df = df2.groupby(['session', 'group'])['bpm'].mean().reset_index()
session_order = ['before', 'during', 'after']
palette_offset = {'control': ('dimgrey', -0.15), 'psilocybin': ('indigo', 0.15)}
stds = df2.groupby(['session', 'group'])['bpm'].std().reset_index()['bpm']
df3 = df2.groupby(['group', 'session', 'fish_name'], as_index=False).mean(numeric_only=True)

plt.figure(figsize=(8, 6))
for _, row in df3.iterrows():
    x = session_order.index(row['session'])
    y = row['bpm']
    group = row['group']
    color, offset = palette_offset[group]
    jitter = np.random.uniform(-jitter_strength, jitter_strength)
    plt.scatter(x + offset + jitter, y, color=color,  # edgecolors='black',
                s=80, alpha=0.25, zorder=0)

for index, row in plot_df.iterrows():
    x = session_order.index(row['session'])
    y = row['bpm']
    group = row['group']
    color, offset = palette_offset[group]

    plt.errorbar(x + offset, y, yerr=stds[index], color=color, linewidth=2, markersize=12, marker='o',
                 markeredgewidth=1, markeredgecolor='black')  # , mfc='paleturquoise')

plt.ylabel("BPM per Fish", fontsize=24)

# for (fish, group), subdf in df3.groupby(['fish_name', 'group']):
#     subdf = subdf.set_index('session').reindex(session_order)
#     x = [session_order.index(sess) for sess in subdf.index]
#     y = subdf['bpm'].values
#     color, offset = palette_offset[group]
#     x_jittered = [xi + offset for xi in x] 
#     plt.plot(x_jittered, y, color=color, alpha=0.2, linewidth=.8, zorder=0)

handles, labels = [], []
for group in ['control', 'psilocybin']:
    color, offset = palette_offset[group]
    label = f"{group.capitalize()}"
    handle = plt.Line2D([0], [0], color=color, linestyle="", label=label, marker='o')
    handles.append(handle)
    labels.append(label)

session_order = ['before', 'during stress', 'after']

plt.legend(handles, labels, title="", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=15)
plt.xticks(range(len(session_order)), session_order, fontsize=24)
plt.yticks(fontsize=24)
sns.despine(top=True)

group1_bef = df3[(df3['group'] == 'control') & (df3['session'] == 'before')]['bpm']
group1_dur = df3[(df3['group'] == 'control') & (df3['session'] == 'during')]['bpm']
group1_aft = df3[(df3['group'] == 'control') & (df3['session'] == 'after')]['bpm']
group2_bef = df3[(df3['group'] == 'psilocybin') & (df3['session'] == 'before')]['bpm']
group2_dur = df3[(df3['group'] == 'psilocybin') & (df3['session'] == 'during')]['bpm']
group2_aft = df3[(df3['group'] == 'psilocybin') & (df3['session'] == 'after')]['bpm']

p_control_bef = wilcoxon(group1_bef, group1_dur).pvalue  # Control (before vs during)
p_control_aft = wilcoxon(group1_aft, group1_dur).pvalue  # Control (after vs during)

p_psilo_bef = wilcoxon(group2_bef, group2_dur).pvalue  # Control (before vs during)
p_psilo_aft = wilcoxon(group2_aft, group2_dur).pvalue  # Control (after vs during)

p_bef = mannwhitneyu(group1_bef, group2_bef).pvalue
p_dur = mannwhitneyu(group1_dur, group2_dur).pvalue
p_aft = mannwhitneyu(group1_aft, group2_aft).pvalue

plt.plot([-0.2, .2], [16.5, 16.5], color='black', linewidth=1.5)  # before compare
plt.text(0, 16.5, pval2asterik(p_bef), ha='center', fontsize=14)

plt.plot([0.8, 1.2], [16.5, 16.5], color='black', linewidth=1.5)  # during compare
plt.text(1, 16.5, pval2asterik(p_dur), ha='center', fontsize=14)

plt.plot([1.8, 2.2], [16.5, 16.5], color='black', linewidth=1.5)  # after compare
plt.text(2, 16.5, pval2asterik(p_aft), ha='center', fontsize=14)

plt.plot([-0.2, 0.8], [18, 18], color='black', linewidth=1.5)  # control bef-dur
plt.text(0.3, 18, pval2asterik(p_control_bef), ha='center', fontsize=14)

plt.plot([0.8, 1.8], [19, 19], color='black', linewidth=1.5)  # control aft-dur
plt.text(1.3, 19, pval2asterik(p_control_aft), ha='center', fontsize=14)

plt.plot([1.2, 2.2], [20, 20], color='black', linewidth=1.5)  # psilo aft-dur
plt.text(1.7, 20, pval2asterik(p_psilo_aft), ha='center', fontsize=14)

# plt.plot([.2,1.1], [20, 20], color='black', linewidth=1.5)  # psilo aft-dur
# plt.text(0.6, 20.3, pval2asterik(p_psilo_bef), ha='center', fontsize=14)

plt.tight_layout()
# plt.savefig(join(figures_path, 'BPM_comparison_means.pdf'), bbox_inches='tight', dpi=1000)

plt.show()
# %%
jitter_strength = 0.07  # Adjust this for more/less jitter

sns.set_style('white')
plot_df = df2.groupby(['session', 'group', 'stim'])['bpm'].mean().reset_index()
session_order = ['before', 'during', 'after']
palette_offset = {
    'control_0': ('lightgrey', -0.3), 'control_1': ('dimgrey', -0.1),
    'psilocybin_0': ('violet', 0.1), 'psilocybin_1': ('indigo', 0.3)
}
markers = {0: 'o', 1: 'v'}  # Define marker styles

# Convert stim to string for better labeling
plot_df['stim'] = plot_df['stim'].astype(str).replace({'0': 'spont', '1': 'OMR'})
stds = df2.groupby(['session', 'group', 'stim'])['bpm'].std().reset_index()['bpm']
plt.figure(figsize=(8, 6))

# Loop through each data point and manually scatter them with jitter
for _, row in df2.iterrows():
    x = session_order.index(row['session'])  # Convert session to numerical
    y = row['bpm']
    group = row['group']
    stim = row['stim']
    color, offset = palette_offset[f'{group}_{stim}']
    marker = markers[stim]

    jitter = np.random.uniform(-jitter_strength, jitter_strength)  # Add small random jitter

    plt.scatter(x + offset + jitter, y, color=color, marker=marker,  # edgecolors='black',
                s=80, alpha=0.25, zorder=0)

# Error bars remain the same
for index, row in plot_df.iterrows():
    x = session_order.index(row['session'])
    y = row['bpm']
    group = row['group']
    stim = row['stim']
    color = palette_offset[f'{group}_{0 if stim == "spont" else 1}'][0]
    offset = palette_offset[f'{group}_{0 if stim == "spont" else 1}'][1]
    marker = markers[0 if stim == "spont" else 1]

    plt.errorbar(x + offset, y, yerr=stds[index], fmt=marker, color=color, linewidth=2, markersize=12,
                 markeredgewidth=1, markeredgecolor='black')  # , mfc='paleturquoise')

plt.xlabel("Session", fontsize=18, fontweight='bold')
plt.ylabel("BPM per Fish", fontweight='bold')
plt.title("Number of Bouts Across Sessions, Groups, and Stimuli")

# Custom legend
handles, labels = [], []
for group in ['control', 'psilocybin']:
    for stim, stim_label in zip([0, 1], ['spont', 'OMR']):
        label = f"{group.capitalize()} {stim_label}"
        handle = plt.Line2D([0], [0], marker=markers[stim], color=palette_offset[f'{group}_{stim}'][0], linestyle="",
                            label=label)
        handles.append(handle)
        labels.append(label)

session_order = ['before', 'during stress', 'after']

plt.legend(handles, labels, title="", loc="center left", bbox_to_anchor=(1, 0.5))
plt.xticks(range(len(session_order)), session_order, fontsize=16)
sns.despine(top=True)

group1_bef = df2[(df2['group'] == 'control') & (df2['session'] == 'before')]['bpm']
group1_dur = df2[(df2['group'] == 'control') & (df2['session'] == 'during')]['bpm']
group1_aft = df2[(df2['group'] == 'control') & (df2['session'] == 'after')]['bpm']
group2_bef = df2[(df2['group'] == 'psilocybin') & (df2['session'] == 'before')]['bpm']
group2_dur = df2[(df2['group'] == 'psilocybin') & (df2['session'] == 'during')]['bpm']
group2_aft = df2[(df2['group'] == 'psilocybin') & (df2['session'] == 'after')]['bpm']

p_control_bef = wilcoxon(group1_bef, group1_dur).pvalue  # Control (before vs during)
p_control_aft = wilcoxon(group1_aft, group1_dur).pvalue  # Control (after vs during)

p_psilo_bef = wilcoxon(group2_bef, group2_dur).pvalue  # Control (before vs during)
p_psilo_aft = wilcoxon(group2_aft, group2_dur).pvalue  # Control (after vs during)

p_bef = mannwhitneyu(group1_bef, group2_bef).pvalue
p_dur = mannwhitneyu(group1_dur, group2_dur).pvalue
p_aft = mannwhitneyu(group1_aft, group2_aft).pvalue

plt.plot([-0.2, .2], [16.5, 16.5], color='black', linewidth=1.5)  # before compare
plt.text(0, 16.5, pval2asterik(p_bef), ha='center', fontsize=14)

plt.plot([0.8, 1.2], [16.5, 16.5], color='black', linewidth=1.5)  # during compare
plt.text(1, 16.5, pval2asterik(p_dur), ha='center', fontsize=14)

plt.plot([1.8, 2.2], [16.5, 16.5], color='black', linewidth=1.5)  # after compare
plt.text(2, 16.5, pval2asterik(p_aft), ha='center', fontsize=14)

plt.plot([-0.2, 0.8], [18, 18], color='black', linewidth=1.5)  # control bef-dur
plt.text(0.3, 18, pval2asterik(p_control_bef), ha='center', fontsize=14)

plt.plot([0.8, 1.8], [19, 19], color='black', linewidth=1.5)  # control aft-dur
plt.text(1.3, 19, pval2asterik(p_control_aft), ha='center', fontsize=14)

plt.plot([1.2, 2.2], [20, 20], color='black', linewidth=1.5)  # psilo aft-dur
plt.text(1.7, 20, pval2asterik(p_psilo_aft), ha='center', fontsize=14)

plt.tight_layout()
plt.savefig(join(figures_path, 'BPM_comparison_strip_final2.pdf'), bbox_inches='tight', dpi=1000)

plt.show()

# %%
agg_df['session'] = pd.Categorical(agg_df['session'], categories=['before', 'during', 'after'], ordered=True)

sns.set_style("whitegrid")
plt.figure(figsize=(20, 12))

sns.boxplot(x='session', y='bpm', hue='combined_group', data=df2, showfliers=False, fill=.2, width=.8, showmeans=True,
            meanprops={'markersize': 12},
            palette={'control_0': 'lightgrey', 'control_1': 'darkgrey', 'psilocybin_0': 'blueviolet',
                     'psilocybin_1': 'indigo'}, dodge=True)
positions = np.arange(len(df['session'].unique()))
width = 0.8
group_width = width / len(df['combined_group'].unique())
sns.stripplot(x='session', y='bpm', hue='combined_group', data=df2, dodge=True, jitter=True, color='black', size=8,
              alpha=0.3)  # Adjust size and alpha as needed

plt.xlabel("")
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.ylabel("BPM", fontsize=22)
sns.despine(trim=True)

handles = []
labels = []

palette = {'control_0': 'lightgrey', 'control_1': 'darkgrey', 'psilocybin_0': 'blueviolet', 'psilocybin_1': 'indigo'}

for combined_group, color in palette.items():
    handle = plt.Rectangle((0, 0), 1, 1, color=color)
    handles.append(handle)
    label = combined_group.replace("_", " ")
    label = label.capitalize()
    label = label.replace("0", "spont")
    label = label.replace("1", "OMR")
    # Replace 1 with "OMR"
    labels.append(label)

legend = plt.legend(handles, labels, title='Group & Stim', bbox_to_anchor=(1, .5), loc="center left", fontsize=18)
legend.get_title().set_fontsize('20')
plt.tight_layout()
# plt.savefig(join(figures_path, 'BPM_comparison_stim.jpg'), dpi=1000, bbox_inches='tight')
plt.show()

# %%
agg_df['session'] = pd.Categorical(agg_df['session'], categories=['before', 'during', 'after'], ordered=True)

sns.set_style("whitegrid")
plt.figure(figsize=(20, 12))

sns.boxplot(x='session', y='bpm', hue='group', data=df2, showfliers=False, fill=.2, width=.8, showmeans=True,
            meanprops={'markersize': 15},
            palette={'control': 'lightgrey', 'psilocybin': 'blueviolet'}, dodge=True)
positions = np.arange(len(df['session'].unique()))
width = 0.8
group_width = width / len(df['group'].unique())
sns.stripplot(x='session', y='bpm', hue='group', data=df2, dodge=True, jitter=True, color='black', size=8,
              alpha=0.3)  # Adjust size and alpha as needed

plt.xlabel("")
plt.yticks(fontsize=22)
plt.xticks(fontsize=28)
plt.ylabel("BPM per fish", fontsize=30)
sns.despine(trim=True)

handles = []
labels = []

palette = {'control': 'lightgrey', 'psilocybin': 'blueviolet'}

for group, color in palette.items():
    # Create a dummy plot element to serve as a handle
    handle = plt.Rectangle((0, 0), 1, 1, color=color)  # Create a rectangle proxy artist
    handles.append(handle)
    label = group.capitalize()  # Cleaned-up label

    # Replace 1 with "OMR"
    labels.append(label)

legend = plt.legend(handles, labels, title='', bbox_to_anchor=(1, .5), loc="center left", fontsize=18)
legend.get_title().set_fontsize('20')
plt.tight_layout()
# plt.savefig(join(figures_path, 'BPMM_comparison_all.jpg'), dpi=1000, bbox_inches='tight')
plt.show()

# %%

time_bins = np.linspace(0, 1800, 19)
jump = 100
# ctrl_counts = {f"{int(key)}-{int(key+300)}": len(ctrl_df[(ctrl_df['swim_start'] >= key) & (ctrl_df['swim_start'] < key + 300)]) 
#  for key in time_bins[:-1]}


# psilo_counts = {f"{int(key)}-{int(key+300)}": len(psilo_df[(psilo_df['swim_start'] >= key) & (psilo_df['swim_start'] < key + 300)]) 
#  for key in time_bins[:-1]}

ctrl_counts = {f"{int(key)}-{int(key + jump)}": (
    round(len(ctrl_df[(ctrl_df['swim_start'] >= key) & (ctrl_df['swim_start'] < key + jump)]) / 10))
               for key in time_bins[:-1]}

psilo_counts = {f"{int(key)}-{int(key + jump)}": (
    round(len(psilo_df[(psilo_df['swim_start'] >= key) & (psilo_df['swim_start'] < key + jump)]) / 11))
                for key in time_bins[:-1]}

counts_df = pd.DataFrame([ctrl_counts, psilo_counts], index=['Control', 'Psilocybin']).T

plt.figure(figsize=(8, 6))
ax = sns.heatmap(counts_df.T, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={'label': 'Mean Fish Event Count'})
rect = plt.Rectangle((9, 0), 3, len(counts_df), linewidth=2, edgecolor='blue', linestyle='--', fill=0)
ax.add_patch(rect)

plt.title("Mean Swim Events Over Time")
plt.xlabel("Time Bin (s)")
y_labels = ['Control', 'Psilocybin']
y_colors = ['gray', 'purple']
ax.set_yticklabels(y_labels, fontweight='bold')
for tick_label, color in zip(ax.get_yticklabels(), y_colors):
    tick_label.set_color(color)
plt.tight_layout()
# plt.savefig(join(figures_path, 'swim_event_counts_mean.jpg'), dpi=1200, bbox_inches='tight')
plt.show()

# %%
ctrl_fish_counts = {}
psilo_fish_counts = {}
not_norm_ast = {}
time_bins = np.linspace(0, 1800, 19)

# not normalized
for fish_name in ctrl_df['fish_name'].unique():
    fish_df = ctrl_df[ctrl_df['fish_name'] == fish_name]
    fish_counts = {f"{int(key)}-{int(key + jump)}": (
        round(len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + jump)])))
                   for key in time_bins[:-1]}
    ctrl_fish_counts[fish_name] = fish_counts
for fish_name in psilo_df['fish_name'].unique():
    fish_df = psilo_df[psilo_df['fish_name'] == fish_name]
    fish_counts = {f"{int(key)}-{int(key + jump)}": (
        round(len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + jump)])))
                   for key in time_bins[:-1]}
    psilo_fish_counts[fish_name] = fish_counts
for key in ctrl_fish_counts['f1-c'].keys():
    ctrl_vals = [ctrl_fish_counts[fish_name][key] for fish_name in ctrl_df['fish_name'].unique()]
    psilo_vals = [psilo_fish_counts[fish_name][key] for fish_name in psilo_df['fish_name'].unique()]
    # print(key, mannwhitneyu(ctrl_vals, psilo_vals))
    not_norm_ast[key] = pval2asterik(mannwhitneyu(ctrl_vals, psilo_vals, alternative='less')[1])

# %%
# normalized
ctrl_fish_counts = {}
psilo_fish_counts = {}
norm_ast = {}
norm_sig = {}
for fish_name in ctrl_df['fish_name'].unique():
    fish_df = ctrl_df[ctrl_df['fish_name'] == fish_name]
    fish_counts = {f"{int(key)}-{int(key + jump)}": (
        round(len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + jump)])))
                   for key in time_bins[:-1]}
    fish_baseline = np.mean(list(fish_counts.values())[:9])
    ctrl_fish_counts[fish_name] = {key: ((value / fish_baseline)) for key, value in fish_counts.items()}

for fish_name in psilo_df['fish_name'].unique():
    fish_df = psilo_df[psilo_df['fish_name'] == fish_name]
    fish_counts = {f"{int(key)}-{int(key + jump)}": (
        round(len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + jump)])))
                   for key in time_bins[:-1]}
    fish_baseline = np.mean(list(fish_counts.values())[:9])
    # psilo_fish_counts[fish_name] = fish_counts

    psilo_fish_counts[fish_name] = {key: ((value / fish_baseline)) for key, value in fish_counts.items()}
for key in ctrl_fish_counts['f1-c'].keys():
    ctrl_vals = [ctrl_fish_counts[fish_name][key] for fish_name in ctrl_df['fish_name'].unique()]
    psilo_vals = [psilo_fish_counts[fish_name][key] for fish_name in psilo_df['fish_name'].unique()]
    # print(key, np.mean(ctrl_vals), np.mean(psilo_vals))
    # print(key, ctrl_vals, psilo_vals)

    # print(mannwhitneyu(ctrl_vals, psilo_vals, alternative='less'))
    norm_ast[key] = pval2asterik(mannwhitneyu(ctrl_vals, psilo_vals, alternative='less')[1])
    norm_sig[key] = mannwhitneyu(ctrl_vals, psilo_vals, alternative='less')[1]

# %%
sns.set_style("white")

jump = 100
n_bins = int(np.floor(1800 / jump))
time_bins = np.linspace(0, 1800, n_bins + 1)
time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2  # Center of each bin for x-axis

time_bins_labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900',
                    '900-1000',
                    '1000-1100', '1100-1200', '1200-1300', '1300-1400', '1400-1500', '1500-1600', '1600-1700',
                    '1700-1800']

ctrl_counts = {f"{int(key)}-{int(key + jump)}": (
    round(len(ctrl_df[(ctrl_df['swim_start'] >= key) & (ctrl_df['swim_start'] < key + jump)]) / 10))
               for key in time_bins[:-1]}
psilo_counts = {f"{int(key)}-{int(key + jump)}": (
    round(len(psilo_df[(psilo_df['swim_start'] >= key) & (psilo_df['swim_start'] < key + jump)]) / 11))
                for key in time_bins[:-1]}

ctrl_baseline = np.mean(list(ctrl_counts.values())[:9])
psilo_baseline = np.mean(list(psilo_counts.values())[:9])

plt.figure(figsize=(12, 8))
plt.plot(time_bin_centers, list(ctrl_counts.values()), color='gray', linewidth=4, label='Control')
plt.scatter(time_bin_centers, list(ctrl_counts.values()), color='gray')

plt.plot(time_bin_centers, list(psilo_counts.values()), color='purple', linewidth=4, label='Psilocybin')
plt.scatter(time_bin_centers, list(psilo_counts.values()), color='purple')

ax = plt.gca()

for i, group in enumerate([ctrl_df, psilo_df]):
    fish_counts_list = []

    for fish_name in group['fish_name'].unique():
        fish_df = group[group['fish_name'] == fish_name]
        fish_counts = {f"{int(key)}-{int(key + 100)}": round(
            len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + 100)]))
                       for key in range(0, 1800, 100)}
        counts_for_fish = [fish_counts[bin] for bin in time_bins_labels]
        fish_counts_list.append(counts_for_fish)

    fish_counts_matrix = np.array(fish_counts_list)

    group_mean = np.mean(fish_counts_matrix, axis=0)
    group_stds = np.std(fish_counts_matrix, axis=0)
    n_samples = len(fish_counts_list)
    group_stes = group_stds / np.sqrt(n_samples)

    # Fill between for the SEM (group_mean ± group_stes)
    ax.fill_between(time_bin_centers, group_mean + group_stes, group_mean - group_stes,
                    color=['gray', 'purple'][i], alpha=0.1)

time_bins = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900',
             '900-1000', '1000-1100', '1100-1200', '1200-1300', '1300-1400', '1400-1500', '1500-1600', '1600-1700',
             '1700-1800']

# Convert time_bins to numeric positions using list comprehension
time_bin_positions = [(int(bin.split('-')[0]) + int(bin.split('-')[1])) / 2 for bin in time_bins]

rect = patches.Rectangle((time_bin_positions[9], -1.5), time_bin_positions[11] - time_bin_positions[9],
                         ax.get_ylim()[1] + 1.5,  # Make the rectangle span the full height of the plot
                         linewidth=0, edgecolor=None, facecolor='lightskyblue', alpha=0.2)

ax.add_patch(rect)
ax.set_xticks(time_bin_positions)
ax.set_xticklabels(time_bins, rotation=45)

ax.tick_params(axis='x', which='minor', length=4, width=1, direction='in', grid_alpha=0.5)

plt.gca().add_patch(rect)
plt.legend(fontsize=12)
plt.xlabel('Time bins (s)', fontsize=13)
plt.ylabel('Mean Swim events', fontsize=13)
plt.title('Swim Events Over Time', fontsize=15, fontweight='bold')

for time_bin, significance in not_norm_ast.items():

    if significance != 'ns':  # Only plot labels for significance
        x_position = time_bin_positions[time_bins.index(time_bin)]
        y_position = 40  # You can adjust this value to place text above or below the plot

        plt.text(x=x_position, y=y_position, s=significance, color='black', fontsize=20, ha='center')

plt.tight_layout()
plt.savefig(join(figures_path, 'onesideswim_event_plot_sig_correct.pdf'), dpi=600, bbox_inches='tight')
plt.show()

# %%
sns.set_style("white")

jump = 100
n_bins = int(np.floor(1800 / jump))
time_bins = np.linspace(0, 1800, n_bins + 1)
time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2  # Center of each bin for x-axis

time_bins_labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900',
                    '900-1000',
                    '1000-1100', '1100-1200', '1200-1300', '1300-1400', '1400-1500', '1500-1600', '1600-1700',
                    '1700-1800']

ctrl_counts = {f"{int(key)}-{int(key + jump)}": (
    round(len(ctrl_df[(ctrl_df['swim_start'] >= key) & (ctrl_df['swim_start'] < key + jump)]) / 10))
               for key in time_bins[:-1]}
psilo_counts = {f"{int(key)}-{int(key + jump)}": (
    round(len(psilo_df[(psilo_df['swim_start'] >= key) & (psilo_df['swim_start'] < key + jump)]) / 11))
                for key in time_bins[:-1]}

ctrl_baseline = np.mean(list(ctrl_counts.values())[:9])
psilo_baseline = np.mean(list(psilo_counts.values())[:9])

plt.figure(figsize=(12, 8))

plt.plot(time_bin_centers, (np.array(list(ctrl_counts.values())) / ctrl_baseline) * 100 - 100, color='gray',
         linewidth=4, label='Control')
plt.scatter(time_bin_centers, (np.array(list(ctrl_counts.values())) / ctrl_baseline) * 100 - 100, color='gray', s=60)

plt.plot(time_bin_centers, (np.array(list(psilo_counts.values())) / psilo_baseline) * 100 - 100, color='blueviolet',
         linewidth=4, label='Psilocybin')
plt.scatter(time_bin_centers, (np.array(list(psilo_counts.values())) / psilo_baseline) * 100 - 100, color='purple',
            s=60)

ax = plt.gca()

for i, group in enumerate([ctrl_df, psilo_df]):
    fish_counts_list = []
    baseline = [ctrl_baseline, psilo_baseline][i]
    for fish_name in group['fish_name'].unique():
        fish_df = group[group['fish_name'] == fish_name]
        fish_counts = {f"{int(key)}-{int(key + 100)}": round(
            len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + 100)]))
                       for key in range(0, 1800, 100)}
        counts_for_fish = [fish_counts[bin] for bin in time_bins_labels]
        counts_for_fish = [((c / baseline) * 100 - 100) for c in counts_for_fish]
        fish_counts_list.append(counts_for_fish)

    fish_counts_matrix = np.array(fish_counts_list)

    group_mean = np.mean(fish_counts_matrix, axis=0)
    group_stds = np.std(fish_counts_matrix, axis=0)
    n_samples = len(fish_counts_list)
    group_stes = group_stds / np.sqrt(n_samples)

    ax.fill_between(time_bin_centers, group_mean + group_stes, group_mean - group_stes,
                    color=['gray', 'purple'][i], alpha=0.1)

time_bins = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900',
             '900-1000', '1000-1100', '1100-1200', '1200-1300', '1300-1400', '1400-1500', '1500-1600', '1600-1700',
             '1700-1800']

time_bin_positions = [(int(bin.split('-')[0]) + int(bin.split('-')[1])) / 2 for bin in time_bins]

rect = patches.Rectangle((time_bin_positions[9], -100), time_bin_positions[11] - time_bin_positions[9],
                         ax.get_ylim()[1] + 101.5,  # Make the rectangle span the full height of the plot
                         linewidth=0, edgecolor=None, facecolor='lightskyblue', alpha=0.2)

ax.add_patch(rect)
ax.set_xticks(time_bin_positions)
ax.set_xticklabels(time_bins, rotation=45)
plt.axhline(y=0, c='r', alpha=.5, linestyle='--')
ax.tick_params(axis='x', which='minor', length=4, width=1, direction='in', grid_alpha=0.5)
plt.gca().add_patch(rect)
plt.legend(fontsize=18)
plt.xlabel('Time bins (s)', fontsize=19)
plt.xticks(fontsize=16)
import statsmodels

for label in ax.get_xticklabels():
    if label.get_text() in ['1100-1200', '1200-1300']:  # make em bold
        label.set_fontweight('bold')
        label.set_color('black')
# plt.xticks(xticks, [f"**{label}**" if label in ['1100-1200', '1200-1300'] else label for label in time_bins], fontweight='bold')


plt.ylabel('% swim amount difference from baseline', fontsize=22)
plt.title('Swim Events Compared to Baseline', fontsize=15, fontweight='bold')

for time_bin, significance in norm_ast.items():
    if int(time_bin.split('-')[0]) < 900:
        continue
    if significance != 'ns':  # Only plot labels for significance
        x_position = time_bin_positions[time_bins.index(time_bin)]
        y_position = 70
        plt.text(x=x_position, y=y_position, s='*', color='black', fontsize=26, ha='center')

plt.tight_layout()
plt.savefig(join(figures_path, 'swim_event_plot_baseline_sig_multi.pdf'), dpi=800, bbox_inches='tight')
plt.show()

# %%
jump = 100
n_bins = int(np.floor(1800 / jump))
time_bins = np.linspace(0, 1800, n_bins + 1)


# Function to compute fish-level normalized counts
def compute_fish_normalized_counts(df):
    fish_counts = {}

    for fish in df['fish_name'].unique():
        fish_df = df[df['fish_name'] == fish]
        baseline = np.mean([len(fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + jump)])
                            for key in time_bins[:9]])  # Baseline: first 900s

        fish_counts[fish] = {key: (len(
            fish_df[(fish_df['swim_start'] >= key) & (fish_df['swim_start'] < key + jump)]) / baseline) * 100 - 100
                             for key in time_bins[:-1]}

    return fish_counts


# Compute for both groups
ctrl_fish_counts = compute_fish_normalized_counts(ctrl_df)
psilo_fish_counts = compute_fish_normalized_counts(psilo_df)

# Convert to DataFrame
df_list = []
for fish in ctrl_fish_counts:
    for time_bin, value in ctrl_fish_counts[fish].items():
        df_list.append({'fish_name': fish, 'group': 'control', 'time_bin': time_bin, 'percent_change': value})

for fish in psilo_fish_counts:
    for time_bin, value in psilo_fish_counts[fish].items():
        df_list.append({'fish_name': fish, 'group': 'psilocybin', 'time_bin': time_bin, 'percent_change': value})

# Create DataFrame
df_recovery = pd.DataFrame(df_list)

# Keep only recovery period (1200s and above)
df_recovery = df_recovery[(df_recovery['time_bin'] >= 900) & (df_recovery['time_bin'] <= 1200)]

# Convert time_bin to a categorical variable if needed
df_recovery['time_bin'] = df_recovery['time_bin'].astype(int)
ks_2samp(df_recovery[df_recovery['group'] == 'control']['percent_change'],
         df_recovery[df_recovery['group'] == 'psilocybin']['percent_change'])
ttest_ind(df_recovery[df_recovery['group'] == 'control']['percent_change'],
          df_recovery[df_recovery['group'] == 'psilocybin']['percent_change'],
          alternative='less')
mannwhitneyu(df_recovery[df_recovery['group'] == 'control']['percent_change'],
             df_recovery[df_recovery['group'] == 'psilocybin']['percent_change'])

# %%
df_subset = df_recovery[df_recovery['time_bin'] >= 1100]
sns.boxplot(data=df_subset, x='time_bin', hue='group', y='percent_change')
plt.show()
# %%
ctrl_crosses = []
ctrl_ratio = []
ctrl_bef = ctrl_df[ctrl_df['session'] == 'before']
ctrl_aft = ctrl_df[ctrl_df['session'] == 'after']
for fish_name in ctrl_aft['fish_name'].unique():
    fish_df = ctrl_df[ctrl_df['fish_name'] == fish_name]
    fish_baseline = len(fish_df[fish_df['session'] == 'before'])
    fish_aft = fish_df[fish_df['session'] == 'after'].reset_index()
    fish_aft_bouts = len(fish_aft)
    fish_dur_bouts = len(fish_df[fish_df['session'] == 'during'])
    ctrl_ratio.append(fish_aft_bouts / fish_baseline)
    if fish_aft_bouts >= fish_baseline:
        crossing_point = fish_aft['swim_start'][fish_baseline] - 1200
        # print(crossing_point)
    else:
        crossing_point = np.nan
        # crossing_point = 600
    ctrl_crosses.append(crossing_point)
    # print(fish_name, crossing_point)

psilo_crosses = []
psilo_ratio = []
psilo_bef = psilo_df[psilo_df['session'] == 'before']
psilo_aft = psilo_df[psilo_df['session'] == 'after']
for fish_name in psilo_aft['fish_name'].unique():
    fish_df = psilo_df[psilo_df['fish_name'] == fish_name]
    fish_baseline = len(fish_df[fish_df['session'] == 'before'])
    fish_aft = fish_df[fish_df['session'] == 'after'].reset_index()
    fish_aft_bouts = len(fish_aft)
    fish_dur_bouts = len(fish_df[fish_df['session'] == 'during'])
    psilo_ratio.append(fish_aft_bouts / fish_baseline)
    if fish_aft_bouts >= fish_baseline:
        crossing_point = fish_aft['swim_start'][fish_baseline] - 1200
        # print(crossing_point)
    else:
        crossing_point = np.nan
        # crossing_point = 600

    psilo_crosses.append(crossing_point)

    # print(fish_name, crossing_point)
print(mannwhitneyu(ctrl_crosses, psilo_crosses, nan_policy='omit', alternative='greater'))

# %%
from scipy.stats import binomtest


# Function to assign a weighted success to each value (non-NaN)
def success_weight(value):
    if np.isnan(value):
        return 0  # Failure
    else:
        return 1  # Success (you can change this for weighted success)


successes1 = sum(success_weight(v) for v in ctrl_crosses)  # Count of successes in group1
successes2 = sum(success_weight(v) for v in psilo_crosses)  # Count of successes in group2

# Count total attempts (all values excluding NaN)
total1 = len([v for v in ctrl_crosses if not np.isnan(v)])
total2 = len([v for v in psilo_crosses if not np.isnan(v)])

# Perform binomial test comparing group1 and group2
# We perform the binomtest for each group individually against the null hypothesis (e.g., p=0.5)

p_value1 = binomtest(successes1, total1, p=0.5, alternative='two-sided')
p_value2 = binomtest(successes2, total2, p=0.5, alternative='two-sided')

# Output the p-values for both groups
print(f"Binomial test p-value for group 1: {p_value1.pvalue}")
print(f"Binomial test p-value for group 2: {p_value2.pvalue}")

# %%
jump = 100
n_bins = int(np.floor(1800 / jump))
time_bins = np.linspace(0, 1800, n_bins + 1)
ctrl_spont = {f"{int(key)}-{int(key + jump)}": (round(
    len(ctrl_df[(ctrl_df['swim_start'] >= key) & (ctrl_df['swim_start'] < key + jump) & (ctrl_df['stim'] == 0)]) / 10))
              for key in time_bins[:-1]}

ctrl_omr = {f"{int(key)}-{int(key + jump)}": (round(
    len(ctrl_df[(ctrl_df['swim_start'] >= key) & (ctrl_df['swim_start'] < key + jump) & (ctrl_df['stim'] == 1)]) / 10))
            for key in time_bins[:-1]}

psilo_spont = {f"{int(key)}-{int(key + jump)}": (round(len(
    psilo_df[(psilo_df['swim_start'] >= key) & (psilo_df['swim_start'] < key + jump) & (psilo_df['stim'] == 0)]) / 10))
               for key in time_bins[:-1]}

psilo_omr = {f"{int(key)}-{int(key + jump)}": (round(len(
    psilo_df[(psilo_df['swim_start'] >= key) & (psilo_df['swim_start'] < key + jump) & (psilo_df['stim'] == 1)]) / 10))
             for key in time_bins[:-1]}

plt.figure(figsize=(12, 8))
plt.plot(ctrl_spont.values(), color='gray', linewidth=3, linestyle='--', label='Control spont')
plt.scatter(ctrl_spont.keys(), ctrl_spont.values())

plt.plot(ctrl_omr.values(), color='gray', linewidth=3, label='Control OMR')
plt.scatter(ctrl_omr.keys(), ctrl_omr.values())

plt.plot(psilo_spont.values(), color='purple', linewidth=3, linestyle='--', label='Psilocybin spont')
plt.scatter(psilo_spont.keys(), psilo_spont.values())

plt.plot(psilo_omr.values(), color='purple', linewidth=3, label='Psilocybin OMR')
plt.scatter(psilo_omr.keys(), psilo_omr.values())

plt.xticks(rotation=45)
rect = patches.Rectangle((9, 0), 2, max(max(ctrl_counts.values()), max(psilo_counts.values())) * 1.1,
                         linewidth=0, edgecolor=None, facecolor='cyan', alpha=0.3)
plt.gca().add_patch(rect)
plt.legend(fontsize=12)
plt.xlabel('Time bins (s)', fontsize=13)
plt.ylabel('Mean Swim events', fontsize=13)
plt.title('Swim Events Over Time - Task Comparison', fontsize=15, fontweight='bold')
# plt.savefig(r'C:\Users\jonathat.WISMAIN\OneDrive - weizmann.ac.il\Desktop\tail-free/swim_event_plot_task.jpg', dpi=1200, bbox_inches='tight')
plt.show()

# %%
ctrl_bef = ctrl_df[ctrl_df['session'] == 'before']
ctrl_dur = ctrl_df[ctrl_df['session'] == 'during']
ctrl_aft = ctrl_df[ctrl_df['session'] == 'after']
psilo_bef = psilo_df[psilo_df['session'] == 'before']
psilo_dur = psilo_df[psilo_df['session'] == 'during']
psilo_aft = psilo_df[psilo_df['session'] == 'after']


def get_bpf(df):
    total_bouts = len(df)
    n_fish = df.nunique()['fish_name']
    bpm = total_bouts / n_fish
    return bpm


print('before', get_bpf(ctrl_bef), '|', get_bpf(psilo_bef))
print('during', get_bpf(ctrl_dur), '|', get_bpf(psilo_dur))
print('after', get_bpf(ctrl_aft), '|', get_bpf(psilo_aft))


# %%
def compare_params(df1, df2, param, session=None):
    print(param)
    if session:
        print(session)
        df1 = df1[df1['session'] == session]
        df2 = df2[df2['session'] == session]
    print(ks_2samp(df1[param], df2[param]), '\n')


for param in params:
    compare_params(ctrl_df, psilo_df, param, 'after')
# %%
n_bouts_c = [sum(ctrl_df['fish_name'] == fish) for fish in ctrl_df['fish_name'].unique()]
n_bouts_p = [sum(psilo_df['fish_name'] == fish) for fish in psilo_df['fish_name'].unique()]
omr_spont_ratio_c = [len(ctrl_df[(ctrl_df['fish_name'] == fish) & (ctrl_df['stim'] == 1)]) /
                     len(ctrl_df[(ctrl_df['fish_name'] == fish) & (ctrl_df['stim'] == 0)])
                     for fish in ctrl_df['fish_name'].unique()]
omr_spont_ratio_p = [len(psilo_df[(psilo_df['fish_name'] == fish) & (psilo_df['stim'] == 1)]) /
                     len(psilo_df[(psilo_df['fish_name'] == fish) & (psilo_df['stim'] == 0)])
                     for fish in psilo_df['fish_name'].unique()]
ttest_ind(omr_spont_ratio_c, omr_spont_ratio_p)
print('OMR/Spontaneous swimming ratio bigger in control fish, meaning psilocybin fish swim more in spontaneous as well')
# Chi-sq stim/group
contingency_table = pd.crosstab(df_data['group'], df_data['stim'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square: {chi2}, p-value: {p}")
ttest_ind(df_data[df_data['group'] == 'control']['swim_LR_balance'],
          df_data[df_data['group'] == 'psilocybin']['swim_LR_balance'])

# %%

pal = sns.color_palette("tab10")

fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(data=df_data[df_data['group'] == 'psilocybin'], x='swim_duration', y='swim_LR_balance', hue='session',
            palette=pal,
            fill=False, levels=10, thresh=0, linewidths=1)
# sns.kdeplot(data=df_data[df_data['group'] == 'psilocybin'], x='swim_frequency', y='swim_max_signal', hue = 'session', palette = pal,
#             fill=False,  levels=10, thresh=0, linewidths=1)
# sns.kdeplot(data=df_data[df_data['stim'] == 0], x='swim_frequency', y='swim_max_signal', hue = 'group', palette = pal0,
#             fill=False,  levels=10, thresh=0, linewidths=1)
# sns.kdeplot(data=df_data[df_data['stim'] == 1], x='swim_frequency', y='swim_max_signal',  hue = 'group',  palette = pal1,
#             fill=False,  levels=10, thresh=0, linewidths=1)
plt.xlabel("swim_duration")
plt.ylabel("swim_LR_balance")
# ax.set_xlim = [ax.get_xlim()[0], 60]
plt.grid(True)
plt.show()

# %%
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# best separation for 'before' vs 'after'
data = df_data[['swim_duration',
                'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance',
                'swim_frequency', 'stim', 'group', 'session']]
group = 'control'
df_control = data[(data['group'] == group) & (data['session'].isin(['before', 'during']))]

df_control['session_binary'] = df_control['session'].map({'before': 0, 'during': 1})

X = df_control[['swim_duration', 'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance', 'swim_frequency']]
Y = df_control['session_binary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pls = PLSRegression(n_components=2)
pls.fit(X_scaled, Y)
X_loadings = pls.x_weights_
print(f"PLS Loadings for swim parameters ({group})")
for i, col in enumerate(X.columns):
    print(f"{col}: {X_loadings[i]}")

# %%
data = df_data[['swim_duration',
                'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance',
                'swim_frequency', 'stim', 'group', 'session']]
# group = 'control'
# df_control = data[data['group'].isin(['before', 'recovery','post']))]

data['group_binary'] = data['group'].map({'control': 0, 'psilocybin': 1})

X = data[['swim_duration', 'swim_max_signal', 'swim_mean_signal', 'swim_LR_balance', 'swim_frequency']]
Y = data['group_binary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pls = PLSRegression(n_components=2)
pls.fit(X_scaled, Y)

X_loadings = pls.x_weights_
print(f"PLS Loadings for swim parameters ({group})")
for i, col in enumerate(X.columns):
    print(f"{col}: {X_loadings[i]}")

# %%
# omr/spont ratio
for session in sessions:
    print(session)
    control_spont, control_omr = len(ctrl_df[(ctrl_df['stim'] == 0) & (ctrl_df['session'] == session)]), len(
        ctrl_df[(ctrl_df['stim'] == 1) & (ctrl_df['session'] == session)])
    psilo_spont, psilo_omr = len(psilo_df[(psilo_df['stim'] == 0) & (psilo_df['session'] == session)]), len(
        psilo_df[(psilo_df['stim'] == 1) & (psilo_df['session'] == session)])
    print('control:', control_spont, control_omr, control_spont / control_omr)
    print('psilo:', psilo_spont, psilo_omr, psilo_spont / psilo_omr)
    print(mannwhitneyu(control_spont / control_omr, psilo_spont / psilo_omr))

ctrl_spont_tot, ctrl_omr_tot = len(ctrl_df[ctrl_df['stim'] == 0]), len(ctrl_df[ctrl_df['stim'] == 1])
psilo_spont_tot, psilo_omr_tot = len(psilo_df[psilo_df['stim'] == 0]), len(psilo_df[psilo_df['stim'] == 1])

print('total\ncontrol:', ctrl_spont_tot, ctrl_omr_tot, ',ratio = ', ctrl_omr_tot / ctrl_spont_tot,
      '\npsilo:', psilo_spont_tot, psilo_omr_tot, ',ratio = ', psilo_omr_tot / psilo_spont_tot)

# %%
ctrl_group = ctrl_df.groupby(['fish_name', 'stim', 'session']).count()['swim_bout_num']
ctrl_ratio = ctrl_group.unstack('stim').fillna(0)
ctrl_ratio['ratio'] = ctrl_ratio[(0.0)] / ctrl_ratio[(1.0)]
ctrl_ratio[np.isinf(ctrl_ratio)] = np.nan
psilo_group = psilo_df.groupby(['fish_name', 'stim', 'session']).count()['swim_bout_num']
psilo_ratio = psilo_group.unstack('stim').fillna(0)
psilo_ratio['ratio'] = psilo_ratio[(0.0)] / psilo_ratio[(1.0)]

# %% more swimming during OMR
plt.figure(figsize=(12, 8))

sns.boxplot(data=df2, x='group', hue='stim', y='n_bouts', showfliers=False, palette=['maroon', 'black'], alpha=.3)
sns.stripplot(data=df2, x='group', hue='stim', y='n_bouts', palette=['maroon', 'black'], dodge=True, jitter=True,
              size=8, alpha=0.6)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
df4 = df2.groupby(['group', 'stim', 'fish_name'], as_index=False).mean(numeric_only=True)

plt.figure(figsize=(10, 8))
group_means = df4.groupby(['group', 'stim'])['n_bouts'].mean()
group_sems = df4.groupby(['group', 'stim'])['n_bouts'].std()
groups = df4['group'].unique()
stims = df4['stim'].unique()
x_positions = np.arange(len(groups))
colors = {0: 'red', 1: 'black'}
offset = 0.15
for i, group in enumerate(groups):
    for j, stim in enumerate(stims):
        mean = group_means.loc[group, stim]
        sem = group_sems.loc[group, stim]
        plt.errorbar(x=i + (j - 0.5) * offset * 2.5, y=mean, yerr=sem, fmt='o', linewidth=2,
                     color=colors[stim], capsize=0, markersize=12)
sns.stripplot(data=df4, x='group', hue='stim', y='n_bouts', palette=colors,
              dodge=True, jitter=0, size=8, alpha=0.4)
for fish in df4['fish_name'].unique():
    fish_data = df4[df4['fish_name'] == fish]
    if len(fish_data) == 2:
        group = fish_data.iloc[0]['group']
        x_base = np.where(groups == group)[0][0]
        stim_vals = fish_data['stim'].values
        y_vals = fish_data['n_bouts'].values
        x_vals = [x_base + (stim - 0.5) * offset * 2.5 for stim in stim_vals]
        plt.plot(x_vals, y_vals, color='gray', alpha=0.3, zorder=0)
group1_stim0 = df4[(df4['group'] == groups[0]) & (df4['stim'] == 0)]['n_bouts']
group1_stim1 = df4[(df4['group'] == groups[0]) & (df4['stim'] == 1)]['n_bouts']
group2_stim0 = df4[(df4['group'] == groups[1]) & (df4['stim'] == 0)]['n_bouts']
group2_stim1 = df4[(df4['group'] == groups[1]) & (df4['stim'] == 1)]['n_bouts']
p_control = wilcoxon(group1_stim0, group1_stim1, alternative='two-sided').pvalue  # Control (stim 0 vs stim 1)
p_psilo = wilcoxon(group2_stim0, group2_stim1, alternative='two-sided').pvalue  # Psilocybin (stim 0 vs stim 1)
y_max = df4['n_bouts'].max() * 1.1  # Position above the highest point
bar_height = y_max * 0.05
plt.plot([0 - 0.15, 0 + 0.15], [y_max - bar_height, y_max - bar_height], color='black', linewidth=1.5)  # Control
plt.text(0, y_max - bar_height, pval2asterik(p_control), ha='center', fontsize=14)
plt.plot([1 - 0.15, 1 + 0.15], [y_max - bar_height, y_max - bar_height], color='black', linewidth=1.5)  # Psilocybin
plt.text(1, y_max - bar_height, pval2asterik(p_psilo), ha='center', fontsize=14)

p_stim0 = mannwhitneyu(group1_stim0, group2_stim0, alternative='two-sided').pvalue  # Stim 0: Control vs Psilocybin
p_stim1 = mannwhitneyu(group1_stim1, group2_stim1, alternative='two-sided').pvalue  # Stim 1: Control vs Psilocybin
y_max = df4['n_bouts'].max() * 1.1
bar_height = y_max * 0.05

plt.plot([-.15, .8], [y_max + bar_height, y_max + bar_height], color='black', linewidth=1.5)  # Stim 0 comparison
plt.text(0.32, y_max + bar_height * 1.2, pval2asterik(p_stim0), ha='center', fontsize=14)

plt.plot([.2, 1.15], [y_max + bar_height * 2, y_max + bar_height * 2], color='black',
         linewidth=1.5)  # Stim 1 comparison
plt.text(0.675, y_max + bar_height * 2.2, pval2asterik(p_stim1), ha='center', fontsize=14)

p_c_stim1_p_stim0 = mannwhitneyu(group1_stim1, group2_stim0,
                                 alternative='two-sided').pvalue  # Stim 1: Control vs Psilocybin
plt.plot([.2, .8], [y_max - bar_height, y_max - bar_height], color='black', linewidth=1.5)  # Stim 1 comparison
# plt.text(0.5, y_max - bar_height * .65, f'p={pval2asterik(p_c_stim1_p_stim0, return_pval=1)}', ha='center', fontsize=14)
plt.text(0.5, y_max - bar_height * .65, f'p=0.9', ha='center', fontsize=14)

plt.xticks(x_positions, groups, fontsize=20)
plt.ylabel("swimming bouts", fontsize=20)
plt.yticks(fontsize=15)
legend = plt.legend(title="Stimulus", labels=['spontaneous', 'OMR'], fontsize=14, loc='center left',
                    bbox_to_anchor=(1, .5))
legend.get_title().set_fontsize(16)
plt.tight_layout()
sns.despine(top=True)
# plt.savefig(join(figures_path, 'stim_comparison_within.jpg'), dpi=700, bbox_inches='tight')
plt.show()

# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols

df4['stim'] = df4['stim'].astype('category')
model = ols('n_bouts ~ C(group) * C(stim)', data=df4).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

from statsmodels.stats.multicomp import MultiComparison

df4['combo'] = df4['group'].astype(str) + '_stim' + df4['stim'].astype(str)
mc = MultiComparison(df4['n_bouts'], df4['combo'])
tukey_result = mc.tukeyhsd()
print(tukey_result)
# %%
plt.figure(figsize=(10, 8))
df3 = df2[df2['combined_group'].isin(['control_1', 'psilocybin_0'])]
group_means = df3.groupby(['group', 'stim'])['n_bouts'].mean()
group_sems = df3.groupby(['group', 'stim'])['n_bouts'].std()  # Standard error of the mean

# Unique  and conditions for positioning
groups = df3['group'].unique()
stims = df3['stim'].unique()
x_positions = np.arange(len(groups))

# Define color mapping with integer keys
colors = {0: 'red', 1: 'black'}
offset = -.15  # Adjust horizontal offset for error bars to prevent overlap

for i, group in enumerate(groups):
    for j, stim in enumerate(stims):
        try:
            mean = group_means.loc[group, stim]
            sem = group_sems.loc[group, stim]

            # Plot error bars with small offsets for separation
            plt.errorbar(x=i + (j - 0.5) * offset * 2.5, y=mean, yerr=sem, fmt='o', linewidth=2,
                         color=colors[stim], capsize=0, markersize=12)
        except:
            continue
sns.stripplot(data=df3, x='group', hue='stim', y='n_bouts', palette=colors,
              dodge=True, jitter=True, size=8, alpha=0.4)
group1_stim1 = df3[(df3['group'] == groups[0]) & (df3['stim'] == 1)]['n_bouts']
group2_stim0 = df3[(df3['group'] == groups[1]) & (df3['stim'] == 0)]['n_bouts']

# Perform statistical tests within each group
# p_control = wilcoxon(group1_stim0, group1_stim1, alternative='two-sided').pvalue  # Control (stim 0 vs stim 1)
# p_psilo = wilcoxon(group2_stim0, group2_stim1, alternative='two-sided').pvalue  # Psilocybin (stim 0 vs stim 1)

# # Define x positions for asterisks
# y_max = df3['n_bouts'].max() * 1.1  # Position above the highest point
# bar_height = y_max * 0.05  # Adjust height of significance bars

# plt.plot([0 - 0.15, 0 + 0.15], [y_max - bar_height, y_max - bar_height], color='black', linewidth=1.5)  # Control
# plt.text(0, y_max - bar_height, pval2asterik(p_control), ha='center', fontsize=14)

# plt.plot([1 - 0.15, 1 + 0.15], [y_max - bar_height, y_max - bar_height], color='black', linewidth=1.5)  # Psilocybin
# plt.text(1, y_max - bar_height , pval2asterik(p_psilo), ha='center', fontsize=14)
# # from statsmodels.stats.multicomp import pairwise_tukeyhsd

# # # Perform Tukey's HSD test
# # tukey = pairwise_tukeyhsd(endog=df2['n_bouts'],  # Dependent variable
# #                            groups=df2[['group', 'stim']].astype(str).agg('_'.join, axis=1),  # Combine group & session
# #                            alpha=0.05)  # Significance level

# # print(tukey)
# p_stim0 = mannwhitneyu(group1_stim0, group2_stim0, alternative='two-sided').pvalue  # Stim 0: Control vs Psilocybin
# p_stim1 = mannwhitneyu(group1_stim1, group2_stim1, alternative='two-sided').pvalue  # Stim 1: Control vs Psilocybin

# # Define y positions for significance bars
# y_max = df3['n_bouts'].max() * 1.1
# bar_height = y_max * 0.05

# # Plot significance bars for between-group comparisons
# plt.plot([-.15, .8], [y_max + bar_height, y_max + bar_height], color='black', linewidth=1.5)  # Stim 0 comparison
# plt.text(0.32, y_max + bar_height * 1.2, pval2asterik(p_stim0), ha='center', fontsize=14)

# plt.plot([.2, 1.15], [y_max + bar_height * 2, y_max + bar_height * 2], color='black', linewidth=1.5)  # Stim 1 comparison
# plt.text(0.675, y_max + bar_height * 2.2, pval2asterik(p_stim1), ha='center', fontsize=14)

p_c_stim1_p_stim0 = mannwhitneyu(group1_stim1, group2_stim0,
                                 alternative='two-sided').pvalue  # Stim 1: Control vs Psilocybin
plt.plot([.2, .8], [y_max - bar_height, y_max - bar_height], color='black', linewidth=1.5)  # Stim 1 comparison
# plt.text(0.5, y_max - bar_height * .65, f'p={pval2asterik(p_c_stim1_p_stim0, return_pval=1)}', ha='center', fontsize=14)
plt.text(0.5, y_max - bar_height * .65, f'p=0.9', ha='center', fontsize=14)

plt.xticks(x_positions, groups, fontsize=20)
plt.ylabel("swimming bouts", fontsize=20)
plt.yticks(fontsize=15)
legend = plt.legend(title="Stimulus", labels=['spontaneous', 'OMR'], fontsize=14, loc='center left',
                    bbox_to_anchor=(1, .5))
legend.get_title().set_fontsize(16)
plt.tight_layout()
sns.despine(top=True)
plt.savefig(join(figures_path, 'stim_comparison_ctrl1psilo0.jpg'), dpi=700, bbox_inches='tight')
plt.show()

# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols

for session in ['before', 'during', 'after']:
    model = ols('n_bouts ~ C(group) + C(stim) + C(group):C(stim)',
                data=df2[df2['session'] == session]).fit()
    result = sm.stats.anova_lm(model, type=2)
    print(session, result)

# %%
# model = ols('n_bouts ~ C(group) + C(stim) + C(session) + C(group):C(stim) + C(group):C(session)', 
#             data=df2[df2['session']==session]).fit() 
# result = sm.stats.anova_lm(model, type=2) 
# print(result)
# pg.mixed_anova(dv='n_bouts', between='group', within='stim', subject='fish_name', data=df2)

model = ols("""n_bouts ~ C(session) + C(group) + C(stim) +
               C(session):C(group) + C(session):C(stim) + C(group):C(stim) +
               C(session):C(group):C(stim)""", data=df2).fit()

sm.stats.anova_lm(model, typ=2)

# %% Check fish freezing
from scipy import stats

ctrl_rate = 5 / 10
psilo_rate = 0 / 11
table = [[5, 5], [0, 11]]
stats.fisher_exact(table, alternative='two-sided')

# check fish recovery to baseline
ctrl_rate = 6 / 10
psilo_rate = 4 / 11
table = [[4, 10], [7, 11]]
stats.fisher_exact(table, alternative='two-sided')
