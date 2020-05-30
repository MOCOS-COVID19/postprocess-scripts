data = '29/05/2020'
place = 'wroclaw'
run_type = 'bt'
types = {'bt': 'backtracking', 'qt': 'no-backtracking'}
titles = {'dw': 'na Dolnym Śląsku', 'pl': 'w Polsce', 'wroclaw': 'we Wrocławiu'}
sync_dates = {'dw': '2020/04/16', 'pl': '2020/04/01', 'wroclaw': '2020/04/16'}
home_dir = '.'
dir_exp = 'raport-20200529-wroclaw-1k/julia-modulation-b_vs_limit_value-u0-c01.35-q0.3'
plt_title = f'Rozwój epidemii {titles[place]} od {data}'
data123 = ''.join(data.split('/'))
fig_path = f'prognoza_pojedyncze_wiazki_{data123}_{place}_{types[run_type]}.png'

pl_det_path = f'~/postprocess-scripts/data/{place}_detections.csv'

import h5py
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

pl_det = pd.read_csv(pl_det_path)
pl_det['date2'] = pl_det.date.apply(lambda x: '/'.join(x.split('/')[::-1]))

prefix=os.path.join(home_dir, dir_exp)
x_=np.arange(41)
y_=np.arange(21)
DEATHS='daily_deaths'
DETECTIONS='daily_detections'
HOSPITALIZATIONS='daily_hospitalizations'
INFECTIONS='daily_infections'
cum_tol_down=0.25
cum_tol_up=0.15
daily_tol_down=0.2
daily_tol_up=0.2
daily_tol_down2=10
daily_tol_up2=10
if place == 'pl':
    daily_tol_down2 = 50
    daily_tol_up2 = 50

date_sync = sync_dates[place]
daily_dates = pl_det[pl_det.date2>date_sync].date2.values
daily7_cases = pl_det[pl_det.date2>date_sync].daily_average7.values
daily1_cases = pl_det[pl_det.date2>date_sync].daily.values
det_cum = pl_det[pl_det.date2==date_sync].detected.values[0]
#fig, ax = plt.subplots(figsize=(10, 6))
success_dict=defaultdict(int)
dets=[]
trys=[]
infdet=defaultdict(list)
parameters_path = os.path.join(prefix, 'parameters_map.csv')
parameters = pd.read_csv(parameters_path)
for x in x_:
    print(x)
    for y in y_:
        daily_path = os.path.join(prefix, f'grid_{x}_{y}', 'output', 'daily_trajectories.jld2')
        if not os.path.exists(daily_path):
            print(f'[ERR] path {daily_path} not exists')
            continue
        daily = h5py.File(daily_path, "r")
        successes = 0
        for k_i, k in enumerate(daily.keys()):
            if k_i > 100:
                break
            det = daily[k][DETECTIONS][()]
            inf = daily[k][INFECTIONS][()]
            #inf = daily[k][HO][()]
            cum_det = det.cumsum()
            cd1 = np.argmax(cum_det>det_cum*(1-cum_tol_down)) - 1
            cd2 = np.argmax(cum_det>det_cum*(1+cum_tol_up))
            success = False
            for try_id in np.arange(cd1, cd2 + 1):
                if success:
                    continue
                try_success = True
                for i, d7, d1 in zip(np.arange(try_id, try_id + len(daily1_cases)), daily7_cases, daily1_cases):
                    if i >= len(det):
                        try_success = False
                    if not try_success:
                        continue
                    threshold_down = np.maximum(daily_tol_down*d1, np.maximum(daily_tol_down2, np.abs(d1 - d7)))
                    threshold_up = np.maximum(daily_tol_up*d1, np.maximum(daily_tol_up2, np.abs(d1 - d7)))

                    #if det[i] < d * (1 - daily_tol_down):
                    #if det[i] < d - daily_tol_down2:
                    if det[i] < d7 - threshold_down:
                        try_success = False
                        continue
                    #if det[i] > d * (1 + daily_tol_up):
                    #if det[i] > d + daily_tol_up2:
                    if det[i] > d7 + threshold_up:
                        try_success = False
                        continue
                if try_success:
                    success = True
                    #ax.plot(det)
                    dets.append(det)
                    trys.append(try_id)
                    cum_inf = inf.cumsum()
                    days_offset = len(daily_dates)
                    today_id = try_id + days_offset
                    infdet[y].append(cum_inf[today_id]/cum_det[today_id])
            if success:
                successes += 1
        if successes > 0:
            success_dict[(x,y)] = successes
            red = 1-parameters[parameters.path.str.startswith(f'grid_{x}_{y}/')][' limit_value'].values[0]
            b = parameters[parameters.path.str.startswith(f'grid_{x}_{y}/')][' probabblity'].values[0]
            print(f'{x} {y} (b={b}, redukcja o {red*100:.1f}%) - liczba sukcesów: {successes}/{len(daily.keys())}')
            #print(inf.cumsum())

str_=''
str_s=''
for k in infdet.keys():
    m = np.array(infdet[k]).mean()
    s = np.array(infdet[k]).std()
    str_+=f'{m},'
    str_s+=f'{s},'
    print(f'{k/20}\t{m}\t{s}')
print('str')
print(str_)
print(str_s)
print('success_dict')
print(success_dict)

# fig, ax = plt.subplots(figsize=(10, 6))
plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots(figsize=(12, 8))
obs_line = [[] for _ in range(400)]
min_line = np.ones(400) * 9999
max_line = np.zeros(400)
ax.plot(dets[0][trys[0] - np.array(trys).min():], 'b-', linewidth=0.25, label='Trajektoria modelu')
daily_cases7 = pl_det[pl_det.date2 > date_sync].daily_average7.values
daily_cases1 = pl_det[pl_det.date2 > date_sync].daily.values
diffs = []

earlier_days = 0
later_days = 140

for det, try_ in zip(dets, trys):
    threshold_down = []
    threshold_up = []
    for enum_, e in enumerate(det[try_ - np.array(trys).min():try_ - np.array(trys).min() + 290]):
        obs_line[enum_].append(e)
        if min_line[enum_] > e:
            min_line[enum_] = e
        if max_line[enum_] < e:
            max_line[enum_] = e
        if enum_ < len(daily1_cases):
            d1 = daily1_cases[enum_]
            d7 = daily7_cases[enum_]
            diffs.append((np.maximum(daily_tol_down * d1, np.maximum(daily_tol_down2, np.abs(d1 - d7))),
                          np.maximum(daily_tol_up * d1, np.maximum(daily_tol_up2, np.abs(d1 - d7)))))
            threshold_down.append(d7 - np.maximum(daily_tol_down * d1, np.maximum(daily_tol_down2, np.abs(d1 - d7))))
            threshold_up.append(d7 + np.maximum(daily_tol_up * d1, np.maximum(daily_tol_up2, np.abs(d1 - d7))))
    # ax.plot(threshold_down, 'b--', linewidth=1.0)
    # ax.plot(threshold_up, 'g--', linewidth=1.0)
    ax.plot(np.arange(-np.array(trys).min(), -np.array(trys).min() + len(det[try_ - np.array(trys).min():])),
            det[try_ - np.array(trys).min():], 'b-', linewidth=0.10)
    ax.plot(np.arange(-np.array(trys).min() - len(det[:try_ - np.array(trys).min()]), -np.array(trys).min() + 1),
            det[:try_ - np.array(trys).min() + 1], 'b-', linewidth=0.10)

# if try_ - np.array(trys).min() >0:
#    ax.plot(np.arange(np.array(trys).min() - try_, 1), det[:try_ - np.array(trys).min()+1], 'b-', linewidth=0.15)
daily_cases7all = pl_det.daily_average7.values
daily_cases1all = pl_det.daily.values
min_x = len(daily_cases7) - len(daily_cases7all)
ax.plot(np.arange(min_x, min_x + len(daily_cases7all)), daily_cases7all, '--', color='brown', linewidth=1.5,
        label='7-dniowa średnia dziennie zdiagnozowanych na Dolnym Śląsku')
ax.plot(np.arange(min_x, min_x + len(daily_cases7all)), daily_cases7all, 'k.', markersize=2.5)

percentiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
linewidths = [1.0, 1.25, 1.5, 1.75, 2.0, 1.75, 1.5, 1.25, 1.0]
linestyles = ['k--', 'k:', 'k--', 'k:', 'k--', 'k:', 'k--', 'k:', 'k--']
obs_lines = []
for i in range(9):
    obs_lines.append(obs_line[:len(bound_x)].copy())

for enum_, e in enumerate(bound_x):
    for i in range(9):
        # print(len(obs_lines[i]))
        # print(enum_)
        obs_lines[i][enum_] = np.quantile(np.array(obs_lines[i][enum_]), percentiles[i])

for i in range(9):
    if i == 0:
        label = 'Kwantyle trajektorii. Od dołu: 2.5%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 97.5%'

        ax.plot(np.arange(-np.array(trys).min() - min_x + np.array(trys).min() - 3,
                          -np.array(trys).min() - min_x + np.array(trys).min() - 3 + len(
                              obs_lines[i][-min_x + np.array(trys).min() - 3:])),
                obs_lines[i][-min_x + np.array(trys).min() - 3:],
                linestyles[i],
                linewidth=linewidths[i], label=label)
    else:
        ax.plot(np.arange(-np.array(trys).min() - min_x + np.array(trys).min() - 3,
                          -np.array(trys).min() - min_x + np.array(trys).min() - 3 + len(
                              obs_lines[i][-min_x + np.array(trys).min() - 3:])),
                obs_lines[i][-min_x + np.array(trys).min() - 3:],
                linestyles[i],
                linewidth=linewidths[i])
min_line_7days = pd.Series(min_line).rolling(7).mean().values
trys_min = np.array(trys).min()
ax.plot(np.arange(-trys_min, -trys_min + len(min_line_7days)), min_line_7days, 'r--',
        label='Minimum i maksimum trajektorii (7-dniowa średnia krocząca)')
max_line_7days = pd.Series(max_line).rolling(7).mean().values
ax.plot(np.arange(-trys_min, -trys_min + len(min_line_7days)), max_line_7days, 'r--')
# ax.legend()
import datetime as dt
from dateutil import parser
import matplotlib.dates as mdates

begin_date = '16-04-2020'  # date_last#'22-05-2020'
now = parser.parse(begin_date, dayfirst=True)

earlier = now - dt.timedelta(days=earlier_days)  # parser.parse(begin_date)
later = now + dt.timedelta(days=later_days + 1)
days = mdates.drange(earlier, later, dt.timedelta(days=7))
t = [dt.datetime.fromordinal(int(day)).strftime('%d/%m/%y') for day in days]
ax.set_xticks(np.arange(-earlier_days, later_days + 1, 7))
ax.set_xticklabels([t[i] for i, v in enumerate(range(len(t)))], rotation=90)
ax.set_xlim([-earlier_days, later_days])
ax.set_ylim([0, 100])
ax.set_xlabel('Data', fontsize=18)
ax.set_ylabel('Liczba dziennie zdiagnozowanych przypadków', fontsize=18)
ax.grid(which='both')
ax.legend(fontsize=14, loc='upper left')
plt.tight_layout()
plt.title(plt_title, fontsize=18)

plt.savefig(fig_path, dpi=300)

prefix=os.path.join(home_dir, dir_exp)
x_=np.arange(41)
y_=np.arange(21)
DEATHS='daily_deaths'
DETECTIONS='daily_detections'
HOSPITALIZATIONS='daily_hospitalizations'
INFECTIONS='daily_infections'
det_cum = pl_det[pl_det.date2==date_sync].detected.values[0]
#fig, ax = plt.subplots(figsize=(10, 6))
success_dict=defaultdict(int)
dets=[]
trys=[]
subcritical=defaultdict(int)
subcritical2=defaultdict(int)
for x in x_:
    for y in y_:
        daily_path = os.path.join(prefix, f'grid_{x}_{y}', 'output', 'daily_trajectories.jld2')
        if not os.path.exists(daily_path):
            print(f'[ERR] path {daily_path} not exists')
            continue
        daily = h5py.File(daily_path, "r")
        successes = 0
        sub_key = f'grid_{x}_{y}'
        for k in daily.keys():
            inf = daily[k][INFECTIONS][()]
            if len(inf)<=400:
                if sum(inf)<100000:
                    subcritical[sub_key] += 1
                continue
            if inf[400] < 50:
                subcritical[sub_key] += 1
        if subcritical[sub_key] == 1000:
            subcritical2[y] = x+1
print(subcritical2)

