from runner import run
from config import build_simulation_elements
import numpy as np
import matplotlib.pyplot as plt
from setup.chengdu_route_3_data.dataloader import DataLoader
from scipy.stats import norm


blueprint, agent, run_config, record_config = build_simulation_elements()
name_metric, route_trip_times = run(
    blueprint, agent, run_config, record_config)

simulate_trip_times = route_trip_times['3']
real_trip_times = DataLoader().trip_times

simulate_trip_times = [x/60 for x in simulate_trip_times]
real_trip_times = [x/60 for x in real_trip_times]

params_simulated = norm.fit(simulate_trip_times)
params_real = norm.fit(real_trip_times)

fitted_simulated = norm.rvs(*params_simulated, size=len(simulate_trip_times))
fitted_real = norm.rvs(*params_real, size=len(real_trip_times))

# Convert lists to numpy arrays if they aren't already
simulate_trip_times = np.array(simulate_trip_times)
real_trip_times = np.array(real_trip_times)

fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(min(simulate_trip_times.min(), real_trip_times.min()),
                   max(simulate_trip_times.max(), real_trip_times.max()), 16)

# Plot histograms side by side
ax.hist([simulate_trip_times, real_trip_times], bins=bins,
        label=['Simulated', 'Real'], density=True,
        color=['#171717', '#DA0037'], alpha=0.8,
        edgecolor='black', linewidth=1, rwidth=0.9)

xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 60)

# Plot fitted curves
pdf_simulated = norm.pdf(x, *params_simulated)
pdf_real = norm.pdf(x, *params_real)
ax.plot(x, pdf_simulated, '--', label='Fitted simulated times',
        color='#171717', linewidth=2.5)
ax.plot(x, pdf_real, '-', label='Fitted real times',
        color='#DA0037', linewidth=2.5)

ax.set_xlabel('Trip Times (min)', fontsize=14, labelpad=12)
ax.set_ylabel('Density', fontsize=14, labelpad=12)
ax.legend(fontsize=12, loc='upper right')
ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.patch.set_facecolor('white')
ax.set_facecolor('#EDEDED')

plt.tight_layout()

# fig.savefig('/Users/samuel/research/bunching_strategies_evaluation/realistic_bunching_draft/figs_in_paper/calibration.png', dpi=300)

plt.show()
