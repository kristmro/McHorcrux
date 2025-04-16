import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


#which_test = 'four_corner'
which_test = 'loop'
seed, M, ctrl_pen, act, test_act = 3, 20, 3, 'off', 'off'





# Load the test results
print("Loading test results...")
with open('data/testing_results/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}.pkl'.format(act,which_test,test_act,ctrl_pen,seed,M), 'rb') as file:
    results = pickle.load(file)

# Create figures directory if it doesn't exist
os.makedirs('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}'.format(act,which_test,test_act,ctrl_pen, seed, M), exist_ok=True)

# Check what keys are actually available in the results
print("Available methods:", [key for key in results.keys() if key not in ['w', 'gains']])

# Define methods based on what's available
available_methods = [key for key in results.keys() if key not in ['w', 'gains']]
if not available_methods:
    print("Error: No method results found in the data file!")
    exit(1)

# Define styling based on available methods
if 'ours_meta' in available_methods:
    if len(available_methods) > 1:
        methods = available_methods
        labels = ['Meta-learned' if m == 'ours_meta' else m.replace('_', ' ').title() for m in methods]
    else:
        methods = ['ours_meta']
        labels = ['Meta-learned']
else:
    methods = available_methods
    labels = [m.replace('_', ' ').title() for m in methods]

# Generate enough distinct colors
colors = ['b', 'r', 'g', 'm', 'c', 'y'] + ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(max(0, len(methods)-6))]
coord_labels = ['x', 'y', 'φ']

print(f"Plotting data for methods: {methods}")

# Get time data
t = results[methods[0]]['t']

# Figure 1: Position tracking
fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig1.suptitle('Position Tracking Performance', fontsize=16)

for i in range(3):
    ax = axes1[i]
    # Plot reference
    ax.plot(t, results[methods[0]]['r'][:, i], 'k--', label='Reference')
    
    # Plot each method
    for j, method in enumerate(methods):
        ax.plot(t, results[method]['q'][:, i], colors[j], label=labels[j])
    
    ax.set_ylabel(f'{coord_labels[i]} position')
    ax.grid(True)
    if i == 0:
        ax.legend()

axes1[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/position_tracking.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

# Continue with the rest of your plotting code using the dynamically determined methods
# Figure 2: 2D trajectory plot
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(results[methods[0]]['r'][:, 0], results[methods[0]]['r'][:, 1], 'k--', label='Reference')

for j, method in enumerate(methods):
    ax2.plot(results[method]['q'][:, 0], results[method]['q'][:, 1], colors[j], label=labels[j])

ax2.set_xlabel('x position (m)')
ax2.set_ylabel('y position (m)')
ax2.set_title('2D Trajectory')
ax2.grid(True)
ax2.legend()
ax2.set_aspect('equal')
plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/trajectory_2d.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

# Rest of your plotting code with the dynamic methods list...
# Figure 3: Tracking errors
fig3, axes3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig3.suptitle('Tracking Errors', fontsize=16)

for i in range(3):
    ax = axes3[i]
    
    for j, method in enumerate(methods):
        # Extract position error (first 3 components of the error vector)
        ax.plot(t, results[method]['e'][:, i], colors[j], label=labels[j])
    
    ax.set_ylabel(f'{coord_labels[i]} error')
    ax.grid(True)
    if i == 0:
        ax.legend()

axes3[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/tracking_errors.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

# Figure 4: Control efforts
fig4, axes4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig4.suptitle('Control Efforts cmd', fontsize=16)

for i in range(3):
    ax = axes4[i]
    
    for j, method in enumerate(methods):
        ax.plot(t, results[method]['τ'][:, i], colors[j], label=labels[j])
    
    ax.set_ylabel(f'τ_{coord_labels[i]}')
    ax.grid(True)
    if i == 0:
        ax.legend()

axes4[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/control_efforts_cmd.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

# Figure 4: Control efforts
fig4, axes4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig4.suptitle('Control Efforts body after saturation', fontsize=16)

for i in range(3):
    ax = axes4[i]
    
    for j, method in enumerate(methods):
        ax.plot(t, results[method]['u'][:, i], colors[j], label=labels[j])
    
    ax.set_ylabel(f'u_{coord_labels[i]}')
    ax.grid(True)
    if i == 0:
        ax.legend()

axes4[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/control_efforts_u_after.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

# Figure 5: RMS error comparison
if len(methods) > 1:  # Only make comparison if we have multiple methods
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    index = np.arange(3)

    for j, method in enumerate(methods):
        rms_errors = [np.sqrt(np.mean(results[method]['e'][:, i]**2)) for i in range(3)]
        ax5.bar(index + j*bar_width, rms_errors, bar_width, label=labels[j], color=colors[j])

    ax5.set_xlabel('Coordinate')
    ax5.set_ylabel('RMS Error')
    ax5.set_title('RMS Tracking Error Comparison')
    ax5.set_xticks(index + bar_width/2)
    ax5.set_xticklabels(coord_labels)
    ax5.legend()
    plt.tight_layout()
    plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/rms_error_comparison.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

print("Plots saved")
plt.show()