import os
print("TEST START")
for d in ['results_test', 'plots_test', 'simulations_test']:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f"Created {d}")
print("TEST END")
