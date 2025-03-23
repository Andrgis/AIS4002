import pandas as pd
import matplotlib.pyplot as plt

tmp_path = "./tmp/sb3_log_main/"  # Path to save logs
adder_path = "./tmp/adder/"  # Path to save logs

log_data = pd.read_csv(tmp_path + "monitor.csv")
adder_data = pd.read_csv(adder_path + "monitor.csv")
adder_data['t'] += log_data['t'].iloc[-1]  # Adjust the timesteps
df = pd.concat([log_data, adder_data], ignore_index=True)
print(df)
plt.plot(df["t"], df['r'])
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.title('Training Rewards Over Time')
plt.grid()
plt.show()
