import matplotlib.pyplot as plt
import numpy as np
init_lr = 0.001
decay_rate = 0.7
decay_step = 200000
total_step = 201 * 3000
steps = np.arange(total_step)
lr = init_lr * (decay_rate ** (np.round(steps / decay_step)))
plt.plot(steps, lr, 'b')
plt.show()


