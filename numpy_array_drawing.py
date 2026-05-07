import numpy as np
import matplotlib.pyplot as plt

array = np.zeros((500, 500, 3), dtype=np.uint8)
array[250:300, 250:300] = [255, 0, 0]
plt.imshow(array)
plt.show()