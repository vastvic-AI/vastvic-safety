import numpy as np
import matplotlib.pyplot as plt

# Create a blank white image (100x100 pixels)
img = np.ones((100, 100, 3))

# Draw walls (black lines)
img[0:100, 0:2] = 0  # Left wall
img[0:100, 98:100] = 0  # Right wall
img[0:2, 0:100] = 0  # Top wall
img[98:100, 0:100] = 0  # Bottom wall

# Draw some internal walls (gray)
img[30:70, 40:42] = 0.5  # Vertical wall 1
img[30:70, 58:60] = 0.5  # Vertical wall 2
img[40:42, 30:70] = 0.5  # Horizontal wall 1
img[58:60, 30:70] = 0.5  # Horizontal wall 2

# Draw exits (green)
img[48:52, 0:5] = [0, 1, 0]  # Left exit
img[48:52, 95:100] = [0, 1, 0]  # Right exit
img[0:5, 48:52] = [0, 1, 0]  # Top exit
img[95:100, 48:52] = [0, 1, 0]  # Bottom exit

# Save the image
import os
os.makedirs('evacuation_simulation/assets', exist_ok=True)
plt.imsave('evacuation_simulation/assets/floor_plan.png', img)
print("Floor plan image saved to 'evacuation_simulation/assets/floor_plan.png'")
