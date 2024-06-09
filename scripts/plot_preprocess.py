import pandas as pd
import matplotlib.pyplot as plt

# Let's read the data from the files again
python_data = pd.read_csv('Python_ff3751e9-a2c0-4902-9d8d-43dd5f7b26a6.txt')
r_data = pd.read_csv('R_ff3751e9-a2c0-4902-9d8d-43dd5f7b26a6.txt')

# Now, let's plot the data
plt.figure(figsize=(10, 5))

# Python data
plt.plot(python_data['mass'], python_data['int'], label='Python Data', linestyle='-', alpha=0.6)

# R data
plt.plot(r_data['mass'], r_data['int'], label='R Data', linestyle='-', alpha=0.6)

plt.title('Comparison of Python and R Processed Data')
plt.xlabel('Mass')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)
plt.savefig('R_Python_compare.png')
plt.show()
