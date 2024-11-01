import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import numpy as np

fig, ax = plt.subplots()

# Read the data, filter values by specie and sort by sepal length
df = pd.read_csv('IRIS.csv', sep=',')

iris_setosa_data = df.loc[df['species'] == 'Iris-setosa'].sort_values('sepal_length')
iris_versicolor_data = df.loc[df['species'] == 'Iris-versicolor'].sort_values('sepal_length')
iris_virginica_data = df.loc[df['species'] == 'Iris-virginica'].sort_values('sepal_length')

# Set the plot structure
iris_setosa_a, iris_setosa_b = iris_setosa_data['sepal_length'], iris_setosa_data['sepal_width']
iris_setosa_scatter = ax.scatter(iris_setosa_a.values[0], iris_setosa_b.values[0], label='Iris Setosa', color='blue')

iris_versicolor_a, iris_versicolor_b = iris_versicolor_data['sepal_length'], iris_versicolor_data['sepal_width']
iris_versicolor_scatter = ax.scatter(iris_versicolor_a.values[0], iris_versicolor_b.values[0], label='Iris Versicolor', color='yellow')

iris_virginica_a, iris_virginica_b = iris_virginica_data['sepal_length'], iris_virginica_data['sepal_width']
iris_virginica_scatter = ax.scatter(iris_virginica_a.values[0], iris_virginica_b.values[0], label='Iris Virginica', color='green')

ax.set_title("Iris Species Data - Sepal Lengths x Widths")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set(xlim=[4, 8.5], ylim=[1.5, 5])
ax.legend()


# Update the plot
def update(frame):
    # Update the iris setosa plot:
    data = np.stack([iris_setosa_a[:frame], iris_setosa_b[:frame]]).T
    iris_setosa_scatter.set_offsets(data)

    # Update the iris versicolor plot:
    data = np.stack([iris_versicolor_a[:frame], iris_versicolor_b[:frame]]).T
    iris_versicolor_scatter.set_offsets(data)

    # Update the iris virginica plot:
    data = np.stack([iris_virginica_a[:frame], iris_virginica_b[:frame]]).T
    iris_virginica_scatter.set_offsets(data)

    return iris_setosa_scatter, iris_versicolor_scatter, iris_virginica_scatter


ani = animation.FuncAnimation(fig=fig, func=update, frames=80, interval=50)
plt.show()
