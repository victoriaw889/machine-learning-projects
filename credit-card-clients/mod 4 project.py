# ----------------------------------------------------------------------------------
# 1: Supervised Learning – Predicting Bill Amounts

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read Excel Sheet Data
creditcarddata = pd.read_excel(r"C:\Users\vicky\AppData\Local\Python 3.12\default of credit card clients.xls")

# Print the column names of the dataset (for debugging)
print(creditcarddata.columns)

# Drop rows with missing values in specified columns
creditcarddata = creditcarddata.dropna(subset=['X5', 'X1', 'X12'])

# X5 is age, X1 is limit balance
X = creditcarddata[['X5', 'X1']]

# X12 is bill amount
y = creditcarddata['X12']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Compare actual vs predicted fares
comparison = pd.DataFrame({'Actual Bill Amount': y_test, 'Predicted Bill Amount': y_pred})
print(comparison.head())

# ----------------------------------------------------------------------------------
# 2: Unsupervised Learning – Clustering Customers

from sklearn.cluster import KMeans

# Load the dataset
credit_data = pd.read_excel(r"C:\Users\vicky\AppData\Local\Python 3.12\default of credit card clients.xls")

# Cluster customers based on credit card usage data
x = credit_data[['X1', 'X6']]

# Initialize KMeans with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(x)

# Add the cluster labels to the dataset
credit_data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(credit_data['X1'], credit_data['X6'], c=credit_data['Cluster'], cmap='viridis')
plt.xlabel('Credit Limit')
plt.ylabel('September 2005 Payment Records')
plt.title('Customer Clusters Based on Credit Card Payment Data')
plt.show()

# ----------------------------------------------------------------------------------
# 3: Reinforcement Learning – Decision-Making with a Simple Environment

import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 10,000 timesteps
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model and test it
model = PPO.load("ppo_cartpole")

# Reset the environment and extract the observation
obs, info = env.reset()

# Initialize done variable
done = False

# Loop to simulate the environment
for _ in range(1000):
    # Predict action based on the observation
    action, _states = model.predict(obs)
    
    # Take the action in the environment
    step_result = env.step(action)
    
    # Unpack the results based on their length
    if len(step_result) == 4:
        obs, reward, done, info = step_result
    elif len(step_result) == 3:
        obs, reward, done = step_result
        info = {}  # Or handle the absence of info appropriately
    
    # Render the environment
    env.render()
    
    # If the episode is done, reset the environment
    if done:
        obs, info = env.reset()
        done = False  # Reset done flag after resetting the environment

# Close the environment rendering
env.close()