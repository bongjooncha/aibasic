import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 가상의 데이터 생성
np.random.seed(42)    

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1                  
n_iterations = 1000        
m = 100                   

# Batch Gradient Descent (BGD) 구현
theta_bgd = np.random.randn(2, 1)  
theta_bgd_path = []                

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - eta * gradients
    theta_bgd_path.append(theta_bgd)

theta_bgd_path = np.array(theta_bgd_path)

print(gradients)
print(theta_bgd)

def animate(i):
    plt.clf()
    plt.plot(X, y, 'b.')  # 데이터 플롯
    plt.plot(X, X_b.dot(theta_bgd_path[i]), 'r-')  # 선형 회귀 선 플롯
    theta0 = float(theta_bgd_path[i, 0])
    theta1 = float(theta_bgd_path[i, 1])
    plt.title(f"Iteration: {i}, theta0: {theta0:.2f}, theta1: {theta1:.2f}")
    plt.xlabel('X')
    plt.ylabel('y')

# 애니메이션 설정
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=len(theta_bgd_path), interval=50)

plt.show()
