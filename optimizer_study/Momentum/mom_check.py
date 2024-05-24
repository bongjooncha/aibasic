import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# 공통 값
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# 설정 값
lr = 0.01  # 학습률
n_iterations = 300  # 반복 횟수
m = 100  # 샘플 개수

#초기 theta 값
start = np.random.randn(2, 1)  # 무작위 초기화


# Batch Gradient Descent (BGD) 구현
theta_bgd = start.copy()
theta_bgd_path = []

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - lr * gradients
    theta_bgd_path.append(theta_bgd)
    
theta_bgd_path = np.array(theta_bgd_path)


# momentom 구현
theta_mom = start.copy()  # 세타값을 무작위로 생성
theta_mom_path = []  

#모멘텀 구현
a = 0.1                     # 모멘텀 상수
mt = np.zeros((2, 1))   # 모멘텀 값을 초기화

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_mom) - y)
    mt = a * mt - lr * gradients
    theta_mom = theta_mom + mt
    theta_mom_path.append(theta_mom.copy())

theta_mom_path = np.array(theta_mom_path)


# nag 구현
theta_nag = start.copy()  # 세타값을 무작위로 생성
theta_nag_path = []    

a = 0.1
mt = np.zeros((2, 1))  # 모멘텀 값을 초기화

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    lookahead_theta = theta_nag + a * mt  # 현재 모멘텀을 반영한 미래의 위치 예측
    gradients = 2/m * X_b.T.dot(X_b.dot(lookahead_theta) - y)
    mt = a * mt - lr * gradients
    theta_nag = theta_nag + mt
    theta_nag_path.append(theta_nag.copy())

theta_nag_path = np.array(theta_nag_path)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(10, 8))

# # 함수 animation
def animate(i):
    plt.clf()
    plt.plot(X, y, 'b.')  # 데이터 플롯

    # Batch Gradient Descent (BGD)
    plt.plot(X, X_b.dot(theta_bgd_path[i]), 'r-', label="BGD")

    # Stochastic Gradient Descent (SGD)
    plt.plot(X, X_b.dot(theta_mom_path[i]), 'g-', label="SGD")

    # Mini-batch Gradient Descent (nag)
    plt.plot(X, X_b.dot(theta_nag_path[i]), 'b-', label="nag")

    plt.title(f"Iteration: {i}")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

ani = FuncAnimation(fig, animate, frames=len(theta_bgd_path), interval=50)
ani.save('./gradient_descent_animation.gif', writer='Pillow')



# theta animation
# def animate(i):
#     plt.clf()
#     plt.plot(theta_bgd_path[:i+1, 0], theta_bgd_path[:i+1, 1], 'r-', label="BGD")
#     plt.scatter(theta_bgd_path[i, 0], theta_bgd_path[i, 1], color='red', s=10, marker='o')

#     plt.plot(theta_mom_path[:i+1, 0], theta_mom_path[:i+1, 1], 'g-', label="MOM")
#     plt.scatter(theta_mom_path[i, 0], theta_mom_path[i, 1], color='green', s=20, marker='o')

#     plt.plot(theta_nag_path[:i+1, 0], theta_nag_path[:i+1, 1], 'b-', label="NAG")
#     plt.scatter(theta_nag_path[i, 0], theta_nag_path[i, 1], color='blue', s=10, marker='o')


#     plt.title(f"Iteration: {i}")
#     plt.xlabel('Theta_0')
#     plt.ylabel('Theta_1')
#     plt.legend()
#     plt.grid(True)

# ani = FuncAnimation(fig, animate, frames=len(theta_bgd_path), interval=50)
# ani.save('./momentom_theta_animation.gif', writer='Pillow')

plt.show()
