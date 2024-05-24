import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import pandas as pd

# 공통 값
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# 설정 값
eta = 0.1  # 학습률
n_iterations = 300  # 반복 횟수
m = 100  # 샘플 개수

#초기 theta 값
start = np.random.randn(2, 1)  # 무작위 초기화


# Batch Gradient Descent (BGD) 구현
theta_bgd = start.copy()
theta_bgd_path = []

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - eta * gradients
    theta_bgd_path.append(theta_bgd)
    
theta_bgd_path = np.array(theta_bgd_path)


# agagrad 구현
theta_agagrad = start.copy()  # 세타값을 무작위로 생성
theta_agagrad_path = []  

    # agagrad 변수 초기화
epsilon = 1e-8  # 분모가 0이 되는 것을 방지하기 위한 작은 값
G = np.zeros((2, 1))  # 그라디언트의 제곱합을 저장

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_agagrad) - y)
    G += gradients**2
    adjusted_eta = eta / (np.sqrt(G) + epsilon)
    theta_agagrad -= adjusted_eta * gradients
    theta_agagrad_path.append(theta_agagrad.copy())

theta_agagrad_path = np.array(theta_agagrad_path)


# RMSProp 구현
theta_rmsprop = start.copy()  # 세타값을 무작위로 생성
theta_rmsprop_path = []    

    # RMSProp 변수 초기화
gamma = 0.9
epsilon = 1e-8  # 분모가 0이 되는 것을 방지하기 위한 작은 값
Eg2 = np.zeros((2, 1))  # 그라디언트의 제곱에 대한 지수 가중 이동 평균

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_rmsprop) - y)
    Eg2 = gamma * Eg2 + (1 - gamma) * gradients**2
    adjusted_eta = eta / (np.sqrt(Eg2) + epsilon)
    theta_rmsprop -= adjusted_eta * gradients
    theta_rmsprop_path.append(theta_rmsprop.copy())

theta_rmsprop_path = np.array(theta_rmsprop_path)


# AdaDelta 구현
theta_adadelta = start.copy()  # 세타값을 무작위로 생성
theta_adadelta_path = []   

    # AdaDelta 변수 초기화
gamma = 0.9
epsilon = 1e-8  # 분모가 0이 되는 것을 방지하기 위한 작은 값
Eg2 = np.zeros((2, 1))  # 그라디언트의 제곱에 대한 지수 가중 이동 평균
Edelta_theta2 = np.zeros((2, 1))  # 업데이트의 제곱에 대한 지수 가중 이동 평균

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_adadelta) - y)
    Eg2 = gamma * Eg2 + (1 - gamma) * gradients**2
    delta_theta = - (np.sqrt(Edelta_theta2 + epsilon) / np.sqrt(Eg2 + epsilon)) * gradients
    theta_adadelta += delta_theta
    Edelta_theta2 = gamma * Edelta_theta2 + (1 - gamma) * delta_theta**2
    theta_adadelta_path.append(theta_adadelta.copy())

theta_adadelta_path = np.array(theta_adadelta_path)

# Adam 구현
theta_adam = start.copy()
theta_m = np.zeros_like(theta_adam)  # 그라디언트의 지수 가중 평균
theta_v = np.zeros_like(theta_adam)  # 그라디언트 제곱의 지수 가중 평균
theta_adam_path = []  

    # Adam 변수 초기화
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8  # 분모가 0이 되는 것을 방지하기 위한 작은 값

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_adam) - y)
    theta_m = beta1 * theta_m + (1 - beta1) * gradients
    theta_v = beta2 * theta_v + (1 - beta2) * (gradients ** 2)
    m_hat = theta_m / (1 - beta1**(iteration + 1))
    v_hat = theta_v / (1 - beta2**(iteration + 1))
    eta_t = eta / (np.sqrt(v_hat) + epsilon)
    theta_adam -= eta_t * m_hat
    theta_adam_path.append(theta_adam.copy())

theta_adam_path = np.array(theta_adam_path)


# 애니메이션 설정
fig, ax = plt.subplots(figsize=(10, 8))

# 함수 animation
def animate(i):/
    plt.clf()
    plt.plot(X, y, 'k.')  # 데이터 플롯

    # Batch Gradient Descent (BGD)
    plt.plot(X, X_b.dot(theta_bgd_path[i]), 'r-', label="BGD")

    # agagrad
    plt.plot(X, X_b.dot(theta_agagrad_path[i]), 'g-', label="adag")

    # RMSProp
    plt.plot(X, X_b.dot(theta_rmsprop_path[i]),  color='purple', linestyle='-', label="rmsp")

    # AdaDelta
    plt.plot(X, X_b.dot(theta_adadelta_path[i]), color='brown', label="adad")

    # Adam
    plt.plot(X, X_b.dot(theta_adam_path[i]), 'b-', label="adam")


    plt.title(f"Iteration: {i}")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

ani = FuncAnimation(fig, animate, frames=len(theta_bgd_path), interval=50)
ani.save('./gradient_descent_animation.gif', writer='Pillow')



# # theta animation
# def animate(i):
#     plt.clf()
#     plt.plot(theta_bgd_path[:i+1, 0], theta_bgd_path[:i+1, 1], 'r-', label="BGD")
#     plt.scatter(theta_bgd_path[i, 0], theta_bgd_path[i, 1], color='red', s=10, marker='o')

#     plt.plot(theta_agagrad_path[:i+1, 0], theta_agagrad_path[:i+1, 1], 'g-', label="adag")
#     plt.scatter(theta_agagrad_path[i, 0], theta_agagrad_path[i, 1], color='green', s=20, marker='o')

#     plt.plot(theta_rmsprop_path[:i+1, 0], theta_rmsprop_path[:i+1, 1], color='purple', label="rmsp")
#     plt.scatter(theta_rmsprop_path[i, 0], theta_rmsprop_path[i, 1], color='purple', s=10, marker='o')
    
#     plt.plot(theta_adadelta_path[:i+1, 0], theta_adadelta_path[:i+1, 1], color='brown', label="adad")
#     plt.scatter(theta_adadelta_path[i, 0], theta_adadelta_path[i, 1], color='brown', s=10, marker='o')

#     plt.plot(theta_adam_path[:i+1, 0], theta_adam_path[:i+1, 1], 'b-', label="adam")
#     plt.scatter(theta_adam_path[i, 0], theta_adam_path[i, 1], color='blue', s=10, marker='o')


#     plt.title(f"Iteration: {i}")
#     plt.xlabel('Theta_0')
#     plt.ylabel('Theta_1')
#     plt.legend()
#     plt.grid(True)

# ani = FuncAnimation(fig, animate, frames=len(theta_bgd_path), interval=50)
# ani.save('./ada_theta_animation.gif', writer='Pillow')

plt.show()
