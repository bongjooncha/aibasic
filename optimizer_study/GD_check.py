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
lr = 0.12  # 학습률
n_epochs = 1999  # 반복 횟수
m = 100  # 샘플 개수

#초기 theta 값
start = np.random.randn(2, 1)  # 무작위 초기화


# Batch Gradient Descent (BGD) 구현
theta_bgd = start
theta_bgd_path = []

for iteration in range(n_epochs):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - lr * gradients
    theta_bgd_path.append(theta_bgd)
    
theta_bgd_path = np.array(theta_bgd_path)


t0, t1 = 5, 50  
def learning_schedule(t):
    return t0 / (t + t1)

# SGD 구현
theta_sgd = start
theta_sgd_path = []

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        lr = learning_schedule(epoch * m + i)
        theta_sgd = theta_sgd - lr * gradients
    theta_sgd_path.append(theta_sgd)

theta_sgd_path = np.array(theta_sgd_path)

# MGD 구현
batch_size = 10  # 미니 배치 크기

theta_mgd = start 
theta_mgd_path = []

for n_epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, batch_size):
        xi = X_b_shuffled[i:i+batch_size]
        yi = y_shuffled[i:i+batch_size]
        gradients = 2/batch_size * xi.T.dot(xi.dot(theta_mgd) - yi)
        lr = learning_schedule(n_epoch * m + i)
        theta_mgd = theta_mgd - lr * gradients
    theta_mgd_path.append(theta_mgd)

theta_mgd_path = np.array(theta_mgd_path)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(10, 8))

# 함수 animation
def animate(i):
    plt.clf()
    plt.plot(X, y, 'b.')  # 데이터 플롯

    # Batch Gradient Descent (BGD)
    plt.plot(X, X_b.dot(theta_bgd_path[i]), 'r-', label="BGD")

    # Stochastic Gradient Descent (SGD)
    plt.plot(X, X_b.dot(theta_sgd_path[i]), 'g-', label="SGD")

    # Mini-batch Gradient Descent (MGD)
    plt.plot(X, X_b.dot(theta_mgd_path[i]), 'b-', label="MGD")

    plt.title(f"Iteration: {i}")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

ani = FuncAnimation(fig, animate, frames=len(theta_bgd_path), interval=50)
# ani.save('./gradient_descent_animation.gif', writer='imagemagick')



# theta animation
# def animate(i):
#     plt.clf()
#     plt.plot(theta_bgd_path[:i+1, 0], theta_bgd_path[:i+1, 1], 'r-', label="BGD")
#     plt.scatter(theta_bgd_path[i, 0], theta_bgd_path[i, 1], color='red', s=10, marker='o')

#     plt.plot(theta_sgd_path[:i+1, 0], theta_sgd_path[:i+1, 1], 'g-', label="SGD")
#     plt.scatter(theta_sgd_path[i, 0], theta_sgd_path[i, 1], color='green', s=10, marker='o')

#     plt.plot(theta_mgd_path[:i+1, 0], theta_mgd_path[:i+1, 1], 'b-', label="MGD")
#     plt.scatter(theta_mgd_path[i, 0], theta_mgd_path[i, 1], color='blue', s=10, marker='o')


#     plt.title(f"Iteration: {i}")
#     plt.xlabel('Theta_0')
#     plt.ylabel('Theta_1')
#     plt.legend()
#     plt.grid(True)

# ani = FuncAnimation(fig, animate, frames=len(theta_sgd_path), interval=50)
# ani.save('./gradient_descent_theta_animation.gif', writer='imagemagick')

plt.show()

# theta_bgd_2d = np.squeeze(theta_bgd_path, axis=2)
# theta_sgd_2d = np.squeeze(theta_sgd_path, axis=2)
# theta_mgd_2d = np.squeeze(theta_mgd_path, axis=2)

# # theta_bgd_2d, theta_sgd_2d, theta_mgd_2d를 데이터프레임으로 변환
# df_bgd = pd.DataFrame(theta_bgd_2d, columns=['Theta0', 'Theta1'])
# df_sgd = pd.DataFrame(theta_sgd_2d, columns=['Theta0', 'Theta1'])
# df_mgd = pd.DataFrame(theta_mgd_2d, columns=['Theta0', 'Theta1'])

# # 데이터프레임 출력
# print("Batch Gradient Descent:")
# print(df_bgd.head())

# print("\nStochastic Gradient Descent:")
# print(df_sgd.head())

# print("\nMini-batch Gradient Descent:")
# print(df_mgd.head())