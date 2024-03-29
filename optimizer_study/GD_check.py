import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 시드 작성
np.random.seed(42)
# 가상의 데이터 생성
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 편향을 위해 X에 1 추가
X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1  # 학습률
n_iterations = 1000  # 반복 횟수
m = 100  # 샘플 개수
batch_size = 10  # 미니 배치 크기

# Batch Gradient Descent (BGD) 구현
theta_bgd = np.random.randn(2, 1)  # 무작위 초기화
theta_bgd_path = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - eta * gradients
    theta_bgd_path.append(theta_bgd)

theta_bgd_path = np.array(theta_bgd_path)

# SGD 구현
theta_sgd = np.random.randn(2, 1)  # 무작위 초기화
theta_sgd_path = []

for epoch in range(n_iterations):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        eta = 0.1 / (epoch * m + i + 1)
        theta_sgd = theta_sgd - eta * gradients
        theta_sgd_path.append(theta_sgd)

theta_sgd_path = np.array(theta_sgd_path)

# MGD 구현
theta_mgd = np.random.randn(2, 1)  # 무작위 초기화
theta_mgd_path = []

for iteration in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, batch_size):
        xi = X_b_shuffled[i:i+batch_size]
        yi = y_shuffled[i:i+batch_size]
        gradients = 2/batch_size * xi.T.dot(xi.dot(theta_mgd) - yi)
        eta = 0.1
        theta_mgd = theta_mgd - eta * gradients
        theta_mgd_path.append(theta_mgd)

theta_mgd_path = np.array(theta_mgd_path)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(10, 8))

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

plt.show()
