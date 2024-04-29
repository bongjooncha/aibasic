import numpy as np
import matplotlib.pyplot as plt

# 가상의 데이터 생성
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 편향을 위해 X에 1 추가
X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
lr = 0.1  # 학습률
n_iterations = 100  # 반복 횟수
m = 100  # 샘플 개수
batch_size = 10  # 미니 배치 크기

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
        theta_mgd = theta_mgd - lr * gradients
        theta_mgd_path.append(theta_mgd)

theta_mgd_path = np.array(theta_mgd_path)