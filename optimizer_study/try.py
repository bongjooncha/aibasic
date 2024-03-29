import numpy as np

# 데이터 생성
X = 2 * np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # 절편 추가

y = 4 + 3 * X + np.random.randn(100, 1)

# 학습률, 반복 횟수, 샘플 개수 설정
eta = 0.1
n_iterations = 1000
m = 100

# 파라미터 초기화
theta_gd = np.random.randn(2,1)  # GD용 파라미터
theta_bgd = np.random.randn(2,1) # BGD용 파라미터

# GD
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta_gd - eta * gradients

# BGD
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - eta * gradients

# 결과 출력
print("GD 결과:\n", theta_gd)
print("\nBGD 결과:\n", theta_bgd)
