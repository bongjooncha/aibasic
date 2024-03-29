import numpy as np
import matplotlib.pyplot as plt

# 가상의 데이터 생성
X = 2 * np.random.rand(100, 1) # 0과 2사이의 수 100개 생성(독립변수)
y = 4 + 3 * X + np.random.randn(100, 1) # 4+3*X에 노이즈가 추가된 y생성

# 편향을 위해 X에 1 추가
X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1  # 학습률
n_epochs = 50  # 에포크 횟수
m = 100  # 샘플 개수

# 학습 스케줄 파라미터 설정(하이퍼 파라미터임)
t0, t1 = 5, 50  
def learning_schedule(t):
    return t0 / (t + t1)

# SGD 구현
theta_sgd = np.random.randn(2, 1)  # 무작위 초기화
theta_sgd_path = []

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        eta = learning_schedule(epoch * m + i)
        theta_sgd = theta_sgd - eta * gradients
        theta_sgd_path.append(theta_sgd)

theta_sgd_path = np.array(theta_sgd_path)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.plot(theta_sgd_path[:, 0], theta_sgd_path[:, 1], "g-+", linewidth=1, label="SGD")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.title("Stochastic Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
