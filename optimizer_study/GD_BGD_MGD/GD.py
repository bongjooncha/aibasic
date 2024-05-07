import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 가상의 데이터 생성
np.random.seed(42)      #다른 GD와 비교하기 위해서 seed 정의

X = 2 * np.random.rand(100, 1)           # 0과 2사이의 수 100개 생성(독립변수)
y = 4 + 3 * X + np.random.randn(100, 1)  # 4+3*X에 노이즈가 추가된 y생성(종속변수)
                                         # theta0이 4, theta1이 3에 가깝게 나와야함

# 편향을 위해 X에 1 추가
X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1                   # 학습률
n_iterations = 1000         # 반복 횟수(1epoch에서 반복되는 회수)
m = 100                     # 샘플 개수 100개

# Gradient Descent (GD) 구현
theta_gd = np.random.randn(2, 1)  # 세타값을 무작위로 생성
theta_gd_path = []                # 앞으로 생성될 세타 쌍을 저장하는 공간

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta_gd - eta * gradients
    theta_gd_path.append(theta_gd)

theta_gd_path = np.array(theta_gd_path)

print(gradients)
print(theta_gd)

def animate(i):
    plt.clf()
    plt.plot(X, y, 'b.')  # 데이터 플롯
    plt.plot(X, X_b.dot(theta_gd_path[i]), 'r-')  # 선형 회귀 선 플롯
    theta0 = float(theta_gd_path[i, 0])
    theta1 = float(theta_gd_path[i, 1])
    plt.title(f"Iteration: {i}, theta0: {theta0:.2f}, theta1: {theta1:.2f}")
    plt.xlabel('X')
    plt.ylabel('y')

# 애니메이션 설정
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=len(theta_gd_path), interval=50)

plt.show()
