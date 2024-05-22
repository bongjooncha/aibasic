import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 가상의 데이터 생성
np.random.seed(42)  # 다른 GD와 비교하기 위해서 seed 정의

X = 2 * np.random.rand(100, 1)  # 0과 2사이의 수 100개 생성(독립변수)
y = 4 + 3 * X + np.random.randn(100, 1)  # 4+3*X에 노이즈가 추가된 y생성(종속변수)
                                         # theta0이 4, theta1이 3에 가깝게 나와야함

# 편향을 위해 X에 1 추가
X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1                   # 학습률
n_iterations = 300          # 반복 횟수
m = 100                     # 샘플 개수 100개

# NAG 구현
theta_nag = np.random.randn(2, 1)  # 세타값을 무작위로 생성
theta_nag_path = []                # 앞으로 생성될 세타 쌍을 저장하는 공간

# 모멘텀 상수
a = 0.1
mt = np.zeros((2, 1))  # 모멘텀 값을 초기화

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    lookahead_theta = theta_nag + a * mt  # 현재 모멘텀을 반영한 미래의 위치 예측
    gradients = 2/m * X_b.T.dot(X_b.dot(lookahead_theta) - y)
    mt = a * mt - eta * gradients
    theta_nag = theta_nag + mt
    theta_nag_path.append(theta_nag.copy())

theta_nag_path = np.array(theta_nag_path)

# 결과를 시각화하기 위해 애니메이션 생성
fig, ax = plt.subplots()
ax.plot(X, y, "b.")
line, = ax.plot(X, X_b.dot(theta_nag), "r-")

def update(frame):
    line.set_ydata(X_b.dot(theta_nag_path[frame]))
    return line,

ani = FuncAnimation(fig, update, frames=n_iterations, blit=True)
plt.show()
