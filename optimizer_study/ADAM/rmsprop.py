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
eta = 0.1                   # 초기 학습률
n_iterations = 300          # 반복 횟수
m = 100                     # 샘플 개수 100개

# RMSProp 구현
theta_rmsprop = np.random.randn(2, 1)  # 세타값을 무작위로 생성
theta_rmsprop_path = []                # 앞으로 생성될 세타 쌍을 저장하는 공간

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

# 결과를 시각화하기 위해 애니메이션 생성
fig, ax = plt.subplots()
ax.plot(X, y, "b.")
line, = ax.plot(X, X_b.dot(theta_rmsprop), "r-")

def update(frame):
    line.set_ydata(X_b.dot(theta_rmsprop_path[frame]))
    return line,

ani = FuncAnimation(fig, update, frames=n_iterations, blit=True)
plt.show()
