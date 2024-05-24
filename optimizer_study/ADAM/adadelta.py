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
n_iterations = 300          # 반복 횟수
m = 100                     # 샘플 개수 100개

# AdaDelta 구현
theta_adadelta = np.random.randn(2, 1)  # 세타값을 무작위로 생성
theta_adadelta_path = []                # 앞으로 생성될 세타 쌍을 저장하는 공간

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

# 결과를 시각화하기 위해 애니메이션 생성
fig, ax = plt.subplots()
ax.plot(X, y, "b.")
line, = ax.plot(X, X_b.dot(theta_adadelta), "r-")

def update(frame):
    line.set_ydata(X_b.dot(theta_adadelta_path[frame]))
    return line,

ani = FuncAnimation(fig, update, frames=n_iterations, blit=True)
plt.show()
