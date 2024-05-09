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
lr = 0.01  # 학습률
n_epochs = 1000  # 반복 횟수
m = 100  # 샘플 개수

#초기 theta 값
start = np.random.randn(2, 1)  # 무작위 초기화


# MSE
theta_mse = start.copy()
theta_mse_path = []

for iteration in range(n_epochs):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_mse) - y)
    theta_mse = theta_mse - lr * gradients
    theta_mse_path.append(theta_mse)
    
theta_mse_path = np.array(theta_mse_path)


# MAE 구현
theta_mae = start.copy()
theta_mae_path = []

def mae_loss(y_true, y_pred):
    n = len(y_true)
    derivative = 0
    
    for i in range(n):
        diff = y_pred[i] - y_true[i]
        sign = 1 if diff > 0 else -1 if diff < 0 else 0
        derivative += sign
    
    derivative /= n
    
    return derivative

for iteration in range(n_epochs):  # 1회 반복 당 시행되는 과정
    gradients = mae_loss(y, X_b.dot(theta_mae))
    # print(gradients)
    theta_mae = theta_mae - lr * gradients
    theta_mae_path.append(theta_mae)
    
theta_mae_path = np.array(theta_mae_path)

# RMSE 구현
theta_rmse = start.copy()
theta_rmse_path = []

for iteration in range(n_epochs):  # 1회 반복 당 시행되는 과정
    gradients = 2*(X_b.dot(theta_mse) - y)/(m^0.5 * (X_b.dot(theta_mae))^2)
    theta_rmse = theta_rmse - lr * gradients
    theta_rmse_path.append(theta_rmse)
    
theta_rmse_path = np.array(theta_rmse_path)


# 애니메이션 설정
fig, ax = plt.subplots(figsize=(10, 8))

# # 함수 animation
# def animate(i):
#     plt.clf()
#     plt.plot(X, y, 'b.')  # 데이터 플롯

#     # Batch Gradient Descent (BGD)
#     plt.plot(X, X_b.dot(theta_mse_path[i]), 'r-', label="BGD")

#     # Stochastic Gradient Descent (mae)
#     plt.plot(X, X_b.dot(theta_mae_path[i]), 'g-', label="mae")

#     # Mini-batch Gradient Descent (MGD)
#     plt.plot(X, X_b.dot(theta_mgd_path[i]), 'b-', label="MGD")

#     plt.title(f"Iteration: {i}")
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid(True)

# ani = FuncAnimation(fig, animate, frames=len(theta_mse_path), interval=50)
# ani.save('./gradient_descent_animation.gif', writer='imagemagick')



# theta animation
def animate(i):
    plt.clf()
    # plt.plot(theta_mse_path[:i+1, 0], theta_mse_path[:i+1, 1], 'r-', label="MSE")
    # plt.scatter(theta_mse_path[i, 0], theta_mse_path[i, 1], color='red', s=10, marker='o')

    # plt.plot(theta_mae_path[:i+1, 0], theta_mae_path[:i+1, 1], 'g-', label="MAE")
    # plt.scatter(theta_mae_path[i, 0], theta_mae_path[i, 1], color='green', s=10, marker='o')

    plt.plot(theta_rmse_path[:i+1, 0], theta_rmse_path[:i+1, 1], 'b-', label="RMSE")
    plt.scatter(theta_rmse_path[i, 0], theta_rmse_path[i, 1], color='blue', s=10, marker='o')


    plt.title(f"Iteration: {i}")
    plt.xlabel('Theta_0')
    plt.ylabel('Theta_1')
    plt.legend()
    plt.grid(True)

ani = FuncAnimation(fig, animate, frames=len(theta_mse_path), interval=50)
# ani.save('./lossfunction_theta_animation.gif', writer='imagemagick')

plt.show()

# theta_mse_2d = np.squeeze(theta_mse_path, axis=2)
# theta_mae_2d = np.squeeze(theta_mae_path, axis=2)
# theta_mgd_2d = np.squeeze(theta_mgd_path, axis=2)

# # theta_mse_2d, theta_mae_2d, theta_mgd_2d를 데이터프레임으로 변환
# df_bgd = pd.DataFrame(theta_mse_2d, columns=['Theta0', 'Theta1'])
# df_mae = pd.DataFrame(theta_mae_2d, columns=['Theta0', 'Theta1'])
# df_mgd = pd.DataFrame(theta_mgd_2d, columns=['Theta0', 'Theta1'])

# # 데이터프레임 출력
# print("Batch Gradient Descent:")
# print(df_bgd.head())

# print("\nStochastic Gradient Descent:")
# print(df_mae.head())

# print("\nMini-batch Gradient Descent:")
# print(df_mgd.head())