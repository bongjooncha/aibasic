import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 선형 회귀 모델 예시
x = np.linspace(-10, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 0.5, 100)

def model(x, w, b):
  return w * x + b

def loss(y_pred, y_true):
  return np.mean((y_pred - y_true)**2)

def visualize(x, y, w, b, losses, algo_name):
  # 모델 예측값 계산
  y_pred = model(x, w[-1], b[-1])

  # 그래프 초기화
  fig, ax = plt.subplots(figsize=(10, 8))
  ax.set_title(f"{algo_name} 알고리즘 시각화")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.scatter(x, y, c="blue", label="data")
  line, = ax.plot(x, y_pred, c="red", label="model asum")
  loss_line, = ax.plot(losses, c="green", label="loss")
  plt.legend()

  # 애니메이션 함수
  def animate(i):
    # 모델 업데이트
    y_pred_new = model(x, w[i], b[i])
    
    # 그래프 업데이트
    line.set_ydata(y_pred_new)
    loss_line.set_data(list(range(i+1)), losses[:i+1])
    
    # 이전 예측값 라인 제거 (애니메이션 자연스러움 향상)
    if i > 0:
      ax.lines.remove(line)
    line = ax.plot(x, y_pred_new, c="red", label="모델 예측")
    
    return line, loss_line

  # 애니메이션 시작
  anim = animation.FuncAnimation(fig, animate, interval=10, frames=len(losses))
  plt.show()



def gd(x, y, w_init, b_init, eta, n_iter):
  w = [w_init]
  b = [b_init]
  losses = []
  
  for i in range(n_iter):
    # 예측값 계산
    y_pred = model(x, w[-1], b[-1])
    
    # 손실값 계산
    loss_val = loss(y_pred, y)
    losses.append(loss_val)
    
    # 기울기 계산
    dw = 2 * np.mean((y_pred - y) * x)
    db = 2 * np.mean(y_pred - y)
    
    # 가중치 업데이트
    w.append(w[-1] - eta * dw)
    b.append(b[-1] - eta * db)
  
  return w, b, losses


w_gd, b_gd, losses_gd = gd(x, y, 0.01, 0.01, 100, n_iter=100)
visualize(x, y, w_gd, b_gd, losses_gd, "GD")