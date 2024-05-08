import torch
import torch.nn as nn
import torch.optim as optim

# 더미 데이터 생성
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.1  # y = 3X + 2 + noise

# 선형 회귀 모델 정의
model = nn.Linear(1, 1)  # 입력 차원: 1, 출력 차원: 1

# SGD 옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 손실 함수 정의
criterion = nn.MSELoss()

# 학습
epochs = 1000
for epoch in range(epochs):
    # 예측
    y_pred = model(X)
    
    # 손실 계산
    loss = criterion(y_pred, y)
    
    # 역전파 단계 전에 그래디언트를 0으로 설정
    optimizer.zero_grad()
    
    # 역전파 단계
    loss.backward()
    
    # 가중치 업데이트
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 학습된 모델 파라미터 확인
print('Learned parameters:')
for name, param in model.named_parameters():
    print(f'{name}: {param.item()}')
