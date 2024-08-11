import torch
import torch.nn as nn
import torch.optim as optim


# 創建一個簡單的線性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 創建模型實例
model = SimpleModel()

# 檢查初始 requires_grad 狀態
print("Initial requires_grad status:")
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad} {param.data.requires_grad}")

# 禁用線性層的權重梯度計算
model.linear.weight.requires_grad = False

# 檢查 requires_grad 狀態
print("\nAfter disabling gradients:")
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad} {param.data.requires_grad}")

# 啟用線性層的權重梯度計算
model.linear.weight.requires_grad = True

# 再次檢查 requires_grad 狀態
print("\nAfter enabling gradients:")
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad} {param.data.requires_grad}")


# 創建模型和優化器
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 創建一些訓練數據
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# 訓練循環
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 設置模型為訓練模式
    optimizer.zero_grad()  # 清零梯度
    outputs = model(inputs)  # 前向傳播
    loss = criterion(outputs, targets)  # 計算損失
    loss.backward()  # 反向傳播
    optimizer.step()  # 更新參數

    # 印出更新後的參數
    print(f"Epoch {epoch + 1}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    # Optional: 印出損失值
    print(f"Loss: {loss.item()}\n")

# 檢查 requires_grad 狀態
print("\nFinal requires_grad status:")
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")