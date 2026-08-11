import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ربط واستدعاء الموديولات من مجلد src
from src.model import PINN
from src.physics import schrodinger_residual

def train():
    # 1. إعداد الأجهزة والبيئة
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. بناء النموذج والمُحسّن (Optimizer)
    model = PINN(input_dim=2, output_dim=2, hidden_layers=4, hidden_units=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. توليد نقاط التدريب (Collocation Points) للفيزياء
    N_f = 2000
    x_f = (2 * torch.rand(N_f, 1) - 1).to(device)  # x in [-1, 1]
    t_f = torch.rand(N_f, 1).to(device)           # t in [0, 1]

    # 4. الشروط الإبتدائية Initial Conditions (t=0)
    N_0 = 200
    x_0 = (2 * torch.rand(N_0, 1) - 1).to(device)
    t_0 = torch.zeros(N_0, 1).to(device)
    # حزمة موجية جارسية كشرط ابتدائي Psi(x,0) = exp(-x^2)
    u_0_target = torch.exp(-x_0**2)
    v_0_target = torch.zeros_like(x_0)

    # 5. حلقة التدريب (Training Loop)
    epochs = 1000
    loss_history = []

    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # حساب الخسارة للشرط الابتدائي
        u_0_pred, v_0_pred = model(x_0, t_0)
        loss_ic = torch.mean((u_0_pred - u_0_target)**2) + torch.mean((v_0_pred - v_0_target)**2)

        # حساب الخسارة الفيزيائية (Residual Loss)
        loss_p = schrodinger_residual(model, x_f, t_f)

        # الخسارة الكلية
        total_loss = loss_ic + 10.0 * loss_p

        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss.item():.6f} (IC: {loss_ic.item():.6f}, Physics: {loss_p.item():.6f})")

    # 6. رسم وحفظ منحنى الخسارة (Loss Curve)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Total Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("PINN Schrödinger Training Convergence")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_curve.png")
    print("Training Complete! Saved loss curve as 'loss_curve.png'.")

if __name__ == "__main__":
    train()