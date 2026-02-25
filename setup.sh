#!/bin/bash
# ==============================================
# Univisual Segmentation API - 华为云一键部署脚本
# 适用系统: Ubuntu 22.04 LTS
# 用法: bash setup.sh
# ==============================================
set -e

echo "=============================================="
echo "🔬 Univisual Segmentation API — 华为云部署"
echo "=============================================="

# 1. 系统更新
echo "[1/6] 更新系统..."
sudo apt update && sudo apt upgrade -y

# 2. 安装 Python 3.10 和 pip
echo "[2/6] 安装 Python..."
sudo apt install -y python3 python3-pip python3-venv git

# 3. 克隆代码
echo "[3/6] 克隆代码仓库..."
cd /opt
if [ -d "univisual-segmentation-api" ]; then
    cd univisual-segmentation-api && git pull
else
    sudo git clone https://github.com/aaronliu9999mu/univisual-segmentation-api.git
    cd univisual-segmentation-api
fi
sudo chown -R $USER:$USER /opt/univisual-segmentation-api

# 4. 创建虚拟环境并安装依赖
echo "[4/6] 安装 Python 依赖 (含 Cellpose, 可能需要几分钟)..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. 配置 systemd 服务
echo "[5/6] 配置开机自启服务..."
sudo tee /etc/systemd/system/univisual-api.service > /dev/null << 'EOF'
[Unit]
Description=Univisual Segmentation API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/univisual-segmentation-api
ExecStart=/opt/univisual-segmentation-api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable univisual-api
sudo systemctl start univisual-api

# 6. 开放防火墙端口
echo "[6/6] 开放 8000 端口..."
sudo ufw allow 8000/tcp 2>/dev/null || true
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT 2>/dev/null || true

echo ""
echo "=============================================="
echo "✅ 部署完成！"
echo "=============================================="
echo ""
echo "本地测试: curl http://localhost:8000/health"
echo "外部测试: curl http://$(curl -s ifconfig.me):8000/health"
echo ""
echo "管理命令:"
echo "  查看状态: sudo systemctl status univisual-api"
echo "  查看日志: sudo journalctl -u univisual-api -f"
echo "  重启服务: sudo systemctl restart univisual-api"
echo ""
