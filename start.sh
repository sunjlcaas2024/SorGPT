#!/bin/bash
cd "$(dirname "$0")"
echo "=========================================="
echo "  SorGPT API 服务启动"
echo "=========================================="
echo ""
echo "项目目录: $(pwd)"
echo ""
echo "请选择运行模式:"
echo "  1) 开发模式 (debug, 前端访问 http://服务器IP:8000)"
echo "  2) 生产模式 (后台运行, 日志到 app.log)"
echo "  3) 仅检查状态"
echo ""
read -p "请输入选择 [1/2/3]: " choice

case $choice in
    1)
        echo "正在启动开发模式..."
        python api_server.py
        ;;
    2)
        echo "正在后台启动生产模式..."
        nohup python api_server.py > app.log 2>&1 &
        sleep 3
        if ps aux | grep -v grep | grep api_server.py > /dev/null; then
            echo "服务已启动!"
            echo "访问地址: http://$(hostname -I | awk '{print $1}'):8000"
            echo "日志文件: $(pwd)/app.log"
        else
            echo "启动失败，请查看 app.log"
        fi
        ;;
    3)
        if ps aux | grep -v grep | grep api_server.py > /dev/null; then
            echo "服务状态: 运行中"
        else
            echo "服务状态: 未运行"
        fi
        ;;
esac
