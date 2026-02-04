# InkTime · 墨水屏回忆相框

<p align="left">
  <img src="esp32/InkTime.jpeg" width="80%">
</p>

InkTime 是一个「拉回你相册里的记忆」的墨水屏电子相框项目。

它不会随机展示照片，也不是简单的按时间轴播放，而是：

- 用 AI 理解每一张照片在"“"拍什么"
- 给照片按照"值得回忆度"、"美观度"打分
- 写一句灵光一现的旁白文案
- 每天从"历史上的今天"里选出**最值得被再次看到的照片**
- 推送到 ESP32 墨水屏上展示

---
## 项目整体结构

InkTime 分为三部分：

1. **照片分析（Python）**  
   扫描相册 → 调用视觉模型 → 分类 / 评分 / 写文案 → 存入数据库


2. **图片渲染（Python）**  
   从数据库里选出「历史上的今天」高分照片 → 渲染成 ESP32 可直接显示的 `.bin`


3. **下载与展示（ESP32）**  
   ESP32 定时从服务器拉取 `.bin` → 刷新墨水屏 → 深度休眠直至下次唤醒

---
## 环境准备

### 1）Python
推荐 Python 3.10+。

建议使用虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2）安装 exiftool （可选）
InkTime 可以在不装 exiftool的情况下运行，但不一定能完整地获取 EXIF 中的 GPS 信息。

建议使用 exiftool 获取 GPS 信息：  

MacOS(Homebrew): ```brew install exiftool```  
Linux: ```sudo apt-get install -y libimage-exiftool-perl```

### 3) 配置 config.py
```
cp config-example.py config.py  
vi config.py
```
必须配置以下字段：  
照片库路径 ```IMAGE_DIR```  
VLM 模型接口 ```API_URL``` ```MODEL_NAME```  
InkTime 使用 OpenAI 接口（LM Studio / 其它兼容服务均可）。

为防止照片隐私泄露，建议修改```DOWNLOAD_KEY```，为 ESP32 下载路径加一个随机前缀作为密钥。   
同时，请同步修改```esp32/ink-display-7C-photo/ink-display-7C-photo.ino```固件中的```DAILY_PHOTO_PATH_PREFIX```字段。  
注意，这不是“加密”，只是一个简单的验证路径口令。公网部署建议加 HTTPS/反代鉴权，或只允许内网访问。

## 分析照片
分析照片前，请先确保：
- LM Studio（或你的云端 VLM 服务）已启动
- config.py 已正确配置

执行：

```python3 analyze_photos.py```

视觉大模型会读取并理解相册目录中的所有文件，为每张照片生成：

- 画面描述
- 照片类型
- 值得回忆度 / 画面美观度评分
- 一句话文案

图片数据会保存在```photos.db```中（SQLite数据库）。

请自行修改```analyze_photos.py```中的提示词，以调整模型的评价标准和文案风格。

程序可以断点续跑，已处理过的照片信息不会重复分析。你可以分几天分析完你的整个相册。

*请根据你拥有的算力选择合适的模型，作者使用的 qwen3-vl-30b 已经能取得相当不错的文案。*

## 为 ESP32 渲染"历史上的今天"照片
执行：

```python3 render_daily_photo.py```

## 启动 ESP32 下载服务器和 WebUI
执行：

```python3 server.py```

#### WebUI（如果开启）：
Server 将提供一个简明的可视化前端，用于查看已处理照片的描述、文案，并预览模拟墨水屏渲染效果。

在浏览器中访问：

```http://127.0.0.1:8765/review```

程序跑通后，建议在```config.py```中关闭WebUI，仅保留 ESP32 下载接口。

## 服务器部署与定时任务示例（可选）

创建 systemd 服务：

```sudo vi /etc/systemd/system/inktime-server.service```

示例（请自行修改项目路径）：

```
[Unit]
Description=InkTime Server
After=network.target

[Service]
Type=simple
# 改成你的项目路径
WorkingDirectory=/path/to/InkTime
ExecStart=/path/to/InkTime/venv/bin/python server.py
Restart=always
RestartSec=3
User=inktime
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl daemon-reload
sudo systemctl enable inktime-server
sudo systemctl start inktime-server
```

使用 crontab 每天凌晨自动选片、渲染：

```
chmod +x scripts/daily_render.sh
sudo -u inktime crontab -e
0 5 * * * /path/to/InkTime/scripts/daily_render.sh
```

在 ```logs/render.log```可查看日志。

