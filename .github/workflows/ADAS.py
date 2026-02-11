import torch
import cv2
import numpy as np
import os
import sys

# --- 1. 参数设置 ---
CAM_HEIGHT = 1.5      
HORIZON_Y_RATIO = 0.45 
F_PIXELS = 850        

# --- 2. 导出 CoreML 逻辑 (iOS 专用优化版) ---
def export_for_ios(model):
    print("🚀 开始导出 CoreML 模型...")
    try:
        import coremltools as ct
        model.eval()
        
        # YOLOP 标准输入尺寸 320x320
        example_input = torch.rand(1, 3, 320, 320)
        
        # 使用 TorchScript 追踪模型
        traced_model = torch.jit.trace(model, example_input)
        
        # 转换配置：定义输入为图像，并设置预处理参数
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(
                name="image", 
                shape=example_input.shape, 
                scale=1/255.0, 
                color_layout=ct.colorlayout.RGB
            )]
        )
        
        # 保存模型
        mlmodel.save("YOLOP_ADAS.mlpackage")
        print("✅ 导出成功: YOLOP_ADAS.mlpackage")
    except Exception as e:
        print(f"❌ 导出过程出错: {e}")

# --- 3. 模型加载逻辑 ---
# 强制使用 CPU 提高云端稳定性
device = torch.device("cpu")

def load_model():
    print("📦 正在从 TorchHub 加载 YOLOP 预训练模型...")
    # trust_repo=True 是必须的，否则 Actions 会因为权限问题卡住
    return torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True).to(device).eval()

# --- 4. 执行入口 ---
if __name__ == "__main__":
    # 无论在本地还是云端，先初始化模型
    model = load_model()

    # 如果带有 --export 参数，则执行转换逻辑并退出
    if "--export" in sys.argv:
        export_for_ios(model)
        sys.exit(0)

    # 如果是本地运行（非 GitHub 环境），启动摄像头调试
    if 'GITHUB_ACTIONS' not in os.environ:
        print("💻 检测到本地环境，启动调试模式...")
        cap = cv2.VideoCapture(0) # 改为 0 可以直接调用你电脑摄像头
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # 简单的 ADAS 激活提示
            cv2.putText(frame, "ADAS SYSTEM ACTIVE", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Local Debug", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()
