import torch
import cv2
import numpy as np
import os
import sys

# --- 1. 参数设置 ---
CAM_HEIGHT = 1.5      
HORIZON_Y_RATIO = 0.45 
F_PIXELS = 850        

# --- 2. 导出 CoreML 逻辑 (iOS 专用) ---
def export_for_ios(model):
    print("开始导出 CoreML 模型...")
    try:
        import coremltools as ct
        model.eval()
        # YOLOP 标准输入尺寸
        example_input = torch.rand(1, 3, 320, 320)
        traced_model = torch.jit.trace(model, example_input)
        
        # 转换并设置输入类型为图像
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="image", shape=example_input.shape, 
                                 scale=1/255.0, color_layout=ct.colorlayout.RGB)]
        )
        mlmodel.save("YOLOP_ADAS.mlpackage")
        print("✅ 导出成功: YOLOP_ADAS.mlpackage")
    except Exception as e:
        print(f"❌ 导出过程出错: {e}")

# --- 3. 模型加载 ---
# GitHub Actions 环境没有 GPU，强制使用 CPU
device = torch.device("cpu")
print("正在从 TorchHub 加载预训练模型...")
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True).to(device).eval()

# 判断是否是云端导出指令
if "--export" in sys.argv:
    export_for_ios(model)
    sys.exit(0)

# --- 4. 本地调试逻辑 ---
def run_local():
    cap = cv2.VideoCapture("car-detection.mp4")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        h_ori, w_ori = frame.shape[:2]

        # 预处理与推理
        img = cv2.resize(frame, (320, 320))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        input_tensor = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            det_out = model(input_tensor)[0]
        
        # 简化版渲染 (为了演示)
        res = frame.copy()
        cv2.putText(res, "ADAS ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Debug", res)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if 'GITHUB_ACTIONS' not in os.environ:
        run_local()
