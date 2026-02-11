import torch
import cv2
import numpy as np
import os
import sys

# --- 1. 自动化环境配置 ---
# 检测是否在 GitHub Actions 环境运行
IS_GITHUB = 'GITHUB_ACTIONS' in os.environ

# --- 2. 核心校准参数 ---
CAM_HEIGHT = 1.5      
HORIZON_Y_RATIO = 0.45 
F_PIXELS = 850        

# --- 3. 模型转换逻辑 (GitHub Actions 专用) ---
def export_for_ios(model):
    print("正在启动模型转换流程 [CoreML]...")
    import coremltools as ct
    
    model.eval()
    example_input = torch.rand(1, 3, 320, 320).to(next(model.parameters()).device)
    
    # 使用 TorchScript 追踪模型
    traced_model = torch.jit.trace(model, example_input)
    
    # 转换为 CoreML 格式
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=example_input.shape, 
                             scale=1/255.0, color_layout=ct.colorlayout.RGB)]
    )
    mlmodel.save("YOLOP_ADAS.mlpackage")
    print("✅ 转换完成: YOLOP_ADAS.mlpackage 已生成")

# --- 4. 初始化模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True).to(device).eval()

# 如果运行参数包含 --export，则执行转换并退出
if "--export" in sys.argv:
    export_for_ios(model)
    sys.exit(0)

# --- 5. NMS 去重函数 (保持原样) ---
def apply_nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0: return []
    boxes = np.array(boxes); scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.maximum(y1[i], y1[order[1:]]), np.maximum(y1[i], y1[order[1:]]) # 修正
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - (w * h))
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return keep

# --- 6. 视频处理主循环 ---
cap = cv2.VideoCapture("car-detection.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h_ori, w_ori = frame.shape[:2]

    # 推理预处理
    img = cv2.resize(frame, (320, 320))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    input_tensor = torch.from_numpy(np.ascontiguousarray(img)).to(device).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        det_out = outputs[0]

    # 解析结果
    preds = det_out[0].squeeze(0) if isinstance(det_out, (list, tuple)) else det_out.squeeze(0)
    valid_dets = preds[preds[:, 4] > 0.45].cpu().numpy()
    boxes_nms, scores_nms = [], []
    for det in valid_dets:
        cx, cy, w, h, conf, cls = det
        boxes_nms.append([int((cx-w/2)*w_ori/320), int((cy-h/2)*h_ori/320), 
                          int((cx+w/2)*w_ori/320), int((cy+h/2)*h_ori/320)])
        scores_nms.append(conf)
    
    keep_idx = apply_nms(boxes_nms, scores_nms)
    res = frame.copy()
    horizon_pixel = h_ori * HORIZON_Y_RATIO
    
    for idx in keep_idx:
        x1, y1, x2, y2 = boxes_nms[idx]
        y_pixel_diff = max(1.0, float(y2 - horizon_pixel))
        dist_m = round((CAM_HEIGHT * F_PIXELS) / y_pixel_diff, 1)
        
        color = (0, 255, 0) if dist_m > 15 else (0, 0, 255)
        cv2.rectangle(res, (x1, y1), (x2, y2), color, 3)
        
        label = f"{dist_m}m"
        cv2.putText(res, label, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2)

    # --- 7. 保护措施：GitHub 环境不执行 GUI 显示 ---
    if not IS_GITHUB:
        DISPLAY_W = 1000
        show_res = cv2.resize(res, (DISPLAY_W, int(h_ori * (DISPLAY_W/w_ori))))
        cv2.imshow("ADAS Local Debug", show_res)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        # 如果在云端，我们可以选择把结果存成视频
        print("云端处理中...")
        break 

cap.release()
cv2.destroyAllWindows()
