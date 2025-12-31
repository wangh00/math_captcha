import os
import re
from io import BytesIO

import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
import uuid
import onnxruntime as ort
import yaml

os.environ['YOLO_VERBOSE'] = 'False'  # 关闭详细日志
ort.set_default_logger_severity(3)


def extract_arithmetic_expression(text):
    """
    从识别文本中提取算术表达式部分

    参数:
    text: OCR识别出的文本

    返回:
    提取到的算术表达式字符串
    """
    # def check_expression(exp):
    #     exp1,exp2=exp.split('-')
    #     if exp1.isdigit() and exp2.isdigit() and int(exp1)<int(exp2):
    #

    # 清理文本，移除多余空格和干扰字符
    text = re.sub(r'[xX×]', '*', text)
    re_com_string = r'[^\d+\-*/().=]'
    cleaned_text = re.sub(re_com_string, '', text.replace(' ', ''))
    # 查找等号的位置
    equal_index = cleaned_text.find('=')
    if equal_index != -1:
        # 提取等号前的部分作为表达式
        expression = cleaned_text[:equal_index]
        return expression
    else:
        return cleaned_text


def safe_calculate_expression(expression):
    """
    安全地计算算术表达式

    参数:
    expression: 算术表达式字符串

    返回:
    计算结果
    """
    # 使用正则表达式验证表达式是否只包含数字和算术运算符
    if not re.match(r'^[\d+\-*/().]+$', expression):
        raise ValueError("表达式包含非法字符")

    # 简单的表达式计算（避免使用eval的安全方法）
    # 这里使用eval，但在生产环境中应该使用更安全的方法
    try:
        result = eval(expression)
        return result
    except:
        raise ValueError(f"无法计算表达式:{expression}")


class ArithmeticCaptchaSolver:

    def __init__(self,model_path):
        self.model = YOLO(model_path, verbose=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open('./dataset.yaml', 'r', encoding='utf-8') as file:
            self.yaml_data = yaml.safe_load(file)
        # 映射类别ID到实际字符
        self.class_names = self.yaml_data['names']

    def recognized(self, image_bytes: bytes):
        image = Image.open(BytesIO(image_bytes))
        results = self.model(image, device=self.device, verbose=False)
        for result in results:
            # 获取检测到的框-
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # 提取类别、置信度和坐标
                classes = boxes.cls.cpu().numpy()  # 类别ID
                confidences = boxes.conf.cpu().numpy()  # 置信度
                xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标

                # 按x坐标排序（从左到右）
                sorted_indices = np.argsort(xyxy[:, 0])  # 按第一个x坐标排序
                sorted_classes = classes[sorted_indices]
                sorted_confidences = confidences[sorted_indices]

                # 输出识别结果
                # print("识别结果（从左到右）:")
                recognized_text = ""
                for i, cls_id in enumerate(sorted_classes):
                    char = self.class_names.get(int(cls_id), '?')
                    confidence = sorted_confidences[i]
                    recognized_text += char
                #     print(f"  位置{i + 1}: {char} (置信度: {confidence:.2%})")
                #
                # print(f"\n最终识别文本: {recognized_text}")
                return recognized_text
            else:
                print("未检测到任何目标")

    def solve_captcha(self, image_bytes: bytes):
        recognized_text = self.recognized(image_bytes)
        expression = extract_arithmetic_expression(recognized_text)
        # print(expression)
        result = safe_calculate_expression(expression)
        return result


if __name__ == '__main__':
    with open("./0ad75de6-fd55-49cf-988a-185d4514144f.jpg", "rb") as f:
        image = f.read()
    obj = ArithmeticCaptchaSolver("./best.onnx")
    print('结果:', obj.solve_captcha(image))
