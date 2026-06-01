"""M3：TACE — 三级自适应级联增强。"""
import cv2
import numpy as np


class Enhancer:
    """漫画文本区域的三级增强。

    第一级：CLAHE + Unsharp Mask（传统CV，<1ms）
    第二级：透传（SVTR原生抗模糊能力处理）
    第三级：Real-ESRGAN超分辨率（仅在低置信度回退时使用）
    """

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, image: np.ndarray, tier: int = 1) -> np.ndarray:
        """应用指定级别的增强。"""
        if tier == 1:
            return self._tier1_clahe_sharpen(image)
        elif tier == 2:
            return self._tier2_passthrough(image)
        elif tier == 3:
            return self._tier3_super_resolution(image)
        else:
            raise ValueError(f"未知增强级别: {tier}")

    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """自动应用第一级增强。更高级别在OCR置信度不足时由外部触发。"""
        return self._tier1_clahe_sharpen(image)

    def _tier1_clahe_sharpen(self, image: np.ndarray) -> np.ndarray:
        """CLAHE对比度增强 + Unsharp Mask锐化。"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            merged = cv2.merge([l, a, b])
            result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            result = self._clahe.apply(image)
        result = self._sharpen(result)
        return result

    def _tier2_passthrough(self, image: np.ndarray) -> np.ndarray:
        """透传 — SVTR的原生抗模糊训练处理轻度模糊。"""
        return image

    def _tier3_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """Real-ESRGAN超分辨率处理严重模糊的文本区域。"""
        import torch
        from src.core.model_manager import ModelManager

        mgr = ModelManager()
        esrgan = mgr.get_model("enhancement", "real_esrgan")
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            output = esrgan(tensor)
        result = output.squeeze(0).permute(1, 2, 0).numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _sharpen(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """应用Unsharp Mask锐化。"""
        blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
        result = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return result
