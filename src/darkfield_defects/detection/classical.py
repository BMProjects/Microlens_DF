"""经典检测器 — 传统视觉高召回候选 + 形态学精修 + 多类缺陷分类."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy import ndimage

from darkfield_defects.detection.base import BaseDetector, DefectInstance, DefectType, DetectionResult
from darkfield_defects.detection.features import compute_candidate_map
from darkfield_defects.detection.params import DetectionParams, PreprocessParams, ScoringParams
from darkfield_defects.detection.preprocess import preprocess_image
from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


# ── 缺陷分类阈值 ──────────────────────────────────────────────
DAMAGE_AREA_THRESHOLD = 5000   # 面积超过此值 → 大面积缺损
SPOT_ASPECT_RATIO = 3.0        # 细长比小于此值 → 斑点/凹坑


class ClassicalDetector(BaseDetector):
    """基于传统视觉的经典检测器.

    流程：
    1. 可选 CLAHE 增强
    2. 多特征候选图生成（高召回）
    3. 阈值分割 + 形态学清理
    4. 密度检测 → CRASH 类型
    5. 残余连通域 → 骨架化实例化 + 多类分类
    6. Prominence 过滤（非划痕类）
    7. 划痕合并（旋转包围盒延伸 / 端点距离+方向）
    """

    def __init__(
        self,
        detection_params: Optional[DetectionParams] = None,
        preprocess_params: Optional[PreprocessParams] = None,
        scoring_params: Optional[ScoringParams] = None,
    ):
        self.det_params = detection_params or DetectionParams()
        self.pre_params = preprocess_params or PreprocessParams()
        self.scoring_params = scoring_params or ScoringParams()
        self.det_params.ensure_valid()

    def detect(
        self,
        image: np.ndarray,
        background: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        preprocessed_image: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """执行完整检测流水线."""
        logger.info(f"开始检测: image shape={image.shape}")

        if preprocessed_image is not None:
            corrected = preprocessed_image
            if roi_mask is None:
                raise ValueError("使用 preprocessed_image 时必须同时提供 roi_mask")
            optical_center, lens_radius = _estimate_center_from_roi(roi_mask)
            background_used = True
            logger.info("使用外部预处理图像, 跳过内部预处理")
        else:
            prep = preprocess_image(image, background, self.pre_params)
            corrected = prep.corrected
            if roi_mask is None:
                roi_mask = prep.roi_mask
            optical_center = prep.optical_center
            lens_radius = prep.lens_radius
            background_used = prep.background_used

        # ① 暗场图像增强（gamma + CLAHE）— 仅用于线状滤波器
        # brightness_channel 在原始图上运行，避免增强放大噪声导致 crash 误检
        original_for_bright = corrected.copy()
        if self.det_params.enhance_enabled:
            corrected = self._enhance_darkfield(corrected)
        elif self.det_params.clahe_enabled:
            corrected = self._apply_clahe(corrected)

        # ② 候选图生成（增强图驱动线状检测，原始图驱动亮度检测）
        candidate_map = compute_candidate_map(
            corrected, self.det_params, roi_mask,
            original_image=original_for_bright,
        )

        # ③ 阈值分割
        binary = self._threshold(candidate_map)

        # ④ 形态学清理
        binary = self._morphological_clean(binary, roi_mask)

        # ⑤ 密度检测（CRASH）+ 残余实例提取 + 分类
        instances = self._extract_instances(binary, corrected, optical_center, lens_radius)

        # ⑥ 构建完整掩码
        final_mask = np.zeros_like(binary, dtype=bool)
        for inst in instances:
            final_mask |= inst.mask

        result = DetectionResult(
            mask=final_mask.astype(np.uint8) * 255,
            instances=instances,
            metadata={
                "detector": "classical",
                "optical_center": optical_center,
                "lens_radius": lens_radius,
                "background_used": background_used,
                "roi_mask": roi_mask,
            },
        )

        n_scratch = result.num_scratches
        n_spot = result.num_spots
        n_damage = result.num_damages
        n_crash = result.num_crashes
        logger.info(
            f"检测完成: {n_scratch} 划痕, {n_spot} 斑点, {n_damage} 缺损, "
            f"{n_crash} 密集区, "
            f"总长度={result.total_length:.0f}px, 总面积={result.total_area}px"
        )
        return result

    # ── 图像增强 ────────────────────────────────────────────────

    def _enhance_darkfield(self, image: np.ndarray) -> np.ndarray:
        """暗场图像两阶段增强: gamma提亮 + CLAHE局部对比度.

        两步均为逐像素/小窗口操作，不会扩大缺陷空间范围。
        gamma<1 将暗区非线性提亮，使暗淡划痕灰度拉升到可分辨范围；
        CLAHE 在局部窗口内均衡化，突出微弱结构对比度。
        """
        img8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

        # Step 1: Gamma 校正 — 提亮暗区
        gamma = self.det_params.enhance_gamma
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        enhanced = cv2.LUT(img8, lut)

        # Step 2: CLAHE 局部对比度均衡
        if self.det_params.clahe_enabled:
            enhanced = self._apply_clahe(enhanced)

        logger.debug(
            f"暗场增强: gamma={gamma}, "
            f"clahe={'ON' if self.det_params.clahe_enabled else 'OFF'}"
        )
        return enhanced

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """对图像应用 CLAHE 对比度增强."""
        img8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
        tile = self.det_params.clahe_tile_size
        clahe = cv2.createCLAHE(
            clipLimit=self.det_params.clahe_clip_limit,
            tileGridSize=(tile, tile),
        )
        return clahe.apply(img8)

    # ── 阈值 ──────────────────────────────────────────────────

    def _threshold(self, candidate_map: np.ndarray) -> np.ndarray:
        """对候选图做阈值分割."""
        img8 = (candidate_map * 255).astype(np.uint8)

        method = self.det_params.threshold_method
        if method == "otsu":
            otsu_val, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # tile 模式防噪底限：防止纯暗 tile 无双峰分布导致阈值过低
            floor_val = int(self.det_params.otsu_floor * 255)
            if floor_val > 0 and otsu_val < floor_val:
                logger.debug(f"otsu_floor 生效: Otsu={otsu_val} < floor={floor_val}, 提升阈值")
                _, binary = cv2.threshold(img8, floor_val, 255, cv2.THRESH_BINARY)
        elif method == "triangle":
            _, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        elif method == "adaptive":
            block = self.det_params.adaptive_block_size
            if block % 2 == 0:
                block += 1
            binary = cv2.adaptiveThreshold(
                img8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block,
                -int(self.det_params.adaptive_c),  # 负值因为检测亮区
            )
        else:
            _, binary = cv2.threshold(img8, 30, 255, cv2.THRESH_BINARY)

        return binary.astype(bool)

    # ── 形态学清理 ────────────────────────────────────────────

    def _morphological_clean(
        self,
        binary: np.ndarray,
        roi_mask: np.ndarray,
    ) -> np.ndarray:
        """形态学清理：开运算去点 + 面积过滤."""
        binary_uint8 = binary.astype(np.uint8) * 255

        k = self.det_params.morph_open_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel)

        cleaned[~roi_mask] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.det_params.min_area:
                cleaned[labels == i] = 0

        return cleaned.astype(bool)

    # ── 实例提取（含密度检测 + prominence + 分类 + 合并）────────

    def _extract_instances(
        self,
        binary: np.ndarray,
        image: np.ndarray,
        optical_center: tuple[int, int],
        lens_radius: float,
    ) -> list[DefectInstance]:
        """从二值掩码提取缺陷实例并进行多类分类.

        Steps:
        1. 密度检测 → CRASH 类型（从掩码中分离）
        2. 残余连通域标记 + 骨架化
        3. 形态分析 + 分类: scratch / spot / damage
        4. Prominence 过滤（非划痕类）
        5. 划痕合并
        """
        binary_uint8 = binary.astype(np.uint8) * 255
        instances: list[DefectInstance] = []

        # ─── 1. 密度检测（CRASH） ───────────────────────────
        crash_instances, residual_mask = self._detect_crashes(
            binary_uint8, image, optical_center, lens_radius,
        )
        instances.extend(crash_instances)

        # ─── 2. 残余连通域的骨架化 + 实例化 ──────────────────
        residual_uint8 = residual_mask.astype(np.uint8)
        skeleton = (
            cv2.ximgproc.thinning(residual_uint8 * 255)
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning')
            else self._thin_fallback(residual_uint8)
        )
        skeleton_bool = skeleton > 0

        num_labels, labels = cv2.connectedComponents(residual_uint8)

        for label_id in range(1, num_labels):
            mask_i = (labels == label_id)
            area = int(np.count_nonzero(mask_i))
            if area < self.det_params.min_area:
                continue

            skel_i = skeleton_bool & mask_i
            skel_coords = np.argwhere(skel_i)
            length = float(np.count_nonzero(skel_i))
            avg_width = area / max(length, 1.0) if length > 0 else float(np.sqrt(area))
            aspect = length / max(avg_width, 1.0) if length > 0 else 1.0
            circularity = self._compute_circularity(mask_i)

            rows, cols = np.where(mask_i)
            x, y = int(cols.min()), int(rows.min())
            w, h = int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)
            bbox = (x, y, w, h)

            scatter = float(np.mean(image[mask_i])) if np.any(mask_i) else 0.0
            endpoints = self._find_endpoints(skel_coords) if len(skel_coords) >= 2 else None

            coords_for_zone = skel_coords if len(skel_coords) >= 2 else np.argwhere(mask_i)
            zone = self._classify_zone(
                coords_for_zone, optical_center, lens_radius,
                self.scoring_params.zone_center_ratio,
                self.scoring_params.zone_transition_ratio,
            )

            defect_type = self._classify_defect_type(area, aspect, circularity, length)

            # Prominence 计算
            prominence = self._compute_prominence(bbox, image)

            # Prominence 过滤：仅对非划痕类生效
            if defect_type != DefectType.SCRATCH:
                if prominence < self.det_params.prominence_min_value:
                    continue

            inst = DefectInstance(
                instance_id=len(instances),
                defect_type=defect_type,
                skeleton_coords=skel_coords,
                mask=mask_i,
                length_px=length,
                area_px=area,
                avg_width_px=avg_width,
                bbox=bbox,
                scatter_intensity=scatter,
                prominence=prominence,
                zone=zone,
                endpoints=endpoints,
                circularity=circularity,
                aspect_ratio=aspect,
            )
            instances.append(inst)

        # ─── 3. 划痕合并 ────────────────────────────────────
        merge_method = self.det_params.merge_method
        if merge_method == "endpoint":
            instances = self._merge_fragments_endpoint(instances)
        elif merge_method == "rotated_box":
            instances = self._merge_fragments_rotated_box(instances, image)
        elif merge_method == "both":
            instances = self._merge_fragments_endpoint(instances)
            instances = self._merge_fragments_rotated_box(instances, image)
        else:
            instances = self._merge_fragments_rotated_box(instances, image)

        # 重新编号
        for i, inst in enumerate(instances):
            inst.instance_id = i

        type_counts: dict[str, int] = {}
        for inst in instances:
            t = inst.defect_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"提取到 {len(instances)} 个缺陷: {type_counts}")

        return instances

    # ── 密度检测（CRASH）──────────────────────────────────────

    def _detect_crashes(
        self,
        binary_uint8: np.ndarray,
        image: np.ndarray,
        optical_center: tuple[int, int],
        lens_radius: float,
    ) -> tuple[list[DefectInstance], np.ndarray]:
        """检测密集缺陷区域（CRASH 类型）.

        对二值掩码做高斯模糊估算局部缺陷密度，
        密度超过阈值的连通域标记为 CRASH。

        Returns:
            (crash_instances, residual_binary) — residual 已去除 crash 区域。
        """
        h, w = binary_uint8.shape
        k = max(3, int(min(h, w) * self.det_params.density_kernel_ratio))
        if k % 2 == 0:
            k += 1

        density = cv2.GaussianBlur(binary_uint8, (k, k), 0)
        thresh_val = self.det_params.density_threshold * 255.0
        _, dense_mask = cv2.threshold(density, thresh_val, 255, cv2.THRESH_BINARY)
        dense_mask = dense_mask.astype(np.uint8)

        # 形态学开运算清理碎片
        ck = 5
        dense_mask = cv2.morphologyEx(
            dense_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)),
        )

        crash_instances: list[DefectInstance] = []

        if np.any(dense_mask):
            num, labels, stats, _ = cv2.connectedComponentsWithStats(dense_mask, connectivity=8)
            dense_filtered = np.zeros_like(dense_mask)

            for idx in range(1, num):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < self.det_params.dense_min_area:
                    continue

                mask_i = (labels == idx).astype(bool)
                dense_filtered[mask_i] = 255

                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                bw = int(stats[idx, cv2.CC_STAT_WIDTH])
                bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
                bbox = (x, y, bw, bh)

                scatter = float(np.mean(image[mask_i])) if np.any(mask_i) else 0.0
                prominence = self._compute_prominence(bbox, image)

                coords_for_zone = np.argwhere(mask_i)
                zone = self._classify_zone(
                    coords_for_zone, optical_center, lens_radius,
                    self.scoring_params.zone_center_ratio,
                    self.scoring_params.zone_transition_ratio,
                )

                inst = DefectInstance(
                    instance_id=len(crash_instances),
                    defect_type=DefectType.CRASH,
                    skeleton_coords=np.empty((0, 2)),
                    mask=mask_i,
                    length_px=0.0,
                    area_px=area,
                    avg_width_px=0.0,
                    bbox=bbox,
                    scatter_intensity=scatter,
                    prominence=prominence,
                    zone=zone,
                    endpoints=None,
                    circularity=0.0,
                    aspect_ratio=0.0,
                )
                crash_instances.append(inst)

            dense_mask = dense_filtered

        if len(crash_instances) > 0:
            logger.info(f"密度检测: 发现 {len(crash_instances)} 个密集缺陷区")

        # 从原始掩码减去密集区
        residual = cv2.subtract(binary_uint8, dense_mask)
        residual_bool = residual > 0

        return crash_instances, residual_bool

    # ── Prominence ────────────────────────────────────────────

    @staticmethod
    def _compute_prominence(
        bbox: tuple[int, int, int, int],
        image: np.ndarray,
    ) -> float:
        """计算缺陷的显著性（中心与角落的灰度差）.

        暗场图像中缺陷为亮区，prominence = center_mean - corner_mean。
        """
        x, y, w_box, h_box = bbox
        if w_box <= 0 or h_box <= 0:
            return 0.0

        h_img, w_img = image.shape[:2]

        def _sample(px: float, py: float) -> float:
            px_i = int(min(max(round(px), 0), w_img - 1))
            py_i = int(min(max(round(py), 0), h_img - 1))
            return float(image[py_i, px_i])

        cx = x + w_box / 2.0
        cy = y + h_box / 2.0
        center_vals = [
            _sample(cx, cy),
            _sample(cx + 1, cy),
            _sample(cx - 1, cy),
            _sample(cx, cy + 1),
            _sample(cx, cy - 1),
        ]
        center_mean = sum(center_vals) / len(center_vals)

        corners = [
            _sample(x, y),
            _sample(x + w_box - 1, y),
            _sample(x, y + h_box - 1),
            _sample(x + w_box - 1, y + h_box - 1),
        ]
        corner_mean = sum(corners) / len(corners)

        return center_mean - corner_mean

    # ── 分类 ──────────────────────────────────────────────────

    @staticmethod
    def _classify_defect_type(
        area: int,
        aspect_ratio: float,
        circularity: float,
        length: float,
    ) -> DefectType:
        """根据形态学特征进行缺陷分类.

        分类规则:
        - damage: 面积 > DAMAGE_AREA_THRESHOLD
        - scratch: 细长比 > SPOT_ASPECT_RATIO 且 长度 > 最小值
        - spot: 其余 (近圆形小区域)
        """
        if area >= DAMAGE_AREA_THRESHOLD:
            return DefectType.DAMAGE
        if aspect_ratio >= SPOT_ASPECT_RATIO and length >= 10:
            return DefectType.SCRATCH
        return DefectType.SPOT

    # ── 形态学辅助 ────────────────────────────────────────────

    @staticmethod
    def _compute_circularity(mask: np.ndarray) -> float:
        """计算分量的圆度: 4π·area / perimeter²."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        if perimeter < 1e-6:
            return 0.0
        area = cv2.contourArea(largest)
        return float(4 * np.pi * area / (perimeter * perimeter))

    def _thin_fallback(self, binary: np.ndarray) -> np.ndarray:
        """骨架化回退方案（不依赖 cv2.ximgproc）."""
        from skimage.morphology import skeletonize
        skel = skeletonize(binary > 0)
        return (skel.astype(np.uint8) * 255)

    def _find_endpoints(
        self, skel_coords: np.ndarray
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """找骨架端点：使用坐标的最远两点作为端点近似."""
        if len(skel_coords) < 2:
            return None
        dists = np.linalg.norm(skel_coords - skel_coords[0], axis=1)
        far_idx = int(np.argmax(dists))
        p1 = tuple(skel_coords[0].tolist())
        p2 = tuple(skel_coords[far_idx].tolist())
        return (p1, p2)  # type: ignore[return-value]

    def _classify_zone(
        self,
        coords: np.ndarray,
        center: tuple[int, int],
        radius: float,
        center_ratio: float = 0.30,
        transition_ratio: float = 0.60,
    ) -> str:
        """判断缺陷所属视区."""
        cy, cx = center
        dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
        mean_dist = float(np.mean(dists))

        r_center = radius * center_ratio
        r_transition = radius * transition_ratio

        if mean_dist <= r_center:
            return "center"
        elif mean_dist <= r_transition:
            return "microstructure"
        else:
            return "edge"

    # ── 划痕合并：端点距离+方向一致性 ────────────────────────

    def _merge_fragments_endpoint(
        self,
        instances: list[DefectInstance],
    ) -> list[DefectInstance]:
        """基于端点距离和方向一致性的划痕合并."""
        scratches = [i for i in instances if i.defect_type == DefectType.SCRATCH]
        others = [i for i in instances if i.defect_type != DefectType.SCRATCH]

        if len(scratches) < 2:
            return scratches + others

        max_gap = self.det_params.merge_max_gap
        max_angle = self.det_params.merge_max_angle

        merged = list(scratches)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(merged):
                j = i + 1
                while j < len(merged):
                    if self._should_merge_endpoint(merged[i], merged[j], max_gap, max_angle):
                        merged[i] = self._do_merge(merged[i], merged[j])
                        merged.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

        return merged + others

    @staticmethod
    def _should_merge_endpoint(
        a: DefectInstance,
        b: DefectInstance,
        max_gap: float,
        max_angle: float,
    ) -> bool:
        """判断两个划痕是否应基于端点合并."""
        if a.endpoints is None or b.endpoints is None:
            return False

        min_dist = float('inf')
        for pa in a.endpoints:
            for pb in b.endpoints:
                d = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                min_dist = min(min_dist, d)

        if min_dist > max_gap:
            return False

        dir_a = np.array(a.endpoints[1]) - np.array(a.endpoints[0])
        dir_b = np.array(b.endpoints[1]) - np.array(b.endpoints[0])
        norm_a = np.linalg.norm(dir_a)
        norm_b = np.linalg.norm(dir_b)

        if norm_a < 1 or norm_b < 1:
            return True

        cos_angle = abs(np.dot(dir_a, dir_b) / (norm_a * norm_b))
        cos_angle = min(cos_angle, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        return angle_deg <= max_angle

    # ── 划痕合并：旋转包围盒延伸 ─────────────────────────────

    @staticmethod
    def _rect_to_box(rect: tuple) -> np.ndarray:
        """minAreaRect → 4×2 float32 顶点."""
        return cv2.boxPoints(rect).astype(np.float32)

    @staticmethod
    def _extend_rect(rect: tuple, extend_len: int, extend_width: int) -> tuple:
        """沿长轴延伸 minAreaRect."""
        cx, cy = rect[0]
        wi, hi = rect[1]
        angle = rect[2]
        if wi >= hi:
            new_w = max(1, int(wi + 2 * extend_len))
            new_h = max(1, int(hi + 2 * extend_width))
        else:
            new_w = max(1, int(wi + 2 * extend_width))
            new_h = max(1, int(hi + 2 * extend_len))
        return ((cx, cy), (new_w, new_h), angle)

    @staticmethod
    def _poly_overlap(box_a: np.ndarray, box_b: np.ndarray) -> bool:
        """判断两个凸多边形是否有重叠面积（使用 SAT 轴分离定理的简化版: intersectConvexConvex）."""
        ret, _ = cv2.intersectConvexConvex(box_a, box_b)
        return ret > 0.0

    def _merge_fragments_rotated_box(
        self,
        instances: list[DefectInstance],
        image: np.ndarray,
    ) -> list[DefectInstance]:
        """基于旋转包围盒延伸搜索的划痕合并.

        优化：使用凸多边形交集检测（intersectConvexConvex）替代全图掩码操作，
        避免 O(n²) 的大矩阵内存分配，大幅降低内存占用和运行时间。
        合并条件：延伸矩形多边形重叠 + 散射强度差值在容差内。
        """
        scratches = [i for i in instances if i.defect_type == DefectType.SCRATCH]
        others = [i for i in instances if i.defect_type != DefectType.SCRATCH]

        if len(scratches) < 2:
            return scratches + others

        gray_tol = self.det_params.scratch_merge_gray_tol

        # 预计算每个划痕的 minAreaRect 和 bbox 多边形
        rects: list[tuple] = []
        boxes: list[np.ndarray] = []  # 原始矩形的 4 顶点（用于 j 的相交检测）
        for s in scratches:
            contours, _ = cv2.findContours(
                s.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
            else:
                rect = ((0.0, 0.0), (1.0, 1.0), 0.0)
            rects.append(rect)
            boxes.append(self._rect_to_box(rect))

        removed = [False] * len(scratches)

        for i in range(len(scratches)):
            if removed[i]:
                continue

            rect_i = rects[i]
            wi, hi = rect_i[1]
            if wi <= 0 or hi <= 0:
                continue

            long_side = max(wi, hi)
            short_side = min(wi, hi)

            extend_len = max(
                int(long_side * self.det_params.scratch_extend_ratio),
                self.det_params.scratch_extend_min_px,
            )
            extend_width = max(
                int(short_side * self.det_params.scratch_extend_width_ratio),
                self.det_params.scratch_extend_min_width,
            )

            # 延伸搜索多边形（不分配全图矩阵）
            ext_rect = self._extend_rect(rect_i, extend_len, extend_width)
            search_box = self._rect_to_box(ext_rect)

            for j in range(i + 1, len(scratches)):
                if removed[j]:
                    continue

                # 散射强度容差（快速预筛）
                if abs(scratches[j].scatter_intensity - scratches[i].scatter_intensity) > gray_tol:
                    continue

                # 多边形交集检测（无全图内存分配）
                if not self._poly_overlap(search_box, boxes[j]):
                    continue

                # 执行合并
                scratches[i] = self._do_merge(scratches[i], scratches[j])
                removed[j] = True

                # 更新旋转矩形和多边形
                contours, _ = cv2.findContours(
                    scratches[i].mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    cnt_all = np.vstack(contours)
                    rects[i] = cv2.minAreaRect(cnt_all)
                    boxes[i] = self._rect_to_box(rects[i])
                    # 重新计算搜索多边形（合并后实例更长）
                    wi2, hi2 = rects[i][1]
                    ls2, ss2 = max(wi2, hi2), min(wi2, hi2)
                    el2 = max(int(ls2 * self.det_params.scratch_extend_ratio),
                              self.det_params.scratch_extend_min_px)
                    ew2 = max(int(ss2 * self.det_params.scratch_extend_width_ratio),
                              self.det_params.scratch_extend_min_width)
                    ext_rect = self._extend_rect(rects[i], el2, ew2)
                    search_box = self._rect_to_box(ext_rect)

        merged = [s for s, r in zip(scratches, removed) if not r]
        return merged + others

    # ── 合并执行 ──────────────────────────────────────────────

    def _do_merge(
        self, a: DefectInstance, b: DefectInstance
    ) -> DefectInstance:
        """合并两个划痕实例."""
        combined_mask = a.mask | b.mask
        combined_coords = np.vstack([a.skeleton_coords, b.skeleton_coords]) \
            if len(a.skeleton_coords) > 0 and len(b.skeleton_coords) > 0 \
            else (a.skeleton_coords if len(a.skeleton_coords) > 0 else b.skeleton_coords)
        combined_area = int(np.count_nonzero(combined_mask))
        combined_length = a.length_px + b.length_px

        return DefectInstance(
            instance_id=a.instance_id,
            defect_type=DefectType.SCRATCH,
            skeleton_coords=combined_coords,
            mask=combined_mask,
            length_px=combined_length,
            area_px=combined_area,
            avg_width_px=combined_area / max(combined_length, 1.0),
            bbox=self._merge_bbox(a.bbox, b.bbox),
            scatter_intensity=(a.scatter_intensity + b.scatter_intensity) / 2,
            prominence=(a.prominence + b.prominence) / 2,
            zone=a.zone,
            endpoints=self._find_endpoints(combined_coords),
            circularity=0.0,
            aspect_ratio=combined_length / max(combined_area / max(combined_length, 1.0), 1.0),
        )

    @staticmethod
    def _merge_bbox(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x1 = min(a[0], b[0])
        y1 = min(a[1], b[1])
        x2 = max(a[0] + a[2], b[0] + b[2])
        y2 = max(a[1] + a[3], b[1] + b[3])
        return (x1, y1, x2 - x1, y2 - y1)

    # 向后兼容
    def _merge_fragments(self, instances: list[DefectInstance]) -> list[DefectInstance]:
        return self._merge_fragments_endpoint(instances)


def _estimate_center_from_roi(roi_mask: np.ndarray) -> tuple[tuple[int, int], float]:
    """从 ROI 掩码推算等效圆心和半径."""
    if not np.any(roi_mask):
        h, w = roi_mask.shape[:2]
        return (h // 2, w // 2), 0.0

    contours, _ = cv2.findContours(
        roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        h, w = roi_mask.shape[:2]
        return (h // 2, w // 2), 0.0

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest)
    return (int(cy), int(cx)), float(r)
