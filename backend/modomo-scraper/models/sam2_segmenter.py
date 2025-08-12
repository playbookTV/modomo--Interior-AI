
"""
SAM2 + FBA Matting Segmenter for Modomo
---------------------------------------
- Pure SAM 2 for mask generation from detector boxes
- FBA Matting for high-quality alpha cutouts (trimap-based)
- Outputs mask PNG and RGBA cutout PNG, returns file paths and arrays

Requirements (install from your environment):
  # SAM 2
  - git clone https://github.com/facebookresearch/segment-anything-2
  - pip install -e segment-anything-2  (or add repo to PYTHONPATH)
  # FBA Matting
  - pip install opencv-python pillow numpy torch torchvision
  - Install FBA Matting codebase and place checkpoint .pth somewhere accessible
    (https://github.com/MarcoForte/FBA_Matting)

Expected Checkpoints:
  - SAM 2 checkpoint (e.g., sam2_hiera_large.pt)
  - FBA Matting checkpoint (e.g., fba_matting.pth)

Usage:
  segmenter = SAM2FBA_Segmenter(
      sam2_checkpoint="checkpoints/sam2_hiera_large.pt",
      fba_checkpoint="checkpoints/fba_matting.pth",
      device="cuda"
  )
  result = segmenter.process(
      image_path="/path/to/image.jpg",
      bbox=[x1, y1, x2, y2],
      out_dir="/tmp/segments",
      obj_id="img123_obj0",
      class_hint="sofa"  # optional, improves SAM2 in ambiguous cases
  )
  print(result["mask_path"], result["rgba_path"], result["alpha_path"])
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os
import io

import numpy as np
from PIL import Image

# OpenCV for morphology and trimap
import cv2

import torch

# --- SAM2 imports (from official repo) ---
# Ensure the SAM2 repo is importable (PYTHONPATH) or installed in your env
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    SAM2_AVAILABLE = False
    _SAM2_IMPORT_ERROR = e
else:
    SAM2_AVAILABLE = True
    _SAM2_IMPORT_ERROR = None

# --- FBA Matting imports ---
# We assume you have the FBA library available; adapt paths as needed.
try:
    # Typical FBA Matting import pattern; adjust to your local installation.
    # from fba_matting.networks.models import build_model as build_fba_model
    # For generality (and to avoid import errors here), we lazy-load in class.
    FBA_AVAILABLE = True
except Exception as e:
    FBA_AVAILABLE = False

@dataclass
class SAM2FBAConfig:
    sam2_checkpoint: str
    fba_checkpoint: str
    device: str = "cpu"  # "cuda" or "cpu"
    # Matting/trimap parameters
    trimap_erosion: int = 7    # kernel radius for fg/bg erosion
    unknown_width: int = 16    # width of unknown band
    # Post-processing
    min_mask_area: int = 500   # drop masks smaller than this area (px)
    # Saving
    png_compress_level: int = 6

class SAM2FBA_Segmenter:
    def __init__(self, sam2_checkpoint: str, fba_checkpoint: str, device: str = "cuda",
                 trimap_erosion: int = 7, unknown_width: int = 16, min_mask_area: int = 500,
                 png_compress_level: int = 6):
        self.cfg = SAM2FBAConfig(
            sam2_checkpoint=sam2_checkpoint,
            fba_checkpoint=fba_checkpoint,
            device=device,
            trimap_erosion=trimap_erosion,
            unknown_width=unknown_width,
            min_mask_area=min_mask_area,
            png_compress_level=png_compress_level
        )

        if not SAM2_AVAILABLE:
            raise ImportError(f"SAM2 not available: {_SAM2_IMPORT_ERROR}")

        self.device = torch.device(self.cfg.device if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu")

        # Build & load SAM2 predictor
        self.sam2_model = build_sam2(self.cfg.sam2_checkpoint).to(self.device)
        self.sam2_model.eval()
        self.sam2 = SAM2ImagePredictor(self.sam2_model)

        # FBA matting model lazy-loaded on first use (to allow CPU-only environments)
        self.fba_model = None

    # ---------- public API ----------
    def process(self, image_path: str, bbox: List[int], out_dir: str, obj_id: str,
                class_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        image_path: path to RGB image
        bbox: [x1, y1, x2, y2] in pixel coords
        out_dir: directory to save outputs
        obj_id: unique identifier for naming outputs (e.g., imageID_objN)
        class_hint: optional textual hint ("sofa", "pendant light")

        Returns dict paths & arrays.
        """
        os.makedirs(out_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        np_img = np.array(image)

        # 1) Predict mask with SAM2 using bbox (and optional text hint if supported)
        mask = self._sam2_mask(np_img, bbox, class_hint=class_hint)
        if mask is None or int(mask.sum()) < self.cfg.min_mask_area:
            return {
                "ok": False,
                "reason": "mask_too_small_or_none",
                "mask_path": None,
                "rgba_path": None,
                "alpha_path": None
            }

        # 2) Save binary mask
        mask_path = os.path.join(out_dir, f"{obj_id}_mask.png")
        self._save_mask_png(mask.astype(np.uint8)*255, mask_path)

        # 3) Build trimap & run FBA Matting for high-quality alpha
        trimap = self._make_trimap(mask.astype(np.uint8))
        alpha = self._fba_alpha(np_img, trimap)

        # 4) Compose RGBA cutout with neutral background (keep alpha as is)
        rgba = self._compose_rgba(np_img, alpha)
        rgba_path = os.path.join(out_dir, f"{obj_id}_cutout.png")
        Image.fromarray(rgba, mode="RGBA").save(rgba_path, compress_level=self.cfg.png_compress_level)

        # 5) Save alpha as grayscale PNG as well
        alpha_path = os.path.join(out_dir, f"{obj_id}_alpha.png")
        Image.fromarray((alpha*255).astype(np.uint8), mode="L").save(alpha_path, compress_level=self.cfg.png_compress_level)

        return {
            "ok": True,
            "mask_path": mask_path,
            "rgba_path": rgba_path,
            "alpha_path": alpha_path,
            "mask_area": int(mask.sum())
        }

    # ---------- SAM2 inference ----------
    def _sam2_mask(self, np_img: np.ndarray, bbox: List[int], class_hint: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Runs SAM2 on the given image with a box prompt.
        Returns a boolean mask HxW or None.
        """
        # SAM2 predictor works with numpy RGB images
        self.sam2.set_image(np_img)

        # SAM2 expects box as [x1, y1, x2, y2]
        box = np.array(bbox, dtype=np.float32)

        # If the installed SAM2 supports text prompts, you could pass that here;
        # base predictor uses only box/points. We keep class_hint for future logic.
        masks, scores, logits = self.sam2.predict(
            box=box,
            multimask_output=False  # single best mask
        )
        if masks is None or len(masks) == 0:
            return None
        mask = masks[0].astype(bool)
        # Basic cleanup: fill small holes, remove small islands
        mask = self._postprocess_mask(mask)
        return mask

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_uint = (mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # Close small gaps
        closed = cv2.morphologyEx(mask_uint, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove tiny specks
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((closed>0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(labels, dtype=np.uint8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.cfg.min_mask_area:
                cleaned[labels==label] = 1
        return cleaned.astype(bool)

    # ---------- Trimap & Matting ----------
    def _make_trimap(self, mask: np.ndarray) -> np.ndarray:
        """
        Build a trimap from a binary mask (uint8 0/1).
        0=bg, 128=unknown, 255=fg
        """
        k = self.cfg.trimap_erosion
        kernel = np.ones((k, k), np.uint8)

        fg = cv2.erode(mask, kernel, iterations=1)
        bg = cv2.erode(1-mask, kernel, iterations=1)

        trimap = np.full(mask.shape, 128, dtype=np.uint8)
        trimap[fg == 1] = 255
        trimap[bg == 1] = 0

        # Widen the unknown band to help matting around edges
        if self.cfg.unknown_width > 0:
            dilated_fg = cv2.dilate(fg, np.ones((self.cfg.unknown_width, self.cfg.unknown_width), np.uint8), iterations=1)
            eroded_fg = cv2.erode(fg, np.ones((self.cfg.unknown_width, self.cfg.unknown_width), np.uint8), iterations=1)
            edge_band = (dilated_fg - eroded_fg) > 0
            trimap[edge_band] = 128

        return trimap

    def _lazy_load_fba(self):
        if self.fba_model is not None:
            return
        # Lazy import here to avoid import errors when just inspecting the module
        from importlib import import_module
        # Typical FBA entrypoint (adjust to your local installation)
        # Example assumes something like: from fba_matting.models import build_model
        try:
            fba_mod = import_module("fba_matting.models")
            build_fba_model = getattr(fba_mod, "build_model")
        except Exception as e:
            raise ImportError(f"Could not import FBA Matting build_model: {e}")

        self.fba_model = build_fba_model()
        ckpt = torch.load(self.cfg.fba_checkpoint, map_location=self.device)
        # Adapt the key names if needed depending on the checkpoint
        if "state_dict" in ckpt:
            self.fba_model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            self.fba_model.load_state_dict(ckpt, strict=False)
        self.fba_model.to(self.device).eval()

    def _fba_alpha(self, np_img: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """
        Run FBA Matting to predict alpha in [0,1] float32.
        """
        self._lazy_load_fba()

        img = Image.fromarray(np_img).convert("RGB")
        # FBA expects normalized tensors; adapt preprocessing to your model impl.
        # Below is a generic example; align with your FBA code's expected transforms.
        im = np.asarray(img).astype(np.float32) / 255.0
        tri = (trimap.astype(np.float32) / 255.0)

        # Convert to CHW tensors
        im_t = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).to(self.device)
        tri_t = torch.from_numpy(tri).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Typical FBA forward signature: pred_alpha = model(im_t, trimap=tri_t)
            # Adjust per your implementation.
            pred_alpha = self.fba_model(im_t, trimap=tri_t)
            # Ensure shape [1,1,H,W]
            if isinstance(pred_alpha, (list, tuple)):
                pred_alpha = pred_alpha[0]
            alpha = pred_alpha.squeeze().clamp(0,1).detach().cpu().numpy().astype(np.float32)
        return alpha

    # ---------- IO helpers ----------
    def _save_mask_png(self, mask_uint8: np.ndarray, path: str):
        Image.fromarray(mask_uint8, mode="L").save(path, compress_level=self.cfg.png_compress_level)

    def _compose_rgba(self, np_img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compose RGBA with original RGB and predicted alpha (kept as-is)."""
        h, w, _ = np_img.shape
        a = (alpha * 255).astype(np.uint8)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:,:,:3] = np_img
        rgba[:,:,3] = a
        return rgba
