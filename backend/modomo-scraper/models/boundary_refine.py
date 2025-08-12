# boundary_refine.py
import numpy as np

def needs_hq(mask: np.ndarray) -> bool:
    # Heuristic: thin, long shapes → many boundary pixels relative to area
    area = mask.sum()
    if area < 500: return False
    # perimeter via binary erosion trick
    from scipy.ndimage import binary_erosion
    edge = mask ^ binary_erosion(mask)
    perim = edge.sum()
    return (perim / max(1, area)) > 0.15  # tweak per dataset

def refine_with_hq_sam(image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Placeholder: call HQ‑SAM module if available to refine boundaries.
    If not available, fall back to edge‑snapping w/ Canny + GrabCut warm start.
    """
    try:
        from hq_sam import HQRefiner  # example import; adjust to your package
        refiner = HQRefiner()
        return refiner.refine(image_np, mask)
    except Exception:
        import cv2
        # Fallback: slight blur → Canny → snap contours
        m = (mask*255).astype(np.uint8)
        edges = cv2.Canny((image_np).astype(np.uint8), 50, 150)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)
        # Optional quick GrabCut pass
        bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
        trimap = np.full(m.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[m>0] = cv2.GC_FGD
        cv2.grabCut(image_np, trimap, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
        refined = ( (trimap==cv2.GC_FGD) | (trimap==cv2.GC_PR_FGD) ).astype(np.uint8)
        return refined.astype(bool)