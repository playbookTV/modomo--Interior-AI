# fusion.py
from typing import List, Dict, Tuple
import numpy as np

BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def soft_nms(dets: List[Dict], iou_thr=0.5, sigma=0.5, score_thr=0.05) -> List[Dict]:
    # det = {"bbox":(x1,y1,x2,y2), "score":float, "cls":str, "src":"rtdetr|gdino"}
    dets = dets.copy()
    out = []
    while dets:
        m = max(range(len(dets)), key=lambda i: dets[i]["score"])
        best = dets.pop(m)
        out.append(best)
        keep = []
        for d in dets:
            if d["cls"] != best["cls"]:
                keep.append(d); continue
            o = iou(best["bbox"], d["bbox"])
            if o > iou_thr:
                d = d.copy()
                d["score"] *= np.exp(- (o*o) / sigma)
            if d["score"] >= score_thr:
                keep.append(d)
        dets = keep
    return out

def map_gdino_label_to_taxonomy(text: str) -> str:
    t = text.lower()
    synonyms = {
        "sofa":["sofa","couch","settee","sectional"],
        "armchair":["armchair","accent chair","lounge chair"],
        "dining chair":["dining chair","side chair"],
        "coffee table":["coffee table","centre table"],
        "side table":["end table","side table","nightstand"],
        "pendant light":["pendant","ceiling light","hanging light","chandelier"],
        "floor lamp":["floor lamp","standing lamp"],
        "table lamp":["table lamp","desk lamp","bedside lamp"],
        "rug":["rug","carpet"],
        "curtains":["curtains","drapes"],
        "plant":["plant","potted plant","indoor plant"],
        "mirror":["mirror","wall mirror"],
        "wall art":["art","painting","poster","frame"],
        "bed frame":["bed","bed frame","headboard"]
    }
    for k, vs in synonyms.items():
        if any(v in t for v in vs):
            return k
    return t  # fallback: raw

def fuse(rtdetr: List[Dict], gdino: List[Dict], min_area: int, w: int, h: int) -> List[Dict]:
    # basic score calibration and area filtering
    dets = []
    for d in rtdetr:
        x1,y1,x2,y2 = d["bbox"]; area = (x2-x1)*(y2-y1)
        if area >= min_area:
            dets.append({"bbox":d["bbox"], "score":d["score"], "cls":d["cls"], "src":"rtdetr"})
    for g in gdino:
        x1,y1,x2,y2 = g["bbox"]; area = (x2-x1)*(y2-y1)
        if area < min_area: continue
        cls = map_gdino_label_to_taxonomy(g.get("text",""))
        score = 0.85 * g["score"]  # calibrate slightly lower than RT‑DETR
        dets.append({"bbox":g["bbox"], "score":score, "cls":cls, "src":"gdino"})

    # keep overlapping duplicates within class via soft-NMS;
    # allow near-duplicates across different classes (possible multi-label scenes)
    fused = soft_nms(dets, iou_thr=0.55, sigma=0.5, score_thr=0.12)

    # small sanity: drop extreme aspect-ratio noise
    final = []
    for d in fused:
        x1,y1,x2,y2 = d["bbox"]; wbox=x2-x1; hbox=y2-y1
        ar = wbox / max(1e-6, hbox)
        if 0.2 <= ar <= 5.0:
            final.append(d)
    return final