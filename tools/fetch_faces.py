#!/usr/bin/env python3
"""Baixa rostos gerados por IA e rotula cada um (genero, idade, tracos, embedding).

Uso:
    python tools/fetch_faces.py --target 600          # rostos validos desejados
    python tools/fetch_faces.py --target 2048 --size 384

Resumivel: reler faces.json e continua de onde parou.
Saidas:
    faces/<id>.webp        imagens 384x384 (ou --size)
    faces.json             manifesto: [{f,g,age,traits,pop}, ...]
    embed.bin              int8[N*EMBED_DIM] embedding reduzido (PCA)
"""

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_URL = "https://thispersondoesnotexist.com/random-person.jpeg"
EMBED_DIM = 48
MIN_FACE_PX = 200          # bbox minima para aceitar o rosto
MIN_DET_SCORE = 0.80
MIN_AGE = 24               # margem de seguranca: rejeita estimativas baixas
AGES_8 = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URLS = {
    "age_googlenet.onnx": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/age_gender/models/age_googlenet.onnx",
    "gender_googlenet.onnx": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/age_gender/models/gender_googlenet.onnx",
    "vgg_age.onnx": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx",
}


def log(*a):
    print(*a, flush=True)


def fetch_image(timeout=30):
    import urllib.request
    req = urllib.request.Request(IMG_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def hexhash(b):
    return hashlib.md5(b).hexdigest()


# ---------------------------------------------------------------- traits ----

def _lum(px):
    return 0.114 * px[..., 0] + 0.587 * px[..., 1] + 0.299 * px[..., 2]


def hair_color(img_bgr, bbox, skin_bgr):
    """Amostra um anel ao redor da metade superior do rosto e classifica."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    fw, fh = x2 - x1, y2 - y1
    strips = [
        (y1 - 0.30 * fh, y1 + 0.02 * fh, x1 + 0.10 * fw, x2 - 0.10 * fw),  # topo
        (y1, y1 + 0.55 * fh, x1 - 0.22 * fw, x1),                         # esquerda
        (y1, y1 + 0.55 * fh, x2, x2 + 0.22 * fw),                         # direita
    ]
    px = []
    for a, b, c, d in strips:
        r = img_bgr[int(max(0, a)):int(min(h, b)), int(max(0, c)):int(min(w, d))]
        if r.size:
            px.append(r.reshape(-1, 3))
    if not px:
        return None
    px = np.concatenate(px).astype(np.float32)
    # remove pixels claros dessaturados (fundo) e tom de pele (careca)
    sat = px.max(1) - px.min(1)
    lum = _lum(px)
    keep = (lum < 100) | (sat > 18)
    if keep.sum() < 60:
        return "careca" if skin_bgr is not None else None
    px = px[keep]
    # remove pixels proximos do tom de pele -> cabelo raspado/careca
    if skin_bgr is not None:
        d = np.linalg.norm(px - np.asarray(skin_bgr), axis=1)
        px = px[d > 55]
        if len(px) < 60:
            return "careca"
    mean = px.mean(0)
    lum_m = float(_lum(mean))
    b_, g_, r_ = mean
    if lum_m < 55:
        return "preto"
    if lum_m > 190 and (mean.max() - mean.min()) < 45:
        return "grisalho"
    if r_ - g_ > 38 and r_ - b_ > 55:
        return "ruivo"
    if lum_m > 140 and r_ > g_ + 8 and r_ - b_ > 22:
        return "loiro"
    return "castanho"


def skin_tone(img_bgr, bbox):
    """Amostra as bochechas -> (rotulo, cor media BGR)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    patches = [
        img_bgr[y1 + int(h * .55):y1 + int(h * .75), x1 + int(w * .15):x1 + int(w * .35)],
        img_bgr[y1 + int(h * .55):y1 + int(h * .75), x1 + int(w * .65):x1 + int(w * .85)],
    ]
    px = np.concatenate([p.reshape(-1, 3) for p in patches if p.size])
    if len(px) == 0:
        return None, None
    mean = px.mean(axis=0)
    lum = _lum(mean)
    label = "clara" if lum > 185 else ("media" if lum > 120 else "escura")
    return label, mean


def eye_color(img_bgr, bbox):
    """Amostra regiao das iris -> claros/escuros."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px = []
    for xa, xb in ((0.22, 0.42), (0.58, 0.78)):
        r = img_bgr[int(y1 + h * .38):int(y1 + h * .50),
                    int(x1 + w * xa):int(x1 + w * xb)]
        if r.size:
            px.append(r.reshape(-1, 3))
    if not px:
        return None
    px = np.concatenate(px).astype(np.float32)
    lum = _lum(px)
    # iris = pixels medios (nao branco dos olhos, nao delineador/pupila mto escura)
    keep = (lum > 45) & (lum < 160)
    if keep.sum() < 30:
        return None
    m = px[keep].mean(axis=0)
    b_, g_, r_ = m
    if b_ > r_ + 12 or g_ > r_ + 18:
        return "olhos claros"
    return "olhos escuros"


def smile_score(landmark, bbox):
    """Sorriso = cantos da boca erguidos. Pontos 52-71 = boca (esquema 106pt).
    Retorna 'lift': (y_medio_cantos - y_topo_boca) / altura_boca. Menor = sorriso."""
    if landmark is None or len(landmark) < 72:
        return 1.0
    mo = np.asarray(landmark)[52:72]
    h = mo[:, 1].max() - mo[:, 1].min()
    if h < 1e-6:
        return 1.0
    lc, rc = mo[mo[:, 0].argmin()], mo[mo[:, 0].argmax()]
    return float(((lc[1] + rc[1]) / 2 - mo[:, 1].min()) / h)


# ------------------------------------------------------- segunda opiniao ----

_onnx = {}

def load_verifiers():
    import onnxruntime as ort
    os.makedirs(MODELS_DIR, exist_ok=True)
    import urllib.request
    for name, url in MODEL_URLS.items():
        p = os.path.join(MODELS_DIR, name)
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            log(f"baixando {name}...")
            urllib.request.urlretrieve(url, p)
        _onnx[name] = ort.InferenceSession(p, providers=["CPUExecutionProvider"])


def _goog_blob(img_bgr):
    im = cv2.resize(img_bgr, (224, 224)).astype(np.float32)
    im -= np.array([104, 117, 123], np.float32)
    return im.transpose(2, 0, 1)[None]


def verify_face(img_bgr, insight_gender):
    """Segunda opiniao (Levi-Hassner googlenet).
    Retorna (ok, motivo): rejeita criancas/adolescentes e divergencia de genero."""
    blob = _goog_blob(img_bgr)
    age = _onnx["age_googlenet.onnx"]
    a = age.run(None, {age.get_inputs()[0].name: blob})[0][0]
    pa = np.exp(a - a.max()); pa /= pa.sum()
    if int(pa.argmax()) <= 2:                       # 0-2, 4-6, 8-12
        return False, f"googlenet idade {AGES_8[int(pa.argmax())]}"
    if pa[:3].sum() > 0.40:
        return False, f"googlenet P(crianca)={pa[:3].sum():.2f}"
    gen = _onnx["gender_googlenet.onnx"]
    g = gen.run(None, {gen.get_inputs()[0].name: blob})[0][0]
    goog_gender = int(g.argmax())                   # 0=M, 1=F
    if goog_gender != (1 - insight_gender):         # insightface: 0=F,1=M
        return False, "genero divergente"
    vgg = _onnx["vgg_age.onnx"]
    a = vgg.run(None, {vgg.get_inputs()[0].name: blob})[0][0]
    if int(a.argmax()) <= 18:                       # VGG argmax <=18 ~ menor/teen
        return False, f"vgg argmax {int(a.argmax())}"
    return True, ""


def age_group(age):
    if age < 26: return "jovem"
    if age < 38: return "adulto"
    if age < 55: return "maduro"
    return "senior"


# --------------------------------------------------------------- pipeline ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600, help="rostos validos (max 2048)")
    ap.add_argument("--size", type=int, default=384, help="lado do webp de saida")
    ap.add_argument("--quality", type=int, default=78)
    ap.add_argument("--max-fetch", type=int, default=0,
                    help="limite de downloads (0 = target*3)")
    ap.add_argument("--finalize", action="store_true",
                    help="nao baixa nada; so regrava faces.json e embed.bin")
    ap.add_argument("--verify", action="store_true",
                    help="re-verifica rostos existentes com a 2a opiniao e remove falhas")
    args = ap.parse_args()

    if args.finalize:
        manifest_path = os.path.join(ROOT, "faces.json")
        records = json.load(open(manifest_path, encoding="utf-8"))
        save_manifest(records, manifest_path)
        write_embeddings(records)
        log(f"finalizado: {len(records)} rostos")
        return

    if args.verify:
        verify_existing()
        return

    target = min(args.target, 2048)
    max_fetch = args.max_fetch or int(target * 3.2)

    faces_dir = os.path.join(ROOT, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    manifest_path = os.path.join(ROOT, "faces.json")

    records = []
    seen_hashes = set()
    if os.path.exists(manifest_path):
        records = json.load(open(manifest_path, encoding="utf-8"))
        for r in records:
            seen_hashes.add(r.get("md5", ""))
        log(f"retomando: {len(records)} rostos ja existentes")

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    load_verifiers()

    next_id = (max((r["id"] for r in records), default=-1) + 1)

    import queue, threading
    q = queue.Queue(maxsize=32)
    stop = threading.Event()
    counters = {"fetch": 0, "reject": 0}
    clock = threading.Lock()

    def process_one():
        """fetch + inferencia completa -> dict pronto p/ gravar, ou None."""
        try:
            raw = fetch_image()
        except Exception:
            return None
        h = hexhash(raw)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        try:
            det = app.get(img)
        except Exception:
            return None
        det = [f for f in det if f.det_score >= MIN_DET_SCORE]
        if len(det) != 1:
            return None
        f = det[0]
        x1, y1, x2, y2 = f.bbox
        if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
            return None
        gender = int(f.gender)          # insightface: 0=f, 1=m  (verificado)
        age = float(f.age)
        if age < MIN_AGE:
            return None                 # seguranca: nunca mostrar menores
        skin, skin_bgr = skin_tone(img, f.bbox)
        hair = hair_color(img, f.bbox, skin_bgr)
        eyes = eye_color(img, f.bbox)
        sm = smile_score(getattr(f, "landmark_2d_106", None), f.bbox)
        emb = f.normed_embedding.astype(np.float32) if f.normed_embedding is not None else None

        traits = [age_group(age)]
        if hair: traits.append(f"cabelo {hair}")
        if skin: traits.append(f"pele {skin}")
        if eyes: traits.append(eyes)
        traits.append("sorrindo" if sm < 0.35 else "serio")

        H, W = img.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * 1.55
        crop = img[int(max(0, cy - side / 2)):int(min(H, cy + side / 2)),
                   int(max(0, cx - side / 2)):int(min(W, cx + side / 2))]
        if crop.size == 0:
            return None
        ok, _why = verify_face(crop, gender)
        if not ok:
            with clock: counters["reject"] += 1
            return None
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
            (args.size, args.size), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, "WEBP", quality=args.quality)
        return {"md5": h, "webp": buf.getvalue(), "g": gender,
                "age": round(age, 1), "traits": traits,
                "smile": round(sm, 3), "emb": emb}

    def worker():
        while not stop.is_set():
            with clock:
                if counters["fetch"] >= max_fetch:
                    return
                counters["fetch"] += 1
            r = process_one()
            if r is not None:
                q.put(r)

    n_workers = min(6, (os.cpu_count() or 4))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads: t.start()
    log(f"{n_workers} workers buscando ate {target} rostos...")

    while len(records) < target and counters["fetch"] < max_fetch:
        try:
            r = q.get(timeout=2)
        except queue.Empty:
            if not any(t.is_alive() for t in threads):
                break
            continue
        if r["md5"] in seen_hashes:
            continue
        fid = next_id; next_id += 1
        fname = f"{fid:04d}.webp"
        with open(os.path.join(faces_dir, fname), "wb") as fp:
            fp.write(r["webp"])

        rng2 = random.Random(fid * 7919)
        pop = 0.5
        pop += {"jovem": .12, "adulto": .05, "maduro": -.08, "senior": -.16}[r["traits"][0]]
        pop += {"sorrindo": .10, "serio": -.04}.get(r["traits"][-1], 0)
        pop += {"clara": .04}.get(next((t[5:] for t in r["traits"] if t.startswith("pele ")), ""), 0)
        pop += rng2.uniform(-.18, .18)
        pop = round(min(max(pop, 0.02), 0.98), 4)

        rec = {"id": fid, "f": fname, "g": r["g"], "age": r["age"],
               "traits": r["traits"], "pop": pop, "smile": r["smile"], "md5": r["md5"]}
        if r["emb"] is not None:
            os.makedirs(os.path.join(ROOT, "tools", "embs"), exist_ok=True)
            np.save(os.path.join(ROOT, "tools", "embs", f"{fid:04d}.npy"), r["emb"])
        records.append(rec)
        seen_hashes.add(r["md5"])
        if len(records) % 20 == 0 or len(records) >= target:
            log(f"{len(records)}/{target} rostos ({counters['fetch']} fetches, {counters['reject']} rejeitados)")
            save_manifest(records, manifest_path)

    stop.set()
    for t in threads: t.join(timeout=3)
    save_manifest(records, manifest_path)
    write_embeddings(records)
    log(f"pronto: {len(records)} rostos, {counters['fetch']} downloads, {counters['reject']} rejeitados")


def verify_existing():
    """Re-roda a 2a opiniao em todos os rostos salvos; remove quem falha."""
    load_verifiers()
    manifest_path = os.path.join(ROOT, "faces.json")
    records = json.load(open(manifest_path, encoding="utf-8"))
    keep, drop = [], []
    for r in records:
        img = cv2.imread(os.path.join(ROOT, "faces", r["f"]))
        ok, why = (False, "arquivo ilegivel") if img is None \
            else verify_face(img, r["g"])
        if ok: keep.append(r)
        else: drop.append((r, why))
    for r, why in drop:
        log(f"removendo {r['f']} (idade {r['age']}): {why}")
        for p in (os.path.join(ROOT, "faces", r["f"]),
                  os.path.join(ROOT, "tools", "embs", f"{r['id']:04d}.npy")):
            if os.path.exists(p): os.remove(p)
    save_manifest(keep, manifest_path)
    write_embeddings(keep)
    log(f"verify: {len(keep)} mantidos, {len(drop)} removidos")


def save_manifest(records, path):
    out = []
    for r in records:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(rr)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))


def write_embeddings(records, dim=EMBED_DIM):
    """PCA dos embeddings 512d -> dim dims, quantizado int8 -> embed.bin"""
    emb_dir = os.path.join(ROOT, "tools", "embs")
    embs = []
    for r in records:
        p = os.path.join(emb_dir, f"{r['id']:04d}.npy")
        embs.append(np.load(p).astype(np.float32) if os.path.exists(p) else None)
    if any(e is None for e in embs):
        log("aviso: nem todos os rostos tem embedding; embed.bin nao escrito")
        # limpa campo interno e salva manifesto sem ele
        for r in records:
            r.pop("_emb", None)
        save_manifest(records, os.path.join(ROOT, "faces.json"))
        return
    X = np.asarray(embs, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    P = vt[:dim].T                       # 512 x dim
    Z = X @ P                            # N x dim
    scale = np.abs(Z).max(axis=0) + 1e-6
    Q = np.clip(np.round(Z / scale * 127), -127, 127).astype(np.int8)
    np.save(os.path.join(ROOT, "tools", "pca_basis.npy"),
            {"mean": X.mean(axis=0), "P": P, "scale": scale},
            allow_pickle=True)
    with open(os.path.join(ROOT, "embed.bin"), "wb") as fp:
        fp.write(np.array([len(records), dim], dtype=np.uint32).tobytes())
        fp.write(Q.tobytes())
    # manifesto final sem embeddings
    for r in records:
        r.pop("_emb", None)
    save_manifest(records, os.path.join(ROOT, "faces.json"))
    log(f"embed.bin: {len(records)}x{dim} int8")


if __name__ == "__main__":
    main()
