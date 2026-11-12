# meutipo — clone client-only do mytype.fyi

Descubra seu tipo avaliando rostos gerados por IA. Toque nos que você curte em
"multidões" de 12 rostos; depois de ~20 multidões o app mostra seu tipo em rostos
e em palavras. **Sem email, sem conta, sem servidor** — tudo roda no navegador
(vanilla JS) e seus toques ficam no `localStorage`.

## Rodando local

```
python -m http.server 8000
# abra http://localhost:8000
```

## Deploy no GitHub Pages

1. Suba este repo para o GitHub (`faces/`, `faces.json`, `embed.bin` inclusos).
2. Em **Settings → Pages**, escolha a branch (`main`) e a pasta raiz (`/`).
3. Pronto — é tudo estático (`.nojekyll` já incluso).

## Pipeline de imagens (`tools/fetch_faces.py`)

Baixa rostos do [thispersondoesnotexist.com](https://thispersondoesnotexist.com)
(StyleGAN — não são pessoas reais) e rotula cada um com
[InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`, CPU):

```
pip install insightface onnxruntime opencv-python-headless pillow numpy
python tools/fetch_faces.py --target 600     # resumivel; max 2048
python tools/fetch_faces.py --finalize       # so regrava faces.json + embed.bin
```

Cada rosto recebe: gênero (0=f/1=m), idade estimada (**mínimo 24 — segurança**),
traços (faixa etária, cor do cabelo, tom de pele, cor dos olhos, sorriso),
um score de "popularidade" **simulado** (não há multidão real num site estático)
e um embedding 512d reduzido a 48d via PCA → `embed.bin` (int8).

## Como o modelo funciona (client-side)

- Prior = score de popularidade simulado.
- Personal = 55% taxa de like por traço (Laplace) + 45% similaridade de cosseno
  entre o embedding do rosto e os centroides de curtidos/não-curtidos.
- `p = (1-conf)·prior + conf·(0.72·personal + 0.28·prior)`, `conf = taps/(taps+25)`.
- Multidão: ~8 exploração do topo previsto, ~2 populares de calibração,
  ~2 aleatórios. Toque substitui a face na hora; deslizar pra cima avança.
- Precisão exibida = AUC estimado entre curtidos e passados.
