# Technical Report — ObjectLens AI
### Targeted Object Recognition System · MobileNetV2 + Streamlit
**Author:** Bashir Ahmad | B.Tech CSE, MNNIT Allahabad

---

## 1. The Role of the Softmax Activation Function

The final layer of MobileNetV2 (when loaded with `include_top=True`) is a **Dense layer with 1,000 neurons**, one for each ImageNet class. These neurons produce raw, unbounded numbers called **logits**. On their own, logits cannot be interpreted as probabilities — a logit of 3.7 vs 1.2 tells us the direction of preference but not *how confident* the model is.

**Softmax converts logits into a proper probability distribution:**

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}
$$

Three critical properties result from this:

| Property | What it means for this project |
|---|---|
| **All outputs ∈ (0, 1)** | Every class score can be read as a percentage confidence |
| **All 1,000 scores sum to 1.0** | The model is always "betting" its full probability mass across classes |
| **Exponential sharpening** | A small logit advantage is amplified, making the top prediction stand out clearly |

In our system, Softmax scores are what `decode_predictions()` returns. We search the **top-20 scores** — not just the top-1 — because our target objects (e.g. "charger") may not be the model's single highest prediction but still appear within its top-20 confident guesses.

---

## 2. Why `expand_dims` is Required Before `model.predict()`

`model.predict()` is designed to process **batches of images**, not a single image. The MobileNetV2 input layer expects a 4-D tensor of shape:

```
(batch_size, height, width, channels)
     ↑
     Must be present even for 1 image
```

When we load an image and convert it to a NumPy array, its shape is `(224, 224, 3)` — only 3 dimensions. Passing this directly to `model.predict()` raises a shape mismatch error.

`np.expand_dims(arr, axis=0)` inserts a new dimension at position 0:

```
(224, 224, 3)  →  (1, 224, 224, 3)
```

Now the model sees a batch of **one** image and processes it correctly. This is a fundamental requirement of all Keras/TensorFlow models — the batch axis is never optional, even when predicting on a single sample.

---

## 3. Out-of-Distribution (OOD) Handling

Standard classifiers always output *some* class — even for inputs they were never trained on. This is a known failure mode. Our filtering layer addresses it:

1. We check the **top-20 Softmax predictions** against a dictionary of target keywords.
2. If **no keyword match** is found in the top-20, the object is labelled **"UNKNOWN"** and the app displays a red warning.
3. This prevents the model from confidently mislabelling a water bottle as a "Mobile Phone" — a critical requirement for domain-specific deployment.

---

## 4. Preprocessing Pipeline Summary

```
Raw PIL Image
    │
    ▼
.convert("RGB")           # Ensures 3 channels; handles RGBA/grayscale
    │
    ▼
.resize((224, 224))       # Matches MobileNetV2 input spatial resolution
    │
    ▼
np.array(dtype=float32)   # Shape: (224, 224, 3), pixel values [0, 255]
    │
    ▼
np.expand_dims(axis=0)    # Shape: (1, 224, 224, 3) — adds batch dimension
    │
    ▼
preprocess_input()        # Scales pixels from [0, 255] → [-1, 1]
    │                       (MobileNetV2-specific normalization)
    ▼
model.predict()           # Softmax output: (1, 1000)
```

Normalization to `[-1, 1]` is critical: the ImageNet kernels learned their edge-detection and texture filters under this exact input distribution. Feeding un-normalized data shifts the activation statistics and degrades accuracy significantly.

---

*Report prepared as part of the Targeted Object Recognition System project.*
