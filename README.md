# intake-alphabetsoup

Provides an intake interface to synthetic cryoEM images from [VNE](https://github.com/quantumjot/vne).

Installation:

```
pip install git+https://github.com/alan-turing-institute/intake-alphabetsoup.git
```

Usage:

```python
import intake

cat = intake.open_synthetic_alphabetsoup(
  image_count = 1,
  shape=(512, 512),
  ctf_box_size=512,
  ctf_defocus=5000.0,
)

images, bounding_boxes, labels = cat.read()
```
