# Dev Log

Running notes from building this project.

---

## 2026 March - 2026 April

### Detection not working — objects not recognized

**Problem:** Pointed the camera at a water bottle, a phone, a person — model was either not detecting anything or giving very low confidence (20–30%).  
**First suspicion:** Thought it might be a code issue with how frames were being passed to the model.  
**Code was okay:** Checked the code but everthing seems to be okay. Verified the input shape and dtype matched what the model expected. Looked fine.
**The cause ! :** Lighting. The room looked fine to me but the camera was struggling — images were coming out noisy and underexposed.


## 2026 May

**Method of diagnosis:**
Ran a quick brightness check on a captured frame:
```python
import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()
frame = picam2.capture_array()
print(f"Avg brightness: {np.mean(frame):.1f}")   # was coming back around 45
print(f"Std dev: {np.std(frame):.1f}")
```
Brightness was ~45. Anything below ~80 and the camera is boosting gain, which adds noise.

**Aplied solutions...:**
1. Turned on the overhead light + a desk lamp → brightness went to ~110, detections improved immediately
2. Opened the window blinds (daytime) → best results, clean sharp images
3. Moved closer to the camera (was about 3m away) → also helped a lot

**Outcome:** With good lighting and standing ~1.5m from the camera, the model was detecting people at 80–90% confidence consistently. Same model, totally different results just from fixing the environment.