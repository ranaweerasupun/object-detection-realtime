# Dev Log

Running notes from building this project.

---

## 2026 March - 2026 April

### Detection not working — objects not recognized

**Problem:** Pointed the camera at a water bottle, a phone, a person — model was either not detecting anything or giving very low confidence (20–30%).  
**First suspicion:** Thought it might be a code issue with how frames were being passed to the model.  
**Code was okay:** Checked the code but everthing seems to be okay. Verified the input shape and dtype matched what the model expected. Looked fine.
**The cause ! :** Lighting. The room looked fine to me but the camera was struggling — images were coming out noisy and underexposed.