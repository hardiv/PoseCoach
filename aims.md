# Experiment Aims

We aim to answer the question, through this benchmarking experiment, of:
"Which pose estimation model should we use for our form assessment system, and when will it fail?"

We need to know:

- Which joints are unreliable (so we don't build form rules around them)
- Which models are consistent (low variance matters more than perfect accuracy)
- Speed vs accuracy tradeoffs (for mobile deployment)
- What conditions cause failure (viewpoint, occlusion, clothing, etc.)

Phase 1: Baseline Validation

- Dataset: COCO/MPII (150-300 images)
- Purpose: Sanity check. Do these models work at all? What's their general behavior?
- Metrics: Overall error, speed, consistency
This is fine for showing that we understand benchmarking methodology, but it's
not the real evaluation.

Phase 2: Exercise-Specific Testing

- Dataset: Images/videos of actual exercises
- Purpose: Answer "do these models work for MY problem?"
You need:
- Specific exercises - Pick 2-3 (squat, deadlift, bench press)
- Varied capture conditions:
  - Front view, side view, 45° angle
  - Good lighting, dim lighting
  - Tight clothing, baggy clothing
  - Clear background, cluttered gym background
  - Different body types if possible

Ground truth OR qualitative assessment:

- Ideal: Manually annotate key joints on 50-100 exercise images
- Realistic: Visual inspection + confidence scores (does the skeleton look right?)
