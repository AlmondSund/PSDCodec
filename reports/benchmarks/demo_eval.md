# PSDCodec Demo Rate-Distortion-Complexity Evaluation

## Validation Reference

- Training summary: `/home/marti/Code/PSDCodec/models/exports/demo/training_summary.json`
- Best epoch: `76`
- Held-out mean PSD distortion: `0.012568`
- Held-out mean preprocessing-only distortion: `0.021418`
- Held-out mean rate proxy: `4633.078` bits/frame
- Held-out task monitor: `3.604386`
- Held-out deployment score: `0.828894`

## Deployment Benchmark

- Export directory: `/home/marti/Code/PSDCodec/models/exports/demo`
- Checkpoint: `/home/marti/Code/PSDCodec/models/checkpoints/demo/best.pt`
- ONNX provider: `CPUExecutionProvider`
- Dataset source: `raw_campaigns`
- Dataset path: `/home/marti/Code/PSDCodec/data/raw/campaigns`
- Benchmark boundary: `deployment_benchmark_subset`
- Materialized benchmark frames: `64`
- Distortion/payload frames: `64`
- Timed runtime frames: `64`
- Frame dimensions: `N=4096`, `N_r=1024`, `B=32`
- Compatibility-excluded campaigns: `none`

## Reconstruction Quality

- Mean PSD distortion: `0.009572`
- PSD distortion std: `0.002214`
- PSD distortion range: `[0.006673, 0.016771]`
- Mean preprocessing-only distortion: `0.015517`
- Mean codec-only distortion: `0.012335`
- Mean illustrative task distortion: `2.772227`

## Operational Cost

- Mean operational payload: `4619.141` bits/frame
- Payload std: `10.868` bits/frame
- Payload range: `[4595, 4640]` bits
- Mean side-information payload: `704.000` bits/frame
- Mean index payload: `3915.141` bits/frame
- Mean rate proxy: `4625.896` bits/frame
- Mean bits/original bin: `1.127720`
- Mean bits/reduced bin: `4.510880`
- Mean bits/latent index: `9.021759`

## Runtime

- Encode latency: `304.975 ± 73.900` ms/frame
- Encode latency range: `[191.141, 539.595]` ms
- Decode latency: `176.347 ± 35.175` ms/frame
- Decode latency range: `[79.907, 269.054]` ms
- Exact packet round-trip fraction on timed frames: `1.000000`

## Model Complexity

- Total parameters: `2310690`
- Trainable parameters: `2310690`
- Encoder parameters: `1153048`
- Vector-quantizer parameters: `4096`
- Decoder parameters: `1153034`
- Entropy-model parameters: `512`

## Interpretation

This report is the deployment-oriented rate-distortion-complexity characterization of the PSDCodec demo: the validation reference keeps the original held-out training metrics visible, the deployment benchmark measures practical payload and host-side runtime on a deterministic raw-frame subset, and the complexity section reports the size of the complete learned model.
