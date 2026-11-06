# Roadmap

- **Written:** 2026-11-03, from the planning conversation that started this project.
- **Last updated:** 2026-11-06, after the `eb_jepa` Two Rooms run and the planning protocol draft
  closed Week 1.
- **Status:** living document. Executable milestone plans go in `docs/plans/`. Settled choices
  go in `docs/decisions.md`.

## Why this project exists

Language models produce text, so a developer can tell whether a call worked by reading the output.
That property is why an application layer grew so quickly around them, through hosted APIs, Ollama,
LangChain, LlamaIndex and vLLM. JEPA models (Joint-Embedding Predictive Architectures, the line of
work led by Yann LeCun) predict in a learned latent space instead. Their output is a tensor of
embeddings and an energy, and nothing in it tells a developer whether it is right.

Meta's V-JEPA 2 and the `eb_jepa` library are good research code. What is missing is the developer
layer that would let an engineer without a research background build on them. This project builds
that layer in the open, under Apache-2.0, as a sequence of small tools that each remove one
concrete bottleneck.

## The bottleneck, concretely

Cloning `facebookresearch/vjepa2` and trying to use it shows where things break down.

1. **Environment.** The setup assumes a CUDA cluster. Video decoding has historically used
   `decord`, which is hard to install on Apple Silicon. Configs are YAML files for distributed
   training, not a path for "I have one video".
2. **Weights.** There are several loading routes with different code: Hugging Face checkpoints,
   `torch.hub`, and raw files. The action-conditioned model (V-JEPA 2-AC) loads only through
   `torch.hub`.
3. **Preprocessing.** Frame count, sampling stride, resize, crop and normalisation all have to
   match training. When any of them is wrong, nothing errors. The model returns plausible-looking
   vectors that are simply wrong.
4. **Output.** The encoder returns something like `float16[8192, 1408]` (ViT-g on 64 frames).
   The tensor carries no record of its time, row and column layout, so every tool has to rebuild
   it.
5. **Tasks.** There is no prompt. Every task needs a trained head (a probe), and the published
   evaluation recipes need datasets of hundreds of gigabytes.

`eb_jepa` (arXiv 2602.03604, February 2026) is much friendlier, with small examples that train on
one GPU in a few hours. It is still a training codebase, with no serving, no stable API and an
energy reported as a unitless number in a training log.

| | `ollama run llama3` | V-JEPA 2 today |
| --- | --- | --- |
| Input contract | Text | An undocumented preprocessing recipe |
| Output | Readable, so you can check it yourself | A tensor you cannot check by looking |
| Task interface | The prompt | A probe you train per task |
| Knowing it worked | Read it | Needs instruments |
| Runs on a laptop | Yes | ViT-L does, but nobody packaged it |

## The missing toolchain

These are the layers between a research checkpoint and an application, roughly in the order they
are needed.

- **A. Runtime and packaging.** One loader that bundles the model, its preprocessing recipe and its
  token layout. Decoding that works everywhere, MPS and CPU fallbacks, streaming with latent reuse,
  and later ONNX or Core ML export.
- **B. A latent data contract.** A file format that stores latents together with the model id,
  grid, preprocessing hash and source hash, so tools can share data without guessing. jepa-scope's
  `.latent` file is the first version.
- **C. Visualisation and observability.** PCA colour maps of patch features, surprise maps (the
  per-patch prediction error), embedding health checks such as effective rank and per-dimension
  variance, and latent trajectories over time.
- **D. Latent-to-task adapters.** Probes trained in seconds on cached embeddings, retrieval over
  pooled embeddings, small decoders for debugging, and alignment to language models.
- **E. Latent-to-action planning.** A model-agnostic protocol (`encode(obs) → z`,
  `predict(z, a) → z'`, `energy(z, z_goal)`) with planners such as CEM and MPPI written against
  it, and bridges to simulators and robot-learning datasets.
- **F. A laptop-scale evaluation harness.** Small benchmarks that finish in minutes, so tools and
  new checkpoints can be compared.

The tools target the protocol in B and E rather than V-JEPA 2 itself. When Meta, AMI Labs or anyone
else releases new weights, they become one more adapter instead of a reason to start over.

## The three projects

jepa-scope comes first, because it produces the data contract and viewer shell that the other two
reuse. The order of the other two is a proposal until one of them is planned.

### 1. jepa-scope, a local latent inspector (this repository)

`jepa-scope clip.mp4` embeds a video with V-JEPA 2 on a laptop and opens a browser viewer. The
viewer shows PCA feature colours over the video, a surprise overlay, and a surprise timeline you can
click to seek. It addresses the "I got a tensor, now what?" problem and the silent preprocessing
problem.

- **Frontend leverage.** The whole timeline is uploaded once into WebGL2 texture arrays, so
  scrubbing changes only a uniform. Planned additions are a Three.js view of a clip's path through
  latent space and nearest-neighbour search across a folder of clips.
- **Stack.** A Python backend with FastAPI, Hugging Face `transformers` and PyAV, packaged with uv.
  A Vite, React and TypeScript viewer that ships inside the Python wheel, so users never need npm.
  Latents travel to the browser as raw bytes, not JSON.
- **MVP, four to six weeks.** Weeks 1 and 2 cover inference, the grid, the PCA overlay, the file
  format and the CLI. Week 3 adds surprise maps. Week 4 adds folder indexing and nearest-neighbour
  search. Weeks 5 and 6 add health metrics, polish and a launch demo. Surprise maps arrived early,
  in Week 1, and this roadmap now also proposes the latent trajectory view for Weeks 3 and 4.

### 2. jepa-probe, few-shot task heads behind an HTTP API

This project is the JEPA equivalent of a prompt. A user labels 30 to 100 clip segments in a browser
and gets a working classifier or detector in under a minute. The result is served through endpoints
shaped like familiar model APIs, such as `POST /v1/video/embeddings` and
`POST /v1/heads/{name}/predict`, so an application developer works with labels instead of vectors.

- **Frontend leverage.** Timeline labelling with drag-to-select and hotkeys, active learning that
  surfaces the clips the head is least sure about, a live confusion matrix, and probe attention
  drawn over the frame.
- **Stack.** It reuses jepa-scope's latent cache. Linear and attentive probes in PyTorch, exported
  to ONNX.
- **MVP.** Embed a folder, label it, train and evaluate both probe types, and serve the result. One
  end-to-end example built from about 50 labelled home videos.

### 3. latent-plan, a planning runtime and playground

This project is a planning runtime written against the protocol in layer E, with a browser UI. A
user clicks a goal and watches CEM or MPPI sample action sequences, roll them out in latent space
and converge on the energy landscape. The first adapter is an `eb_jepa` Two Rooms world model
trained locally. The second is V-JEPA 2-AC, run offline on a rented GPU. The protocol draft, a
reference CEM planner and the `eb_jepa` Two Rooms adapter already exist in this repository
(`src/jepa_scope/planning/`, specified in `docs/planning-protocol.md`) and move here when this
project starts.

- **Frontend leverage.** The energy over a two-dimensional slice of action space as an animated
  WebGL heatmap, sampled trajectories with the best ones highlighted, and the real rollout beside
  the predicted one to show where the model's prediction drifts.
- **Stack.** The Python protocol and planners in PyTorch, gymnasium-style environment bridges, and
  the same web shell as jepa-scope.
- **MVP.** The protocol, CEM and MPPI, the `eb_jepa` adapter, the Two Rooms UI, and a written spec
  so other people can add their own models.
- **Risk.** This is the most research-dependent of the three, because planning quality depends on
  the model, and its audience is smaller today.

## Compute strategy

1. **Never pretrain. Freeze the encoder.** Projects 1 and 2 are inference plus small heads.
2. **Embed once, then work on cached arrays.** The expensive encoder runs once per clip. Probes,
   PCA, search and the viewer all work from cached latents on a CPU. At the defaults, patch
   features cost about 4 MB per window and pooled embeddings about 4 KB.
3. **Use ViT-L with 16-frame windows.** Sixteen frames give 2,048 tokens, against 8,192 for the
   64-frame setting, and attention cost grows with the square of the token count. In one measurement
   on an M3 Max, a 10-second clip embedded in about 3.5 seconds once the model was loaded. Use the
   giant model only to confirm that a conclusion holds at scale.
4. **Rent GPUs only for batch jobs,** such as bulk embedding or `eb_jepa` training runs. Use spot
   instances and scripts that tear the machine down when the job ends.
5. **`eb_jepa`'s small environments are the world-model lab.** They retrain in hours on one GPU,
   with no dependence on the large checkpoints.

Expected cost for the first six weeks is close to zero beyond a laptop, plus perhaps tens of dollars
of GPU time for `eb_jepa` runs.

## Progress

### Week 1 (2026-11-03 to 2026-11-06)

| Step | Status |
| --- | --- |
| Run ViT-L on Apple Silicon and check the token count | Done. Shapes are pinned by tests |
| Rebuild the time, row and column grid from the flat tokens | Done. A test lights one pixel and checks which token changes |
| PCA colour overlay | Done |
| Surprise from masked prediction | Done. A cut in a spliced clip is detected; see below |
| `.latent` format and CLI | Done |
| Web viewer with overlay and timeline | Done |
| Run the `eb_jepa` Two Rooms example and draft the planning protocol | Done. Rollouts and same-noise plans match `eb_jepa`'s; in a closed-loop check `eb_jepa`'s planner solved 18 of 20 episodes and the protocol's 16; see `docs/planning-protocol.md` |
| Publish under Apache-2.0 | Done: `ryangswe/jepa-scope`, with CI |

The first local milestone was "open a clip on a MacBook without CUDA and see a PCA overlay and a
surprise timeline whose spikes match what happens in the video". It was met on a clip that splices
an archery scene onto a bowling scene. In one run, surprise stayed between about 0.57 and 0.62
within each scene and rose to between 0.65 and 0.67 for the second after the cut.

### What Week 1 taught

- **Two silent errors in the surprise score.** The Hugging Face forward pass lets context tokens
  attend to the frames being predicted. Comparing predictions with raw targets instead of
  layer-normalised ones also produces a flat line. Both ran without any error. The details are in
  `docs/decisions/method.md`.
- **Test media matters.** The first test clip was two near-identical static title cards, and it
  showed nothing useful. Natural video with motion is the minimum for judging a signal.
- **Expect a floor.** Surprise never reaches zero, and what carries information is the change
  relative to the clip's own floor.
- **A wrapper that runs is not a wrapper that is right.** The `eb_jepa` adapter was trusted only
  after its rollouts matched `eb_jepa`'s own to the last digit and, given the same random noise,
  its planner returned `eb_jepa`'s plans. Closed-loop success on 20 episodes (16 against 18) was a
  useful smoke test but could not show the two planners equal; an outside review pointed out that
  the "paired" pass rule reduced to a net count. Controlled comparisons answer "is it the same
  computation?"; small success-rate runs do not.

### Next

Milestones follow the weekly order. Week 2's scope landed in Week 1, so the next milestones are:

1. jepa-scope Weeks 3 and 4: folder indexing, nearest-neighbour search and the latent
   trajectory view.
2. jepa-scope Weeks 5 and 6: embedding health metrics, polish and a launch demo.

Each milestone gets a plan in `docs/plans/` before code changes, written by the `planner` agent and
approved by the owner.

## Open questions and risks

- **Predictor training distribution.** V-JEPA 2's predictor was trained mostly on masks that span
  the whole clip in time. Predicting the future from the past alone falls outside that training,
  so surprise is a useful signal but not a calibrated one. V-JEPA 2-AC is trained for future
  prediction and may be a better source later.
- **New weights.** AMI Labs has said it will open-source some of its work. We know of no
  open-weights model from it as of November 2026. Targeting a protocol instead of one model is the
  hedge.
- **Who the users are.** The near-term audience is likely robotics and video analytics rather than
  consumer apps. That favours jepa-probe's labelling-to-API path over chat-style demos.
- **Licences.** jepa-scope downloads V-JEPA 2 weights at run time and does not redistribute them.
  Meta releases V-JEPA 2 under MIT.

## Sources

- EB-JEPA: [arXiv 2602.03604](https://arxiv.org/abs/2602.03604)
- V-JEPA 2: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) and the
  [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/vjepa2)
- Prior art for viewers: `mosure/burn_jepa` (Rust and Bevy) and `keon/jepa` (teaching code with
  t-SNE and PCA plots)
- AMI Labs funding: [SiliconANGLE, 2026-03-10](https://siliconangle.com/2026/03/10/yann-lecuns-new-startup-ami-labs-raises-1-03b-train-world-models/)
