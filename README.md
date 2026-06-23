# DeepSigma.AI.ONNX

A small, ergonomic .NET wrapper around [ONNX Runtime](https://onnxruntime.ai/) for inference, plus a framework-agnostic assertion package for testing ONNX models.

```csharp
using DeepSigma.AI.ONNX;

using OnnxModel model = OnnxModel.Load("model.onnx");
model.Describe();    // prints inputs / outputs / metadata

Tensor<float> output = model.Run("input", new Tensor<float>(features, [1, 4]));
```

---

## What is ONNX?

**ONNX** ([Open Neural Network Exchange](https://onnx.ai/)) is a portable file format for machine-learning models. You train a model in any framework — PyTorch, TensorFlow, scikit-learn, Keras, XGBoost — export it to a single `.onnx` file, and then run it anywhere that has an ONNX Runtime: C++, Python, C#, Java, JavaScript, mobile, edge devices.

**ONNX Runtime** is Microsoft's high-performance inference engine for `.onnx` files. It's cross-platform, hardware-accelerated (CPU, CUDA, DirectML, CoreML, TensorRT…), and ships official C# bindings via the `Microsoft.ML.OnnxRuntime` NuGet package.

**Why this wrapper?** The official C# API works, but it surfaces a lot of low-level concepts (`OrtValue`, `NamedOnnxValue`, `DenseTensor<T>`, `OrtMemoryInfo`, `SessionOptions`, lifetime management for native handles) that you don't want sprinkled through your application code. `DeepSigma.AI.ONNX` is a thin, opinionated layer that gives you:

- One main type — `OnnxModel` — that owns the session and exposes input/output specs as plain records
- A simple `Tensor<T>` abstraction so you never touch `DenseTensor<T>` or `NamedOnnxValue`
- Pre-flight input validation with named, actionable errors instead of ORT's generic messages
- A separate `DeepSigma.AI.ONNX.Testing` package with `ShouldHaveInput` / `ShouldNotContainNaN` / `ShouldMatchJsonSnapshot` style assertions for ML regression testing
- An escape hatch (`ModelOptions.Configure`) for the rare case you need the raw `SessionOptions`

It's deliberately small — inference only, no image decoding, no tokenization, no training. You bring those.

---

## Install

The library ships **managed bindings only**. You pick the native runtime package that matches your target:

```xml
<PackageReference Include="DeepSigma.AI.ONNX" Version="..." />

<!-- pick ONE of these: -->
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.27.0" />        <!-- CPU -->
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.27.0" />    <!-- NVIDIA CUDA -->
```

For testing assertions:

```xml
<PackageReference Include="DeepSigma.AI.ONNX.Testing" Version="..." />
```

Target framework: `net10.0`.

---

## Quickstart

### Single-input / single-output

```csharp
using DeepSigma.AI.ONNX;

using OnnxModel model = OnnxModel.Load("model.onnx");

var input = new Tensor<float>([1f, 2f, 3f, 4f], [1, 4]);
Tensor<float> output = model.Run(model.Inputs[0].Name, input);

Console.WriteLine(string.Join(", ", output.Data));
```

### Multi-input / multi-output

```csharp
using OnnxModel model = OnnxModel.Load("model.onnx");

var inputs = new InferenceInput()
    .Add("input_ids",      new Tensor<long>(tokenIds,        [1, 128]))
    .Add("attention_mask", new Tensor<long>(attentionMask,   [1, 128]));

using InferenceResult result = model.Run(inputs);

Tensor<long>  labels = result.Get<long>("label");
Tensor<float> probs  = result.Get<float>("probabilities");
```

### Inspecting a model you've never seen

```csharp
using OnnxModel model = OnnxModel.Load("mystery.onnx");
model.Describe();
// Producer: skl2onnx
// Graph:    ONNX(LogisticRegression)
// Inputs:
//   - X: Float [?, 4]
// Outputs:
//   - label: Int64 [?]
//   - probabilities: Float [?, 3]
```

`?` indicates a dynamic axis (e.g., variable batch size).

### Loading from bytes or a stream

```csharp
OnnxModel.Load(File.ReadAllBytes("model.onnx"));
OnnxModel.Load(httpResponse.GetResponseStream());
```

### Always dispose

`OnnxModel` and `InferenceResult` own native handles. Use `using`:

```csharp
using OnnxModel model = OnnxModel.Load(path);
using InferenceResult result = model.Run(inputs);
// both released here
```

---

## End-to-end: Python → ONNX → C#

This is the typical workflow: train (or fine-tune) in Python, export to ONNX, deploy in .NET. Here's the full loop using scikit-learn's classic Iris dataset.

### 1. Train + export in Python

```python
# pip install scikit-learn skl2onnx onnx numpy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import to_onnx

iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target

clf = LogisticRegression(max_iter=1000).fit(X, y)

# zipmap=False produces a clean float[N, 3] tensor instead of a sequence-of-maps,
# which is easier to consume from non-Python clients.
onnx_model = to_onnx(clf, X[:1], options={id(clf): {"zipmap": False}})

with open("iris.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

The exported model has:
- **Input:**  `X` (float, `[N, 4]`) — N samples, 4 features each
- **Output:** `label` (int64, `[N]`) and `probabilities` (float, `[N, 3]`)

### 2. Run it from C#

```csharp
using DeepSigma.AI.ONNX;

using OnnxModel model = OnnxModel.Load("iris.onnx");
model.Describe();

// One sample: setosa-shaped flower.
float[] features = [5.1f, 3.5f, 1.4f, 0.2f];
var input = new Tensor<float>(features, [1, 4]);

using InferenceResult result = model.Run(new InferenceInput().Add("X", input));

long  predictedClass = result.Get<long>("label").Data[0];
float[] classProbs   = result.Get<float>("probabilities").Data;

Console.WriteLine($"Predicted class: {predictedClass}");
Console.WriteLine($"Probabilities:   [{string.Join(", ", classProbs.Select(p => p.ToString("F4")))}]");
```

Output:
```
Predicted class: 0
Probabilities:   [0.9867, 0.0133, 0.0000]
```

A working version of this — plus a MNIST CNN from the public ONNX Model Zoo — lives in [`DeepSigma.AI.ONNX.Demo`](DeepSigma.AI.ONNX.Demo/) and [`interop/`](interop/).

---

## Pre-flight input validation

`Run(InferenceInput)` checks your inputs against the model's spec *before* handing off to ORT, so you get named errors:

```csharp
// Model expects [1, 1, 28, 28], you feed [1, 1, 30, 30]
// Throws: OnnxException("Input 'Input3' dimension 2 expected 28, got 30.")
```

Validation checks: required inputs present, no duplicates, no unknown names, element type matches, rank matches, every static dimension matches (dynamic axes are skipped).

---

## Testing your model — `DeepSigma.AI.ONNX.Testing`

A separate, **framework-agnostic** package (works with xUnit, NUnit, MSTest — anything that surfaces a thrown exception as a test failure). Built for the common ML problem: *"I re-exported my model. Does it still behave correctly?"*

```csharp
using DeepSigma.AI.ONNX;
using DeepSigma.AI.ONNX.Testing;

using OnnxModel model = OnnxModel.Load("iris.onnx");

// Sanity-check the model contract
model.ShouldHaveInput("X", TensorElementType.Float, expectedShape: [null, 4]);
model.ShouldHaveOutput("label", TensorElementType.Int64);
model.ShouldHaveOutput("probabilities", TensorElementType.Float);

// Run inference and assert on the outputs
using InferenceResult result = model.Run(new InferenceInput().Add("X", input));

result.ShouldHaveShape("probabilities", 1, 3);
result.ShouldNotContainNaN();
result.ShouldNotContainInfinity();
result.ShouldBeWithinRange("probabilities", min: 0f, max: 1f);

// Pin the outputs against a saved snapshot — catch behavior drift
result.ShouldMatchJsonSnapshot("snapshots/iris-setosa.json", tolerance: 1e-5f);
```

### Snapshot workflow

`ShouldMatchJsonSnapshot` writes a JSON file the first time (run with `DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS=1`), then asserts equality on every subsequent run. If you re-export the model and behavior drifts beyond tolerance, the test fails with the exact output name and index that diverged. Great for catching silent regressions during ML pipeline iteration.

```bash
# First run: create the snapshot
DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS=1 dotnet test

# Subsequent runs: just assert
dotnet test
```

---

## Configuration

```csharp
var options = new ModelOptions
{
    Provider             = ExecutionProvider.Cuda,        // default Cpu
    DeviceId             = 0,
    IntraOpThreads       = 4,
    InterOpThreads       = 1,
    Optimization         = GraphOptimizationLevel.All,
    LogLevel             = LogSeverity.Warning,
    EnableMemoryPattern  = true,
    EnableCpuMemArena    = true,
};

using OnnxModel model = OnnxModel.Load("model.onnx", options);
```

CUDA additionally requires `Microsoft.ML.OnnxRuntime.Gpu` (instead of the CPU package) and the matching CUDA runtime on the host.

### Escape hatch

For options not exposed by `ModelOptions`, the `Configure` hook hands you the raw ORT `SessionOptions` right before the session is built:

```csharp
var options = new ModelOptions
{
    Configure = sessionOptions =>
    {
        sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1");
        sessionOptions.RegisterCustomOpLibraryV2("path/to/custom_ops.dll", out _);
    },
};
```

Using this hook intentionally couples your code to `Microsoft.ML.OnnxRuntime`.

---

## What's in scope

**In scope:**
- Loading ONNX models (path / bytes / stream)
- Model inspection (`Inputs`, `Outputs`, `Metadata`, `Describe`)
- Running inference (single-shot and multi-input)
- Pre-flight input validation with clear errors
- Testing assertions in the `Testing` add-on package
- CPU + CUDA execution providers

**Out of scope (use a dedicated library):**
- Image decoding / preprocessing — bring [ImageSharp](https://github.com/SixLabors/ImageSharp), [SkiaSharp](https://github.com/mono/SkiaSharp), or `System.Drawing`; fill a `Tensor<float>` yourself
- Tokenization — bring [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers/) or [BlingFire](https://github.com/microsoft/BlingFire); feed token IDs as `Tensor<long>`
- Model training / fine-tuning — use Python; export to ONNX for inference
- Model conversion — use the official exporter for your training framework (`torch.onnx.export`, `tf2onnx`, `skl2onnx`, …)

---

## Project layout

```
DeepSigma.AI.ONNX/           Core library (managed-only ORT bindings)
DeepSigma.AI.ONNX.Testing/   Framework-agnostic assertions (separate package)
DeepSigma.AI.ONNX.Demo/      End-to-end demo: synthetic, sklearn, MNIST CNN
DeepSigma.AI.ONNX.Test/      xUnit v3 test suite
interop/                     Python export scripts + reference outputs
```

---

## License

MIT — see [LICENSE](LICENSE).
