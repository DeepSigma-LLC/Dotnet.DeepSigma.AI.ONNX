# Dotnet.DeepSigma.AI.ONNX

A small, ergonomic .NET wrapper around [ONNX Runtime](https://onnxruntime.ai/) for inference.

- Load ONNX models from disk, byte arrays, or streams
- Run inference with a clean `Tensor<T>` abstraction — no `OrtValue` / `DenseTensor` / `NamedOnnxValue` leakage
- Multi-input / multi-output and single-shot convenience APIs
- CPU and CUDA execution providers; advanced users get an opt-in `SessionOptions` escape hatch

## Install

This library ships the **managed** ONNX Runtime bindings only. You pick the native package:

```xml
<PackageReference Include="DeepSigma.AI.ONNX" Version="..." />
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.27.0" /> <!-- CPU -->
<!-- or for NVIDIA GPU: -->
<!-- <PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.27.0" /> -->
```

## Quickstart

```csharp
using DeepSigma.AI.ONNX;

using OnnxModel model = OnnxModel.Load("model.onnx");

var input = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 1, 4 });
Tensor<float> output = model.Run(model.Inputs[0].Name, input);

Console.WriteLine(string.Join(", ", output.Data));
```

Multi-input:

```csharp
using OnnxModel model = OnnxModel.Load("model.onnx");

var inputs = new InferenceInput()
    .Add("a", new Tensor<float>(new[] { 1f, 2f }, new[] { 2 }))
    .Add("b", new Tensor<float>(new[] { 3f, 4f }, new[] { 2 }));

using InferenceResult result = model.Run(inputs);
Tensor<float> sum = result.Get<float>("c");
```

## CUDA

```csharp
var options = new ModelOptions
{
    Provider = ExecutionProvider.Cuda,
    DeviceId = 0,
};
using OnnxModel model = OnnxModel.Load("model.onnx", options);
```

Requires `Microsoft.ML.OnnxRuntime.Gpu` and the CUDA runtime on the host.

## Escape hatch

If you need configuration the wrapper doesn't expose, `ModelOptions.Configure` hands you the underlying `Microsoft.ML.OnnxRuntime.SessionOptions`:

```csharp
var options = new ModelOptions
{
    Configure = sessionOptions =>
    {
        sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1");
    },
};
```

## Scope

Inference only. Image decoding, tokenization, and training are intentionally out of scope — consumers bring their own and feed `Tensor<T>`.
