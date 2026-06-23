namespace DeepSigma.AI.ONNX;

/// <summary>Which ONNX Runtime execution provider to use for inference.</summary>
public enum ExecutionProvider
{
    /// <summary>CPU (default). Always available.</summary>
    Cpu,

    /// <summary>NVIDIA CUDA. Requires the Microsoft.ML.OnnxRuntime.Gpu native package + CUDA runtime.</summary>
    Cuda,
}
