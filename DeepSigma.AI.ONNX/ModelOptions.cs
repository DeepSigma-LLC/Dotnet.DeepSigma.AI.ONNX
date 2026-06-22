using Microsoft.ML.OnnxRuntime;

namespace DeepSigma.AI.ONNX;

public sealed class ModelOptions
{
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Cpu;

    public int DeviceId { get; set; } = 0;

    public int? InterOpThreads { get; set; }

    public int? IntraOpThreads { get; set; }

    public GraphOptimizationLevel Optimization { get; set; } = GraphOptimizationLevel.All;

    public LogSeverity LogLevel { get; set; } = LogSeverity.Warning;

    public bool EnableMemoryPattern { get; set; } = true;

    public bool EnableCpuMemArena { get; set; } = true;

    /// <summary>
    /// Escape hatch: mutate the underlying SessionOptions before session construction.
    /// Using this hook intentionally couples your code to Microsoft.ML.OnnxRuntime.
    /// </summary>
    public Action<SessionOptions>? Configure { get; set; }
}
