namespace DeepSigma.AI.ONNX;

/// <summary>How aggressively ONNX Runtime should optimize the graph before inference.</summary>
public enum GraphOptimizationLevel
{
    Disabled,
    Basic,
    Extended,
    All,
}
