namespace DeepSigma.AI.ONNX;

/// <summary>
/// Describes a model input or output tensor. A null entry in <paramref name="Dimensions"/>
/// indicates a dynamic axis (e.g., variable batch size).
/// </summary>
public sealed record TensorSpec(
    string Name,
    TensorElementType ElementType,
    IReadOnlyList<long?> Dimensions);
