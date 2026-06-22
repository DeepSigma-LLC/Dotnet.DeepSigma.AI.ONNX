namespace DeepSigma.AI.ONNX;

public sealed record TensorSpec(
    string Name,
    TensorElementType ElementType,
    IReadOnlyList<long?> Dimensions);
