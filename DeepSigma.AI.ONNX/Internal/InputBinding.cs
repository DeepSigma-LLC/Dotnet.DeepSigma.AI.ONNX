using Microsoft.ML.OnnxRuntime;

namespace DeepSigma.AI.ONNX.Internal;

internal sealed record InputBinding(
    string Name,
    TensorElementType ElementType,
    int[] Shape,
    Func<OrtValue> Materialize);
