namespace DeepSigma.AI.ONNX;

/// <summary>Producer / graph / version metadata embedded in an ONNX model file.</summary>
public sealed record ModelMetadata(
    string ProducerName,
    string GraphName,
    string Domain,
    string Description,
    long Version,
    IReadOnlyDictionary<string, string> CustomMetadata);
