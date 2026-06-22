namespace DeepSigma.AI.ONNX;

public sealed record ModelMetadata(
    string ProducerName,
    string GraphName,
    string Domain,
    string Description,
    long Version,
    IReadOnlyDictionary<string, string> CustomMetadata);
