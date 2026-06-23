namespace DeepSigma.AI.ONNX;

/// <summary>
/// Exception type raised by the wrapper. Wraps underlying ORT errors so consumers don't
/// need to reference Microsoft.ML.OnnxRuntime to catch them.
/// </summary>
public class OnnxException : Exception
{
    public OnnxException(string message) : base(message) { }
    public OnnxException(string message, Exception innerException) : base(message, innerException) { }
}
