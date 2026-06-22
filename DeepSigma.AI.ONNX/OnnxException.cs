namespace DeepSigma.AI.ONNX;

public class OnnxException : Exception
{
    public OnnxException(string message) : base(message) { }
    public OnnxException(string message, Exception innerException) : base(message, innerException) { }
}
