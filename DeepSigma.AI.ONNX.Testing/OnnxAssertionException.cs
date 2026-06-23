namespace DeepSigma.AI.ONNX.Testing;

/// <summary>
/// Thrown when a model or result assertion fails. Any test framework (xUnit, NUnit, MSTest)
/// will surface this as a test failure.
/// </summary>
public sealed class OnnxAssertionException : Exception
{
    public OnnxAssertionException(string message) : base(message) { }
}
