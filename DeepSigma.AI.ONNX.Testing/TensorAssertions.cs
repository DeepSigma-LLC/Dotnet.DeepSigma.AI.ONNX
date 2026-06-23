namespace DeepSigma.AI.ONNX.Testing;

/// <summary>Element-wise tolerance comparison for float tensors.</summary>
public static class TensorAssertions
{
    public static void ShouldBeCloseTo(this Tensor<float> actual, float[] expected, float tolerance = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(expected);

        if (actual.Data.Length != expected.Length)
            throw new OnnxAssertionException(
                $"Tensor length mismatch: actual {actual.Data.Length}, expected {expected.Length}.");

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(actual.Data[i] - expected[i]);
            if (diff > tolerance)
                throw new OnnxAssertionException(
                    $"Tensor mismatch at index {i}: actual {actual.Data[i]}, expected {expected[i]}, diff {diff} > tolerance {tolerance}.");
        }
    }

    public static void ShouldBeCloseTo(this Tensor<float> actual, Tensor<float> expected, float tolerance = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(expected);
        if (!actual.Shape.SequenceEqual(expected.Shape))
            throw new OnnxAssertionException(
                $"Tensor shape mismatch: actual [{string.Join(", ", actual.Shape)}], expected [{string.Join(", ", expected.Shape)}].");
        actual.ShouldBeCloseTo(expected.Data, tolerance);
    }
}
