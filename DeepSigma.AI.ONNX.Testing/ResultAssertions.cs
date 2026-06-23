namespace DeepSigma.AI.ONNX.Testing;

/// <summary>Assertions about the outputs of an inference call.</summary>
public static class ResultAssertions
{
    public static void ShouldHaveOutput(this InferenceResult result, string name)
    {
        if (!result.Names.Contains(name))
            throw new OnnxAssertionException(
                $"Expected output '{name}'. Available: {string.Join(", ", result.Names)}");
    }

    public static void ShouldHaveShape(this InferenceResult result, string name, params int[] expectedShape)
    {
        result.ShouldHaveOutput(name);
        // We read as float to access shape; if the output isn't float, fall back to long, then bytes.
        IReadOnlyList<int> actual = ReadShape(result, name);
        if (actual.Count != expectedShape.Length || !actual.SequenceEqual(expectedShape))
            throw new OnnxAssertionException(
                $"Output '{name}' expected shape [{string.Join(", ", expectedShape)}], got [{string.Join(", ", actual)}].");
    }

    public static void ShouldNotContainNaN(this InferenceResult result)
    {
        foreach (string name in result.Names)
        {
            if (result.TryGet<float>(name, out Tensor<float>? floats) && floats is not null)
                AssertNoNaN(name, floats.Data);
            else if (result.TryGet<double>(name, out Tensor<double>? doubles) && doubles is not null)
                AssertNoNaN(name, doubles.Data);
        }
    }

    public static void ShouldNotContainNaN(this InferenceResult result, string name)
    {
        result.ShouldHaveOutput(name);
        if (result.TryGet<float>(name, out Tensor<float>? floats) && floats is not null)
            AssertNoNaN(name, floats.Data);
        else if (result.TryGet<double>(name, out Tensor<double>? doubles) && doubles is not null)
            AssertNoNaN(name, doubles.Data);
        else
            throw new OnnxAssertionException(
                $"Output '{name}' is not a floating-point tensor; NaN check is not applicable.");
    }

    public static void ShouldNotContainInfinity(this InferenceResult result)
    {
        foreach (string name in result.Names)
        {
            if (result.TryGet<float>(name, out Tensor<float>? floats) && floats is not null)
                AssertNoInfinity(name, floats.Data);
            else if (result.TryGet<double>(name, out Tensor<double>? doubles) && doubles is not null)
                AssertNoInfinity(name, doubles.Data);
        }
    }

    public static void ShouldNotContainInfinity(this InferenceResult result, string name)
    {
        result.ShouldHaveOutput(name);
        if (result.TryGet<float>(name, out Tensor<float>? floats) && floats is not null)
            AssertNoInfinity(name, floats.Data);
        else if (result.TryGet<double>(name, out Tensor<double>? doubles) && doubles is not null)
            AssertNoInfinity(name, doubles.Data);
        else
            throw new OnnxAssertionException(
                $"Output '{name}' is not a floating-point tensor; infinity check is not applicable.");
    }

    public static void ShouldBeWithinRange(this InferenceResult result, string name, float min, float max)
    {
        result.ShouldHaveOutput(name);
        if (!result.TryGet<float>(name, out Tensor<float>? tensor) || tensor is null)
            throw new OnnxAssertionException(
                $"Output '{name}' is not a Float tensor; range check requires Float.");

        for (int i = 0; i < tensor.Data.Length; i++)
        {
            float v = tensor.Data[i];
            if (v < min || v > max)
                throw new OnnxAssertionException(
                    $"Output '{name}' value at index {i} ({v}) is outside [{min}, {max}].");
        }
    }

    private static IReadOnlyList<int> ReadShape(InferenceResult result, string name)
    {
        if (result.TryGet<float>(name, out Tensor<float>? f) && f is not null) return f.Shape;
        if (result.TryGet<double>(name, out Tensor<double>? d) && d is not null) return d.Shape;
        if (result.TryGet<long>(name, out Tensor<long>? l) && l is not null) return l.Shape;
        if (result.TryGet<int>(name, out Tensor<int>? i) && i is not null) return i.Shape;
        if (result.TryGet<byte>(name, out Tensor<byte>? b) && b is not null) return b.Shape;
        throw new OnnxAssertionException(
            $"Could not read shape for output '{name}' as any supported element type.");
    }

    private static void AssertNoNaN(string name, float[] data)
    {
        for (int i = 0; i < data.Length; i++)
            if (float.IsNaN(data[i]))
                throw new OnnxAssertionException($"Output '{name}' contains NaN at index {i}.");
    }

    private static void AssertNoNaN(string name, double[] data)
    {
        for (int i = 0; i < data.Length; i++)
            if (double.IsNaN(data[i]))
                throw new OnnxAssertionException($"Output '{name}' contains NaN at index {i}.");
    }

    private static void AssertNoInfinity(string name, float[] data)
    {
        for (int i = 0; i < data.Length; i++)
            if (float.IsInfinity(data[i]))
                throw new OnnxAssertionException($"Output '{name}' contains Infinity at index {i}.");
    }

    private static void AssertNoInfinity(string name, double[] data)
    {
        for (int i = 0; i < data.Length; i++)
            if (double.IsInfinity(data[i]))
                throw new OnnxAssertionException($"Output '{name}' contains Infinity at index {i}.");
    }
}
