namespace DeepSigma.AI.ONNX.Testing;

/// <summary>Assertions about the static shape of a loaded <see cref="OnnxModel"/>.</summary>
public static class ModelAssertions
{
    public static void ShouldHaveInput(this OnnxModel model, string name)
    {
        FindInput(model, name);
    }

    public static void ShouldHaveInput(this OnnxModel model, string name, TensorElementType expectedType)
    {
        TensorSpec spec = FindInput(model, name);
        if (spec.ElementType != expectedType)
            throw new OnnxAssertionException(
                $"Input '{name}' expected element type {expectedType}, but model declares {spec.ElementType}.");
    }

    public static void ShouldHaveInput(
        this OnnxModel model,
        string name,
        TensorElementType expectedType,
        params long?[] expectedShape)
    {
        model.ShouldHaveInput(name, expectedType);
        TensorSpec spec = FindInput(model, name);
        AssertDimensions("Input", name, expectedShape, spec.Dimensions);
    }

    public static void ShouldHaveOutput(this OnnxModel model, string name)
    {
        FindOutput(model, name);
    }

    public static void ShouldHaveOutput(this OnnxModel model, string name, TensorElementType expectedType)
    {
        TensorSpec spec = FindOutput(model, name);
        if (spec.ElementType != expectedType)
            throw new OnnxAssertionException(
                $"Output '{name}' expected element type {expectedType}, but model declares {spec.ElementType}.");
    }

    private static TensorSpec FindInput(OnnxModel model, string name)
    {
        foreach (TensorSpec spec in model.Inputs)
            if (spec.Name == name) return spec;
        throw new OnnxAssertionException(
            $"Expected input '{name}'. Model inputs: {string.Join(", ", model.Inputs.Select(i => i.Name))}");
    }

    private static TensorSpec FindOutput(OnnxModel model, string name)
    {
        foreach (TensorSpec spec in model.Outputs)
            if (spec.Name == name) return spec;
        throw new OnnxAssertionException(
            $"Expected output '{name}'. Model outputs: {string.Join(", ", model.Outputs.Select(o => o.Name))}");
    }

    private static void AssertDimensions(string kind, string name, long?[] expected, IReadOnlyList<long?> actual)
    {
        if (expected.Length != actual.Count)
            throw new OnnxAssertionException(
                $"{kind} '{name}' expected rank {expected.Length}, but model declares rank {actual.Count}.");

        for (int i = 0; i < expected.Length; i++)
        {
            // null on either side = wildcard (caller doesn't care / model has a dynamic axis).
            if (expected[i] is null || actual[i] is null) continue;
            if (expected[i] != actual[i])
                throw new OnnxAssertionException(
                    $"{kind} '{name}' dimension {i} expected {expected[i]}, but model declares {actual[i]}.");
        }
    }
}
