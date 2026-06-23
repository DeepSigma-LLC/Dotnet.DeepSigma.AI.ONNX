namespace DeepSigma.AI.ONNX.Internal;

internal static class InputValidator
{
    public static void Validate(
        IReadOnlyList<InputBinding> provided,
        IReadOnlyList<TensorSpec> expected)
    {
        // Build a name->spec map and check for missing / extra / duplicate up front.
        var expectedByName = new Dictionary<string, TensorSpec>(expected.Count, StringComparer.Ordinal);
        foreach (TensorSpec spec in expected) expectedByName[spec.Name] = spec;

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (InputBinding binding in provided)
        {
            if (!seen.Add(binding.Name))
                throw new OnnxException($"Input '{binding.Name}' was provided more than once.");

            if (!expectedByName.ContainsKey(binding.Name))
                throw new OnnxException(
                    $"Unknown input '{binding.Name}'. Model expects: {string.Join(", ", expectedByName.Keys)}");
        }

        foreach (TensorSpec spec in expected)
        {
            if (!seen.Contains(spec.Name))
                throw new OnnxException(
                    $"Missing required input '{spec.Name}'. Model expects: {string.Join(", ", expectedByName.Keys)}");
        }

        // Now validate each binding's type and shape against its spec.
        foreach (InputBinding binding in provided)
        {
            TensorSpec spec = expectedByName[binding.Name];

            if (binding.ElementType != spec.ElementType)
                throw new OnnxException(
                    $"Input '{binding.Name}' expected element type {spec.ElementType}, got {binding.ElementType}.");

            if (binding.Shape.Length != spec.Dimensions.Count)
                throw new OnnxException(
                    $"Input '{binding.Name}' expected rank {spec.Dimensions.Count}, got rank {binding.Shape.Length}.");

            for (int i = 0; i < spec.Dimensions.Count; i++)
            {
                long? expectedDim = spec.Dimensions[i];
                if (expectedDim is long fixedDim && fixedDim != binding.Shape[i])
                {
                    throw new OnnxException(
                        $"Input '{binding.Name}' dimension {i} expected {fixedDim}, got {binding.Shape[i]}.");
                }
            }
        }
    }
}
