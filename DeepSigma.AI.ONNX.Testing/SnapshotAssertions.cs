using System.Text.Json;
using System.Text.Json.Serialization;

namespace DeepSigma.AI.ONNX.Testing;

/// <summary>
/// JSON snapshot matching for inference results. Useful for "did my model export change behavior?"
/// regression tests.
/// </summary>
/// <remarks>
/// File format: <c>{ "outputName": { "shape": [...], "values": [...] }, ... }</c>.
/// If the snapshot file is missing and the env var <c>DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS=1</c> is set,
/// the file is created from the current results and the assertion passes. Otherwise, missing
/// files fail with a helpful message.
/// </remarks>
public static class SnapshotAssertions
{
    private const string UpdateEnvVar = "DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static void ShouldMatchJsonSnapshot(
        this InferenceResult result,
        string snapshotPath,
        float tolerance = 1e-5f)
    {
        ArgumentException.ThrowIfNullOrEmpty(snapshotPath);

        Dictionary<string, SnapshotEntry> current = BuildSnapshot(result);
        bool updateMode = string.Equals(
            Environment.GetEnvironmentVariable(UpdateEnvVar),
            "1",
            StringComparison.Ordinal);

        if (!File.Exists(snapshotPath))
        {
            if (!updateMode)
            {
                throw new OnnxAssertionException(
                    $"Snapshot file '{snapshotPath}' does not exist. " +
                    $"Set env var {UpdateEnvVar}=1 to create it from the current outputs.");
            }
            WriteSnapshot(snapshotPath, current);
            return;
        }

        if (updateMode)
        {
            WriteSnapshot(snapshotPath, current);
            return;
        }

        Dictionary<string, SnapshotEntry>? saved = JsonSerializer.Deserialize<Dictionary<string, SnapshotEntry>>(
            File.ReadAllText(snapshotPath), JsonOptions);
        if (saved is null)
            throw new OnnxAssertionException($"Snapshot file '{snapshotPath}' is empty or invalid JSON.");

        CompareSnapshots(saved, current, tolerance, snapshotPath);
    }

    private static Dictionary<string, SnapshotEntry> BuildSnapshot(InferenceResult result)
    {
        var snap = new Dictionary<string, SnapshotEntry>(StringComparer.Ordinal);
        foreach (string name in result.Names)
        {
            if (result.TryGet<float>(name, out Tensor<float>? f) && f is not null)
                snap[name] = new SnapshotEntry(f.Shape.ToArray(), f.Data);
            else if (result.TryGet<double>(name, out Tensor<double>? d) && d is not null)
                snap[name] = new SnapshotEntry(d.Shape.ToArray(), Array.ConvertAll(d.Data, x => (float)x));
            else if (result.TryGet<long>(name, out Tensor<long>? l) && l is not null)
                snap[name] = new SnapshotEntry(l.Shape.ToArray(), Array.ConvertAll(l.Data, x => (float)x));
            else if (result.TryGet<int>(name, out Tensor<int>? i) && i is not null)
                snap[name] = new SnapshotEntry(i.Shape.ToArray(), Array.ConvertAll(i.Data, x => (float)x));
            // Other types skipped silently — strings, bools, etc. don't make sense as numeric snapshots.
        }
        return snap;
    }

    private static void WriteSnapshot(string path, Dictionary<string, SnapshotEntry> snapshot)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(snapshot, JsonOptions));
    }

    private static void CompareSnapshots(
        Dictionary<string, SnapshotEntry> expected,
        Dictionary<string, SnapshotEntry> actual,
        float tolerance,
        string snapshotPath)
    {
        foreach ((string name, SnapshotEntry expectedEntry) in expected)
        {
            if (!actual.TryGetValue(name, out SnapshotEntry? actualEntry) || actualEntry is null)
                throw new OnnxAssertionException(
                    $"Snapshot '{snapshotPath}' has output '{name}' but current results don't.");

            if (!expectedEntry.Shape.SequenceEqual(actualEntry.Shape))
                throw new OnnxAssertionException(
                    $"Snapshot '{snapshotPath}' output '{name}': expected shape [{string.Join(", ", expectedEntry.Shape)}], got [{string.Join(", ", actualEntry.Shape)}].");

            if (expectedEntry.Values.Length != actualEntry.Values.Length)
                throw new OnnxAssertionException(
                    $"Snapshot '{snapshotPath}' output '{name}': expected {expectedEntry.Values.Length} values, got {actualEntry.Values.Length}.");

            for (int i = 0; i < expectedEntry.Values.Length; i++)
            {
                float diff = MathF.Abs(expectedEntry.Values[i] - actualEntry.Values[i]);
                if (diff > tolerance)
                    throw new OnnxAssertionException(
                        $"Snapshot '{snapshotPath}' output '{name}' index {i}: expected {expectedEntry.Values[i]}, got {actualEntry.Values[i]}, diff {diff} > tolerance {tolerance}.");
            }
        }
    }

    private sealed record SnapshotEntry(
        [property: JsonPropertyName("shape")] int[] Shape,
        [property: JsonPropertyName("values")] float[] Values);
}
