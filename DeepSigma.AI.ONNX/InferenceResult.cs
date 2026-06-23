using Microsoft.ML.OnnxRuntime;
using DeepSigma.AI.ONNX.Internal;

namespace DeepSigma.AI.ONNX;

/// <summary>
/// The outputs of an inference call. Holds native ORT tensors until disposed — always dispose
/// (or use a <c>using</c> block) to release them.
/// </summary>
public sealed class InferenceResult : IDisposable
{
    private readonly Dictionary<string, OrtValue> _outputs;
    private bool _disposed;

    public IReadOnlyList<string> Names { get; }

    internal InferenceResult(IReadOnlyList<string> names, IReadOnlyList<OrtValue> values)
    {
        if (names.Count != values.Count)
            throw new OnnxException(
                $"Output name/value count mismatch: {names.Count} names, {values.Count} values.");

        Names = names;
        _outputs = new Dictionary<string, OrtValue>(names.Count, StringComparer.Ordinal);
        for (int i = 0; i < names.Count; i++)
        {
            _outputs[names[i]] = values[i];
        }
    }

    /// <summary>Read an output tensor by name. Throws if the name is missing or the element type doesn't match.</summary>
    public Tensor<T> Get<T>(string name) where T : unmanaged
    {
        ThrowIfDisposed();
        if (!_outputs.TryGetValue(name, out OrtValue? value))
            throw new OnnxException($"No output named '{name}'. Available: {string.Join(", ", Names)}");
        return TensorMarshaller.FromOrtValue<T>(value);
    }

    /// <summary>
    /// Try to read an output tensor by name. Returns false (without throwing) if the name is missing
    /// OR the element type doesn't match T.
    /// </summary>
    public bool TryGet<T>(string name, out Tensor<T>? tensor) where T : unmanaged
    {
        ThrowIfDisposed();
        if (!_outputs.TryGetValue(name, out OrtValue? value) ||
            !TensorMarshaller.ElementTypeMatches<T>(value))
        {
            tensor = null;
            return false;
        }
        tensor = TensorMarshaller.FromOrtValue<T>(value);
        return true;
    }

    public string[] GetStrings(string name, out int[] shape)
    {
        ThrowIfDisposed();
        if (!_outputs.TryGetValue(name, out OrtValue? value))
            throw new OnnxException($"No output named '{name}'. Available: {string.Join(", ", Names)}");
        return TensorMarshaller.StringsFromOrtValue(value, out shape);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var v in _outputs.Values) v.Dispose();
        _outputs.Clear();
        _disposed = true;
    }
}
