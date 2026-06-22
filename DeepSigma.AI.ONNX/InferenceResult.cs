using Microsoft.ML.OnnxRuntime;
using DeepSigma.AI.ONNX.Internal;

namespace DeepSigma.AI.ONNX;

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

    public Tensor<T> Get<T>(string name) where T : unmanaged
    {
        ThrowIfDisposed();
        if (!_outputs.TryGetValue(name, out OrtValue? value))
            throw new OnnxException($"No output named '{name}'. Available: {string.Join(", ", Names)}");
        return TensorMarshaller.FromOrtValue<T>(value);
    }

    public bool TryGet<T>(string name, out Tensor<T>? tensor) where T : unmanaged
    {
        ThrowIfDisposed();
        if (!_outputs.ContainsKey(name))
        {
            tensor = null;
            return false;
        }
        tensor = Get<T>(name);
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
