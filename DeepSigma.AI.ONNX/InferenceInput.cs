using Microsoft.ML.OnnxRuntime;
using DeepSigma.AI.ONNX.Internal;

namespace DeepSigma.AI.ONNX;

public sealed class InferenceInput
{
    private readonly Dictionary<string, Func<OrtValue>> _factories = new(StringComparer.Ordinal);

    public IReadOnlyCollection<string> Names => _factories.Keys;

    public InferenceInput Add<T>(string name, Tensor<T> tensor) where T : unmanaged
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(tensor);
        _factories[name] = () => TensorMarshaller.ToOrtValue(tensor);
        return this;
    }

    public InferenceInput AddString(string name, string[] data, ReadOnlySpan<int> shape)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(data);
        int[] shapeCopy = shape.ToArray();
        _factories[name] = () => TensorMarshaller.StringTensorToOrtValue(data, shapeCopy);
        return this;
    }

    internal Dictionary<string, OrtValue> Materialize()
    {
        var result = new Dictionary<string, OrtValue>(_factories.Count, StringComparer.Ordinal);
        try
        {
            foreach (var (name, factory) in _factories)
            {
                result[name] = factory();
            }
            return result;
        }
        catch
        {
            foreach (var v in result.Values) v.Dispose();
            throw;
        }
    }
}
