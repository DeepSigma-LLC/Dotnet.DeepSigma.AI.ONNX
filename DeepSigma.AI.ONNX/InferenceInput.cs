using Microsoft.ML.OnnxRuntime;
using DeepSigma.AI.ONNX.Internal;

namespace DeepSigma.AI.ONNX;

/// <summary>
/// Builder for the inputs of a single inference call. Chain <c>Add</c> calls per model input,
/// then pass to <see cref="OnnxModel.Run(InferenceInput)"/>.
/// </summary>
public sealed class InferenceInput
{
    private readonly List<InputBinding> _bindings = new();

    /// <summary>Names of inputs added so far, in insertion order.</summary>
    public IReadOnlyList<string> Names => _bindings.ConvertAll(b => b.Name);

    public InferenceInput Add<T>(string name, Tensor<T> tensor) where T : unmanaged
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(tensor);
        _bindings.Add(new InputBinding(
            name,
            ElementTypeMap.ForClrType<T>(),
            tensor.Shape.ToArray(),
            () => TensorMarshaller.ToOrtValue(tensor)));
        return this;
    }

    public InferenceInput AddString(string name, string[] data, ReadOnlySpan<int> shape)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(data);
        int[] shapeCopy = shape.ToArray();
        _bindings.Add(new InputBinding(
            name,
            TensorElementType.String,
            shapeCopy,
            () => TensorMarshaller.StringTensorToOrtValue(data, shapeCopy)));
        return this;
    }

    internal IReadOnlyList<InputBinding> Bindings => _bindings;

    internal Dictionary<string, OrtValue> Materialize()
    {
        var result = new Dictionary<string, OrtValue>(_bindings.Count, StringComparer.Ordinal);
        try
        {
            foreach (InputBinding binding in _bindings)
            {
                result[binding.Name] = binding.Materialize();
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
