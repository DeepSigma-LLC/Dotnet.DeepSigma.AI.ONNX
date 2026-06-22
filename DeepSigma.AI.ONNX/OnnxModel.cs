using Microsoft.ML.OnnxRuntime;
using DeepSigma.AI.ONNX.Internal;

namespace DeepSigma.AI.ONNX;

public sealed class OnnxModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private bool _disposed;

    public IReadOnlyList<TensorSpec> Inputs { get; }
    public IReadOnlyList<TensorSpec> Outputs { get; }
    public ModelMetadata Metadata { get; }

    private OnnxModel(InferenceSession session, SessionOptions sessionOptions)
    {
        _session = session;
        _sessionOptions = sessionOptions;
        Inputs = BuildSpecs(session.InputMetadata);
        Outputs = BuildSpecs(session.OutputMetadata);
        Metadata = BuildMetadata(session.ModelMetadata);
    }

    public static OnnxModel Load(string path, ModelOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new OnnxException($"Model file not found: {path}");

        SessionOptions sessionOptions = SessionOptionsBuilder.Build(options);
        try
        {
            var session = new InferenceSession(path, sessionOptions);
            return new OnnxModel(session, sessionOptions);
        }
        catch (OnnxRuntimeException ex)
        {
            sessionOptions.Dispose();
            throw new OnnxException($"Failed to load model from '{path}': {ex.Message}", ex);
        }
        catch
        {
            sessionOptions.Dispose();
            throw;
        }
    }

    public static OnnxModel Load(byte[] modelBytes, ModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        if (modelBytes.Length == 0)
            throw new OnnxException("Model byte array is empty.");

        SessionOptions sessionOptions = SessionOptionsBuilder.Build(options);
        try
        {
            var session = new InferenceSession(modelBytes, sessionOptions);
            return new OnnxModel(session, sessionOptions);
        }
        catch (OnnxRuntimeException ex)
        {
            sessionOptions.Dispose();
            throw new OnnxException($"Failed to load model from bytes: {ex.Message}", ex);
        }
        catch
        {
            sessionOptions.Dispose();
            throw;
        }
    }

    public static OnnxModel Load(Stream stream, ModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return Load(ms.ToArray(), options);
    }

    public InferenceResult Run(InferenceInput input)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(input);

        Dictionary<string, OrtValue> inputs = input.Materialize();
        try
        {
            string[] outputNames = new string[Outputs.Count];
            for (int i = 0; i < Outputs.Count; i++) outputNames[i] = Outputs[i].Name;

            using var runOptions = new RunOptions();
            IDisposableReadOnlyCollection<OrtValue> ortOutputs;
            try
            {
                ortOutputs = _session.Run(runOptions, inputs, outputNames);
            }
            catch (OnnxRuntimeException ex)
            {
                throw new OnnxException($"Inference failed: {ex.Message}", ex);
            }

            var values = new List<OrtValue>(ortOutputs.Count);
            values.AddRange(ortOutputs);
            return new InferenceResult(outputNames, values);
        }
        finally
        {
            foreach (var v in inputs.Values) v.Dispose();
        }
    }

    public Tensor<T> Run<T>(string inputName, Tensor<T> input) where T : unmanaged
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrEmpty(inputName);
        ArgumentNullException.ThrowIfNull(input);

        if (Outputs.Count != 1)
            throw new OnnxException(
                $"Single-output Run<T> requires a model with exactly one output; this model has {Outputs.Count}. Use Run(InferenceInput) instead.");

        var inferenceInput = new InferenceInput().Add(inputName, input);
        using InferenceResult result = Run(inferenceInput);
        return result.Get<T>(Outputs[0].Name);
    }

    public float[] Predict(float[] features)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(features);

        if (Inputs.Count != 1)
            throw new OnnxException(
                $"Predict requires a model with exactly one input; this model has {Inputs.Count}.");
        if (Outputs.Count != 1)
            throw new OnnxException(
                $"Predict requires a model with exactly one output; this model has {Outputs.Count}.");
        if (Inputs[0].ElementType != TensorElementType.Float)
            throw new OnnxException(
                $"Predict requires a Float input; model input '{Inputs[0].Name}' is {Inputs[0].ElementType}.");
        if (Outputs[0].ElementType != TensorElementType.Float)
            throw new OnnxException(
                $"Predict requires a Float output; model output '{Outputs[0].Name}' is {Outputs[0].ElementType}.");

        var input = new Tensor<float>(features, new[] { 1, features.Length });
        Tensor<float> output = Run(Inputs[0].Name, input);
        return output.Data;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _sessionOptions.Dispose();
        _disposed = true;
    }

    private static IReadOnlyList<TensorSpec> BuildSpecs(IReadOnlyDictionary<string, NodeMetadata> metadata)
    {
        var list = new List<TensorSpec>(metadata.Count);
        foreach (var (name, node) in metadata)
        {
            var dims = new long?[node.Dimensions.Length];
            for (int i = 0; i < node.Dimensions.Length; i++)
            {
                int d = node.Dimensions[i];
                dims[i] = d < 0 ? (long?)null : d;
            }
            TensorElementType elem = ElementTypeMap.FromOrt(node.ElementDataType);
            list.Add(new TensorSpec(name, elem, dims));
        }
        return list;
    }

    private static ModelMetadata BuildMetadata(Microsoft.ML.OnnxRuntime.ModelMetadata m)
    {
        return new ModelMetadata(
            ProducerName: m.ProducerName ?? string.Empty,
            GraphName: m.GraphName ?? string.Empty,
            Domain: m.Domain ?? string.Empty,
            Description: m.Description ?? string.Empty,
            Version: m.Version,
            CustomMetadata: m.CustomMetadataMap ?? new Dictionary<string, string>());
    }
}
