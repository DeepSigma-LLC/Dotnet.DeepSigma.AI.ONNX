using Microsoft.ML.OnnxRuntime;
using DeepSigma.AI.ONNX.Internal;

namespace DeepSigma.AI.ONNX;

/// <summary>
/// A loaded ONNX model ready for inference. Wraps an ORT InferenceSession.
/// Always dispose to release native resources.
/// </summary>
public sealed class OnnxModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private bool _disposed;

    /// <summary>Input tensors expected by the model, in graph order.</summary>
    public IReadOnlyList<TensorSpec> Inputs { get; }

    /// <summary>Output tensors produced by the model, in graph order.</summary>
    public IReadOnlyList<TensorSpec> Outputs { get; }

    /// <summary>Producer, graph, version, and any custom metadata embedded in the model file.</summary>
    public ModelMetadata Metadata { get; }

    private OnnxModel(InferenceSession session, SessionOptions sessionOptions)
    {
        _session = session;
        _sessionOptions = sessionOptions;
        Inputs = BuildSpecs(session.InputMetadata);
        Outputs = BuildSpecs(session.OutputMetadata);
        Metadata = BuildMetadata(session.ModelMetadata);
    }

    /// <summary>Load a model from a file path on disk.</summary>
    /// <exception cref="OnnxException">Thrown if the file is missing or the model fails to parse.</exception>
    public static OnnxModel Load(string path, ModelOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new OnnxException($"Model file not found: {path}");

        return LoadCore(so => new InferenceSession(path, so), $"file '{path}'", options);
    }

    /// <summary>Load a model from a serialized ONNX byte array.</summary>
    /// <exception cref="OnnxException">Thrown if the bytes don't parse as a valid model.</exception>
    public static OnnxModel Load(byte[] modelBytes, ModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        if (modelBytes.Length == 0)
            throw new OnnxException("Model byte array is empty.");

        return LoadCore(so => new InferenceSession(modelBytes, so), "byte array", options);
    }

    /// <summary>Load a model by reading bytes from a stream. The stream is read to end but not disposed.</summary>
    public static OnnxModel Load(Stream stream, ModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return Load(ms.ToArray(), options);
    }

    private static OnnxModel LoadCore(
        Func<SessionOptions, InferenceSession> sessionFactory,
        string sourceDescription,
        ModelOptions? options)
    {
        SessionOptions sessionOptions = SessionOptionsBuilder.Build(options);
        try
        {
            InferenceSession session = sessionFactory(sessionOptions);
            return new OnnxModel(session, sessionOptions);
        }
        catch (OnnxRuntimeException ex)
        {
            sessionOptions.Dispose();
            throw new OnnxException($"Failed to load model from {sourceDescription}: {ex.Message}", ex);
        }
        catch
        {
            sessionOptions.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Run inference with multi-input / multi-output. Caller must dispose the returned result
    /// to release native output tensors.
    /// </summary>
    /// <exception cref="OnnxException">Thrown if the runtime rejects the inputs.</exception>
    public InferenceResult Run(InferenceInput input)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(input);

        InputValidator.Validate(input.Bindings, Inputs);
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

    /// <summary>
    /// Convenience for single-input / single-output models: feeds one tensor and returns
    /// the model's sole output. Throws if the model has more than one output.
    /// </summary>
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

    /// <summary>
    /// Write a human-readable summary of the model (producer, graph, inputs, outputs)
    /// to <paramref name="writer"/>. Defaults to <see cref="Console.Out"/>.
    /// </summary>
    public void Describe(TextWriter? writer = null)
    {
        writer ??= Console.Out;
        if (!string.IsNullOrEmpty(Metadata.ProducerName))
            writer.WriteLine($"  Producer: {Metadata.ProducerName}");
        if (!string.IsNullOrEmpty(Metadata.GraphName))
            writer.WriteLine($"  Graph:    {Metadata.GraphName}");

        writer.WriteLine("  Inputs:");
        foreach (TensorSpec spec in Inputs)
            writer.WriteLine($"    - {spec.Name}: {spec.ElementType} [{FormatDims(spec.Dimensions)}]");

        writer.WriteLine("  Outputs:");
        foreach (TensorSpec spec in Outputs)
            writer.WriteLine($"    - {spec.Name}: {spec.ElementType} [{FormatDims(spec.Dimensions)}]");
    }

    private static string FormatDims(IReadOnlyList<long?> dims) =>
        string.Join(", ", dims.Select(d => d?.ToString() ?? "?"));

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
