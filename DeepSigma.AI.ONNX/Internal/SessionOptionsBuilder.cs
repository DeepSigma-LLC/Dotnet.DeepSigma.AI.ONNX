using Microsoft.ML.OnnxRuntime;
using OrtGraphOpt = Microsoft.ML.OnnxRuntime.GraphOptimizationLevel;
using OrtLogLevel = Microsoft.ML.OnnxRuntime.OrtLoggingLevel;

namespace DeepSigma.AI.ONNX.Internal;

internal static class SessionOptionsBuilder
{
    public static SessionOptions Build(ModelOptions? options)
    {
        options ??= new ModelOptions();

        SessionOptions sessionOptions;
        try
        {
            sessionOptions = options.Provider switch
            {
                ExecutionProvider.Cpu => new SessionOptions(),
                ExecutionProvider.Cuda => SessionOptions.MakeSessionOptionWithCudaProvider(options.DeviceId),
                _ => throw new OnnxException($"Unknown execution provider: {options.Provider}"),
            };
        }
        catch (OnnxRuntimeException ex)
        {
            throw new OnnxException(
                $"Failed to initialize execution provider '{options.Provider}'. " +
                "For CUDA, ensure Microsoft.ML.OnnxRuntime.Gpu and the CUDA runtime are installed.",
                ex);
        }

        try
        {
            sessionOptions.GraphOptimizationLevel = MapOpt(options.Optimization);
            sessionOptions.LogSeverityLevel = MapLog(options.LogLevel);
            sessionOptions.EnableMemoryPattern = options.EnableMemoryPattern;
            sessionOptions.EnableCpuMemArena = options.EnableCpuMemArena;
            if (options.InterOpThreads is int inter) sessionOptions.InterOpNumThreads = inter;
            if (options.IntraOpThreads is int intra) sessionOptions.IntraOpNumThreads = intra;

            options.Configure?.Invoke(sessionOptions);
        }
        catch
        {
            sessionOptions.Dispose();
            throw;
        }

        return sessionOptions;
    }

    private static OrtGraphOpt MapOpt(GraphOptimizationLevel level) => level switch
    {
        GraphOptimizationLevel.Disabled => OrtGraphOpt.ORT_DISABLE_ALL,
        GraphOptimizationLevel.Basic => OrtGraphOpt.ORT_ENABLE_BASIC,
        GraphOptimizationLevel.Extended => OrtGraphOpt.ORT_ENABLE_EXTENDED,
        GraphOptimizationLevel.All => OrtGraphOpt.ORT_ENABLE_ALL,
        _ => OrtGraphOpt.ORT_ENABLE_ALL,
    };

    private static OrtLogLevel MapLog(LogSeverity severity) => severity switch
    {
        LogSeverity.Verbose => OrtLogLevel.ORT_LOGGING_LEVEL_VERBOSE,
        LogSeverity.Info => OrtLogLevel.ORT_LOGGING_LEVEL_INFO,
        LogSeverity.Warning => OrtLogLevel.ORT_LOGGING_LEVEL_WARNING,
        LogSeverity.Error => OrtLogLevel.ORT_LOGGING_LEVEL_ERROR,
        LogSeverity.Fatal => OrtLogLevel.ORT_LOGGING_LEVEL_FATAL,
        _ => OrtLogLevel.ORT_LOGGING_LEVEL_WARNING,
    };
}
