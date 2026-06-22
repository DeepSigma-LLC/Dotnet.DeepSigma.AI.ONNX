using DeepSigma.AI.ONNX.Demo;
using Xunit;

namespace DeepSigma.AI.ONNX.Test;

public class ModelOptionsTests
{
    [Fact]
    public void Defaults_AreSensible()
    {
        var options = new ModelOptions();
        Assert.Equal(ExecutionProvider.Cpu, options.Provider);
        Assert.Equal(0, options.DeviceId);
        Assert.Equal(GraphOptimizationLevel.All, options.Optimization);
        Assert.Equal(LogSeverity.Warning, options.LogLevel);
        Assert.True(options.EnableMemoryPattern);
        Assert.True(options.EnableCpuMemArena);
        Assert.Null(options.InterOpThreads);
        Assert.Null(options.IntraOpThreads);
        Assert.Null(options.Configure);
    }

    [Fact]
    public void ConfigureHook_IsInvokedDuringLoad()
    {
        bool invoked = false;
        var options = new ModelOptions
        {
            Configure = so => invoked = true,
        };
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create(), options);
        Assert.True(invoked);
    }

    [Fact]
    public void CustomOptimizationLevel_LoadsModel()
    {
        var options = new ModelOptions
        {
            Optimization = GraphOptimizationLevel.Basic,
            LogLevel = LogSeverity.Error,
            IntraOpThreads = 1,
        };
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create(), options);
        Assert.NotNull(model);
    }
}
