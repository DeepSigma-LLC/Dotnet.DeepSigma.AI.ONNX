using DeepSigma.AI.ONNX.Demo;
using Xunit;

namespace DeepSigma.AI.ONNX.Test;

public class OnnxModelTests
{
    [Fact]
    public void Load_FromBytes_ExposesInputsOutputsMetadata()
    {
        byte[] bytes = MinimalAddModel.Create();
        using OnnxModel model = OnnxModel.Load(bytes);

        Assert.Equal(2, model.Inputs.Count);
        Assert.Single(model.Outputs);
        Assert.Equal(MinimalAddModel.InputA, model.Inputs[0].Name);
        Assert.Equal(MinimalAddModel.InputB, model.Inputs[1].Name);
        Assert.Equal(MinimalAddModel.Output, model.Outputs[0].Name);
        Assert.All(model.Inputs, i => Assert.Equal(TensorElementType.Float, i.ElementType));
        Assert.Equal(TensorElementType.Float, model.Outputs[0].ElementType);
        Assert.Equal("DeepSigma.AI.ONNX", model.Metadata.ProducerName);
    }

    [Fact]
    public void Load_FromStream_Works()
    {
        byte[] bytes = MinimalAddModel.Create();
        using var ms = new MemoryStream(bytes);
        using OnnxModel model = OnnxModel.Load(ms);
        Assert.Equal(2, model.Inputs.Count);
    }

    [Fact]
    public void Load_FromPath_Works()
    {
        byte[] bytes = MinimalAddModel.Create();
        string path = Path.Combine(Path.GetTempPath(), $"deepsigma-onnx-test-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, bytes);
        try
        {
            using OnnxModel model = OnnxModel.Load(path);
            Assert.Equal(2, model.Inputs.Count);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_MissingPath_Throws()
    {
        Assert.Throws<OnnxException>(() =>
            OnnxModel.Load(Path.Combine(Path.GetTempPath(), "does-not-exist.onnx")));
    }

    [Fact]
    public void Load_EmptyBytes_Throws()
    {
        Assert.Throws<OnnxException>(() => OnnxModel.Load(Array.Empty<byte>()));
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        model.Dispose();
        model.Dispose();
    }

    [Fact]
    public void DynamicDimensions_ReportedAsNull()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        Assert.Null(model.Inputs[0].Dimensions[0]);
        Assert.Null(model.Outputs[0].Dimensions[0]);
    }
}
