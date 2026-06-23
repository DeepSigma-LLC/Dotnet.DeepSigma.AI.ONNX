using DeepSigma.AI.ONNX.Demo;
using Xunit;

namespace DeepSigma.AI.ONNX.Test;

public class InferenceTests
{
    [Fact]
    public void Run_MultiInput_ProducesElementwiseSum()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());

        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, stackalloc int[] { 4 });
        var b = new Tensor<float>(new[] { 10f, 20f, 30f, 40f }, stackalloc int[] { 4 });

        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b);

        using InferenceResult result = model.Run(inputs);
        Tensor<float> sum = result.Get<float>(MinimalAddModel.Output);

        Assert.Equal(new[] { 4 }, sum.Shape);
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, sum.Data);
    }

    [Fact]
    public void Run_RepeatedRunsOnSameModel_ReleaseOutputs()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        int[] shape = { 1 };
        for (int i = 0; i < 50; i++)
        {
            var a = new Tensor<float>(new[] { (float)i }, shape);
            var b = new Tensor<float>(new[] { 1f }, shape);
            var inputs = new InferenceInput()
                .Add(MinimalAddModel.InputA, a)
                .Add(MinimalAddModel.InputB, b);
            using InferenceResult result = model.Run(inputs);
            Tensor<float> sum = result.Get<float>(MinimalAddModel.Output);
            Assert.Equal(i + 1, sum.Data[0]);
        }
    }

    [Fact]
    public void Get_WrongElementType_ThrowsOnnxException()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var a = new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 });
        var b = new Tensor<float>(new[] { 2f }, stackalloc int[] { 1 });
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b);

        using InferenceResult result = model.Run(inputs);
        Assert.Throws<OnnxException>(() => result.Get<int>(MinimalAddModel.Output));
    }

    [Fact]
    public void TryGet_WrongElementType_ReturnsFalseInsteadOfThrowing()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var a = new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 });
        var b = new Tensor<float>(new[] { 2f }, stackalloc int[] { 1 });
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b);

        using InferenceResult result = model.Run(inputs);

        Assert.False(result.TryGet<int>(MinimalAddModel.Output, out Tensor<int>? wrongType));
        Assert.Null(wrongType);

        Assert.False(result.TryGet<float>("not-a-real-output", out Tensor<float>? missing));
        Assert.Null(missing);

        Assert.True(result.TryGet<float>(MinimalAddModel.Output, out Tensor<float>? correct));
        Assert.NotNull(correct);
        Assert.Equal(3f, correct.Data[0]);
    }

    [Fact]
    public void Get_UnknownOutputName_ThrowsOnnxException()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var a = new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 });
        var b = new Tensor<float>(new[] { 2f }, stackalloc int[] { 1 });
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b);

        using InferenceResult result = model.Run(inputs);
        Assert.Throws<OnnxException>(() => result.Get<float>("not-a-real-output"));
    }

    [Fact]
    public void RunSingleInput_OnMultiInputModel_Throws()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var a = new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 });
        // single-Run uses one input; but our Add model needs two — feed only one and expect ORT failure surfaced as OnnxException.
        Assert.Throws<OnnxException>(() => model.Run(MinimalAddModel.InputA, a));
    }

    [Fact]
    public void InferenceResult_AccessAfterDispose_Throws()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var a = new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 });
        var b = new Tensor<float>(new[] { 2f }, stackalloc int[] { 1 });
        InferenceResult result = model.Run(new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b));
        result.Dispose();
        Assert.Throws<ObjectDisposedException>(() => result.Get<float>(MinimalAddModel.Output));
    }
}
