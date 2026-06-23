using DeepSigma.AI.ONNX.Demo;
using Xunit;

namespace DeepSigma.AI.ONNX.Test;

public class InputValidationTests
{
    [Fact]
    public void MissingInput_ThrowsNamedError()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 }));

        OnnxException ex = Assert.Throws<OnnxException>(() => model.Run(inputs));
        Assert.Contains("Missing required input", ex.Message);
        Assert.Contains(MinimalAddModel.InputB, ex.Message);
    }

    [Fact]
    public void UnknownInputName_ThrowsNamedError()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 }))
            .Add(MinimalAddModel.InputB, new Tensor<float>(new[] { 2f }, stackalloc int[] { 1 }))
            .Add("not-a-real-input", new Tensor<float>(new[] { 3f }, stackalloc int[] { 1 }));

        OnnxException ex = Assert.Throws<OnnxException>(() => model.Run(inputs));
        Assert.Contains("Unknown input 'not-a-real-input'", ex.Message);
    }

    [Fact]
    public void DuplicateInput_ThrowsNamedError()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, new Tensor<float>(new[] { 1f }, stackalloc int[] { 1 }))
            .Add(MinimalAddModel.InputA, new Tensor<float>(new[] { 99f }, stackalloc int[] { 1 }));
        // (Add overwrites in the dict-style API of a typical builder, but our builder appends.)

        // Actually our Add<T> uses _bindings.Add — so two Adds with same name produce two bindings.
        // The validator catches that.
        OnnxException ex = Assert.Throws<OnnxException>(() => model.Run(inputs));
        Assert.Contains("more than once", ex.Message);
    }

    [Fact]
    public void WrongElementType_ThrowsNamedError()
    {
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        var inputs = new InferenceInput()
            .Add(MinimalAddModel.InputA, new Tensor<int>(new[] { 1 }, stackalloc int[] { 1 }))
            .Add(MinimalAddModel.InputB, new Tensor<float>(new[] { 2f }, stackalloc int[] { 1 }));

        OnnxException ex = Assert.Throws<OnnxException>(() => model.Run(inputs));
        Assert.Contains("element type Float", ex.Message);
        Assert.Contains("got Int32", ex.Message);
    }

    [Fact]
    public void WrongStaticDimension_ThrowsNamedError()
    {
        // MNIST input is fixed [1, 1, 28, 28]. Feed [1, 1, 30, 30].
        string mnistPath = Path.Combine(AppContext.BaseDirectory, "interop", "mnist-12.onnx");
        if (!File.Exists(mnistPath))
        {
            // Skip if MNIST asset not present in this build context.
            return;
        }

        using OnnxModel model = OnnxModel.Load(mnistPath);
        var badInput = new Tensor<float>(new float[1 * 1 * 30 * 30], new[] { 1, 1, 30, 30 });
        var inputs = new InferenceInput().Add(model.Inputs[0].Name, badInput);

        OnnxException ex = Assert.Throws<OnnxException>(() => model.Run(inputs));
        Assert.Contains("dimension 2 expected 28", ex.Message);
    }

    [Fact]
    public void DynamicDimension_AnyValueAccepted()
    {
        // MinimalAddModel has shape [N] — dynamic batch. Any length should pass validation.
        using OnnxModel model = OnnxModel.Load(MinimalAddModel.Create());
        int[] shape = { 7 };
        var a = new Tensor<float>(new float[7], shape);
        var b = new Tensor<float>(new float[7], shape);
        using InferenceResult result = model.Run(new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b));
        Assert.Equal(7, result.Get<float>(MinimalAddModel.Output).Data.Length);
    }
}
