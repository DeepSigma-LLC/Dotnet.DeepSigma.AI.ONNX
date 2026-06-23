using DeepSigma.AI.ONNX.Demo;
using DeepSigma.AI.ONNX.Testing;
using Xunit;

namespace DeepSigma.AI.ONNX.Test;

public class TestingAssertionsTests
{
    private static OnnxModel LoadAddModel() => OnnxModel.Load(MinimalAddModel.Create());

    private static InferenceResult RunOnes(OnnxModel model, int n = 3)
    {
        int[] shape = { n };
        var a = new Tensor<float>(Enumerable.Repeat(1f, n).ToArray(), shape);
        var b = new Tensor<float>(Enumerable.Repeat(2f, n).ToArray(), shape);
        return model.Run(new InferenceInput()
            .Add(MinimalAddModel.InputA, a)
            .Add(MinimalAddModel.InputB, b));
    }

    // -- ModelAssertions --

    [Fact]
    public void ShouldHaveInput_Passes()
    {
        using OnnxModel model = LoadAddModel();
        model.ShouldHaveInput(MinimalAddModel.InputA);
        model.ShouldHaveInput(MinimalAddModel.InputA, TensorElementType.Float);
        model.ShouldHaveInput(MinimalAddModel.InputA, TensorElementType.Float, expectedShape: new long?[] { null });
    }

    [Fact]
    public void ShouldHaveInput_MissingThrows()
    {
        using OnnxModel model = LoadAddModel();
        Assert.Throws<OnnxAssertionException>(() => model.ShouldHaveInput("nope"));
    }

    [Fact]
    public void ShouldHaveInput_WrongTypeThrows()
    {
        using OnnxModel model = LoadAddModel();
        Assert.Throws<OnnxAssertionException>(() =>
            model.ShouldHaveInput(MinimalAddModel.InputA, TensorElementType.Int64));
    }

    [Fact]
    public void ShouldHaveOutput_Passes()
    {
        using OnnxModel model = LoadAddModel();
        model.ShouldHaveOutput(MinimalAddModel.Output);
        model.ShouldHaveOutput(MinimalAddModel.Output, TensorElementType.Float);
    }

    // -- ResultAssertions --

    [Fact]
    public void ResultShouldHaveOutput_Passes()
    {
        using OnnxModel model = LoadAddModel();
        using InferenceResult result = RunOnes(model);
        result.ShouldHaveOutput(MinimalAddModel.Output);
        Assert.Throws<OnnxAssertionException>(() => result.ShouldHaveOutput("nope"));
    }

    [Fact]
    public void ResultShouldHaveShape_Passes()
    {
        using OnnxModel model = LoadAddModel();
        using InferenceResult result = RunOnes(model, 5);
        result.ShouldHaveShape(MinimalAddModel.Output, 5);
        Assert.Throws<OnnxAssertionException>(() => result.ShouldHaveShape(MinimalAddModel.Output, 4));
    }

    [Fact]
    public void ShouldNotContainNaN_Passes_WhenClean()
    {
        using OnnxModel model = LoadAddModel();
        using InferenceResult result = RunOnes(model);
        result.ShouldNotContainNaN();
        result.ShouldNotContainNaN(MinimalAddModel.Output);
    }

    [Fact]
    public void ShouldNotContainInfinity_Passes_WhenClean()
    {
        using OnnxModel model = LoadAddModel();
        using InferenceResult result = RunOnes(model);
        result.ShouldNotContainInfinity();
        result.ShouldNotContainInfinity(MinimalAddModel.Output);
    }

    [Fact]
    public void ShouldBeWithinRange_Passes_AndFails()
    {
        using OnnxModel model = LoadAddModel();
        using InferenceResult result = RunOnes(model);  // outputs: 3, 3, 3
        result.ShouldBeWithinRange(MinimalAddModel.Output, 0f, 10f);
        Assert.Throws<OnnxAssertionException>(() =>
            result.ShouldBeWithinRange(MinimalAddModel.Output, 0f, 2f));
    }

    // -- TensorAssertions --

    [Fact]
    public void ShouldBeCloseTo_Passes_WithinTolerance()
    {
        var t = new Tensor<float>(new[] { 1.0f, 2.0f, 3.0f }, stackalloc int[] { 3 });
        t.ShouldBeCloseTo(new[] { 1.0001f, 2.0001f, 3.0001f }, tolerance: 1e-3f);
    }

    [Fact]
    public void ShouldBeCloseTo_FailsOutsideTolerance()
    {
        var t = new Tensor<float>(new[] { 1.0f, 2.0f, 3.0f }, stackalloc int[] { 3 });
        OnnxAssertionException ex = Assert.Throws<OnnxAssertionException>(() =>
            t.ShouldBeCloseTo(new[] { 1.0f, 2.0f, 9.0f }, tolerance: 1e-3f));
        Assert.Contains("index 2", ex.Message);
    }

    // -- SnapshotAssertions --

    [Fact]
    public void SnapshotMissing_WithoutUpdateEnv_Throws()
    {
        using OnnxModel model = LoadAddModel();
        using InferenceResult result = RunOnes(model);

        string path = Path.Combine(Path.GetTempPath(), $"snap-{Guid.NewGuid():N}.json");
        try
        {
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", null);
            OnnxAssertionException ex = Assert.Throws<OnnxAssertionException>(() =>
                result.ShouldMatchJsonSnapshot(path));
            Assert.Contains("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS=1", ex.Message);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void SnapshotRoundTrip_CreatesThenMatches()
    {
        using OnnxModel model = LoadAddModel();
        string path = Path.Combine(Path.GetTempPath(), $"snap-{Guid.NewGuid():N}.json");
        try
        {
            // First run: create snapshot.
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", "1");
            using (InferenceResult first = RunOnes(model))
            {
                first.ShouldMatchJsonSnapshot(path);
            }
            Assert.True(File.Exists(path));

            // Second run: same outputs, should match without update env.
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", null);
            using (InferenceResult second = RunOnes(model))
            {
                second.ShouldMatchJsonSnapshot(path);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", null);
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Snapshot_MismatchedValues_Throws()
    {
        using OnnxModel model = LoadAddModel();
        string path = Path.Combine(Path.GetTempPath(), $"snap-{Guid.NewGuid():N}.json");
        try
        {
            // Capture snapshot with 3+3=6.
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", "1");
            using (InferenceResult first = RunOnes(model))
                first.ShouldMatchJsonSnapshot(path);

            // Run different inputs, expect mismatch.
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", null);
            int[] shape = { 3 };
            using InferenceResult mismatched = model.Run(new InferenceInput()
                .Add(MinimalAddModel.InputA, new Tensor<float>(new[] { 100f, 100f, 100f }, shape))
                .Add(MinimalAddModel.InputB, new Tensor<float>(new[] { 100f, 100f, 100f }, shape)));

            Assert.Throws<OnnxAssertionException>(() => mismatched.ShouldMatchJsonSnapshot(path));
        }
        finally
        {
            Environment.SetEnvironmentVariable("DEEPSIGMA_ONNX_UPDATE_SNAPSHOTS", null);
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
