using Xunit;

namespace DeepSigma.AI.ONNX.Test;

public class TensorTests
{
    [Fact]
    public void Constructor_ValidatesDataLengthAgainstShape()
    {
        Assert.Throws<ArgumentException>(() =>
            new Tensor<float>(new float[5], stackalloc int[] { 2, 3 }));
    }

    [Fact]
    public void Constructor_RejectsNegativeDimensions()
    {
        Assert.Throws<ArgumentException>(() =>
            new Tensor<float>(new float[0], stackalloc int[] { -1, 2 }));
    }

    [Fact]
    public void Zeros_AllocatesCorrectLength()
    {
        Tensor<float> t = Tensor<float>.Zeros(stackalloc int[] { 2, 3 });
        Assert.Equal(6, t.Length);
        Assert.Equal(2, t.Rank);
        Assert.All(t.Data, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Indexer_2D_ComputesRowMajorOffset()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, stackalloc int[] { 2, 3 });
        Assert.Equal(1, t[0, 0]);
        Assert.Equal(3, t[0, 2]);
        Assert.Equal(4, t[1, 0]);
        Assert.Equal(6, t[1, 2]);
    }

    [Fact]
    public void Indexer_4D_ComputesNCHWOffset()
    {
        var t = Tensor<float>.Zeros(stackalloc int[] { 1, 3, 4, 5 });
        t[0, 1, 2, 3] = 42f;
        int expectedFlat = ((0 * 3 + 1) * 4 + 2) * 5 + 3;
        Assert.Equal(42f, t.Data[expectedFlat]);
    }

    [Fact]
    public void Indexer_WrongRank_Throws()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3 }, stackalloc int[] { 3 });
        Assert.Throws<InvalidOperationException>(() => t[0, 0]);
    }

    [Fact]
    public void DataSpan_RoundTrips()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3 }, stackalloc int[] { 3 });
        Span<int> span = t.Data.AsSpan();
        span[1] = 99;
        Assert.Equal(99, t.Data[1]);
    }

    [Fact]
    public void Shape_IsReadOnly()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3, 4 }, stackalloc int[] { 2, 2 });
        Assert.IsAssignableFrom<IReadOnlyList<int>>(t.Shape);
        Assert.Equal(new[] { 2, 2 }, t.Shape);
    }
}
