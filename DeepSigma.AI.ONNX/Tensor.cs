namespace DeepSigma.AI.ONNX;

public sealed class Tensor<T> where T : unmanaged
{
    public T[] Data { get; }
    public int[] Shape { get; }

    public int Rank => Shape.Length;
    public int Length => Data.Length;

    public Tensor(T[] data, ReadOnlySpan<int> shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (shape.Length == 0)
            throw new ArgumentException("Shape must have at least one dimension.", nameof(shape));

        long expected = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
                throw new ArgumentException($"Shape dimensions must be non-negative; got {shape[i]} at index {i}.", nameof(shape));
            expected *= shape[i];
        }
        if (expected != data.Length)
            throw new ArgumentException(
                $"Data length ({data.Length}) does not match product of shape ({expected}).",
                nameof(data));

        Data = data;
        Shape = shape.ToArray();
    }

    public static Tensor<T> Zeros(ReadOnlySpan<int> shape)
    {
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        return new Tensor<T>(new T[length], shape);
    }

    public static Tensor<T> From(T[] data, ReadOnlySpan<int> shape) => new(data, shape);

    public Span<T> AsSpan() => Data.AsSpan();
    public ReadOnlySpan<T> AsReadOnlySpan() => Data.AsSpan();

    public T this[int i]
    {
        get
        {
            CheckRank(1);
            return Data[i];
        }
        set
        {
            CheckRank(1);
            Data[i] = value;
        }
    }

    public T this[int i, int j]
    {
        get
        {
            CheckRank(2);
            return Data[i * Shape[1] + j];
        }
        set
        {
            CheckRank(2);
            Data[i * Shape[1] + j] = value;
        }
    }

    public T this[int i, int j, int k]
    {
        get
        {
            CheckRank(3);
            return Data[(i * Shape[1] + j) * Shape[2] + k];
        }
        set
        {
            CheckRank(3);
            Data[(i * Shape[1] + j) * Shape[2] + k] = value;
        }
    }

    public T this[int n, int c, int h, int w]
    {
        get
        {
            CheckRank(4);
            return Data[((n * Shape[1] + c) * Shape[2] + h) * Shape[3] + w];
        }
        set
        {
            CheckRank(4);
            Data[((n * Shape[1] + c) * Shape[2] + h) * Shape[3] + w] = value;
        }
    }

    private void CheckRank(int expected)
    {
        if (Rank != expected)
            throw new InvalidOperationException(
                $"Indexer requires rank-{expected} tensor; this tensor has rank {Rank}.");
    }
}
