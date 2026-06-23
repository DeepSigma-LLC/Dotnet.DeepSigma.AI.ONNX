namespace DeepSigma.AI.ONNX;

/// <summary>
/// A dense tensor with a flat row-major buffer and a shape. The element type T must be unmanaged
/// (float, double, int, long, byte, etc.). Indexers are provided up to rank 4 (NCHW); for higher
/// ranks, compute the flat offset manually and index into <see cref="Data"/>.
/// </summary>
public sealed class Tensor<T> where T : unmanaged
{
    private readonly int[] _shape;

    /// <summary>The underlying row-major data buffer. Mutating is intentional (zero-copy edits).</summary>
    public T[] Data { get; }

    /// <summary>The tensor's shape, in row-major order.</summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>Number of dimensions.</summary>
    public int Rank => _shape.Length;

    /// <summary>Total element count (product of <see cref="Shape"/>).</summary>
    public int Length => Data.Length;

    public Tensor(T[] data, ReadOnlySpan<int> shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (shape.Length == 0)
            throw new ArgumentException("Shape must have at least one dimension.", nameof(shape));

        long expected = Product(shape);
        if (expected != data.Length)
            throw new ArgumentException(
                $"Data length ({data.Length}) does not match product of shape ({expected}).",
                nameof(data));

        Data = data;
        _shape = shape.ToArray();
    }

    /// <summary>Allocate a zero-filled tensor of the given shape.</summary>
    public static Tensor<T> Zeros(ReadOnlySpan<int> shape) =>
        new(new T[(int)Product(shape)], shape);

    internal ReadOnlySpan<int> ShapeSpan => _shape;

    private static long Product(ReadOnlySpan<int> shape)
    {
        long product = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
                throw new ArgumentException(
                    $"Shape dimensions must be non-negative; got {shape[i]} at index {i}.", nameof(shape));
            product *= shape[i];
        }
        return product;
    }

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
            return Data[i * _shape[1] + j];
        }
        set
        {
            CheckRank(2);
            Data[i * _shape[1] + j] = value;
        }
    }

    public T this[int i, int j, int k]
    {
        get
        {
            CheckRank(3);
            return Data[(i * _shape[1] + j) * _shape[2] + k];
        }
        set
        {
            CheckRank(3);
            Data[(i * _shape[1] + j) * _shape[2] + k] = value;
        }
    }

    public T this[int n, int c, int h, int w]
    {
        get
        {
            CheckRank(4);
            return Data[((n * _shape[1] + c) * _shape[2] + h) * _shape[3] + w];
        }
        set
        {
            CheckRank(4);
            Data[((n * _shape[1] + c) * _shape[2] + h) * _shape[3] + w] = value;
        }
    }

    private void CheckRank(int expected)
    {
        if (Rank != expected)
            throw new InvalidOperationException(
                $"Indexer requires rank-{expected} tensor; this tensor has rank {Rank}.");
    }
}
