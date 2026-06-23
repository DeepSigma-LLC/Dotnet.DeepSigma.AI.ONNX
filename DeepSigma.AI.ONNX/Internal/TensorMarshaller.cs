using Microsoft.ML.OnnxRuntime;
using Ort = Microsoft.ML.OnnxRuntime.Tensors;

namespace DeepSigma.AI.ONNX.Internal;

internal static class TensorMarshaller
{
    public static OrtValue ToOrtValue<T>(Tensor<T> tensor) where T : unmanaged
    {
        long[] shape = LongShape(tensor.ShapeSpan);
        return OrtValue.CreateTensorValueFromMemory(tensor.Data, shape);
    }

    public static bool ElementTypeMatches<T>(OrtValue value) where T : unmanaged
    {
        var typeShape = value.GetTensorTypeAndShape();
        return ElementTypeMap.FromOrt(typeShape.ElementDataType) == ElementTypeMap.ForClrType<T>();
    }

    public static OrtValue StringTensorToOrtValue(string[] data, ReadOnlySpan<int> shape)
    {
        long[] longShape = LongShape(shape);
        long expected = 1;
        foreach (long d in longShape) expected *= d;
        if (expected != data.Length)
            throw new ArgumentException(
                $"String data length ({data.Length}) does not match product of shape ({expected}).",
                nameof(data));

        OrtValue value = OrtValue.CreateTensorWithEmptyStrings(
            OrtAllocator.DefaultInstance,
            longShape);
        try
        {
            for (int i = 0; i < data.Length; i++)
            {
                value.StringTensorSetElementAt((data[i] ?? string.Empty).AsSpan(), i);
            }
            return value;
        }
        catch
        {
            value.Dispose();
            throw;
        }
    }

    public static Tensor<T> FromOrtValue<T>(OrtValue value) where T : unmanaged
    {
        var typeShape = value.GetTensorTypeAndShape();
        TensorElementType actual = ElementTypeMap.FromOrt(typeShape.ElementDataType);
        TensorElementType expected = ElementTypeMap.ForClrType<T>();
        if (actual != expected)
            throw new OnnxException(
                $"Tensor element type mismatch: model output is {actual}, requested {expected}.");

        ReadOnlySpan<T> span = value.GetTensorDataAsSpan<T>();
        T[] data = span.ToArray();
        int[] shape = IntShape(typeShape.Shape);
        return new Tensor<T>(data, shape);
    }

    public static string[] StringsFromOrtValue(OrtValue value, out int[] shape)
    {
        var typeShape = value.GetTensorTypeAndShape();
        if (typeShape.ElementDataType != Ort.TensorElementType.String)
            throw new OnnxException(
                $"Tensor element type mismatch: model output is {ElementTypeMap.FromOrt(typeShape.ElementDataType)}, requested String.");
        shape = IntShape(typeShape.Shape);

        long count = typeShape.ElementCount;
        if (count > int.MaxValue)
            throw new OnnxException($"String tensor element count {count} exceeds int.MaxValue.");
        var result = new string[count];
        for (int i = 0; i < (int)count; i++)
        {
            result[i] = value.GetStringElement(i);
        }
        return result;
    }

    private static long[] LongShape(ReadOnlySpan<int> shape)
    {
        var result = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++) result[i] = shape[i];
        return result;
    }

    private static int[] IntShape(ReadOnlySpan<long> shape)
    {
        var result = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            long d = shape[i];
            if (d < 0 || d > int.MaxValue)
                throw new OnnxException($"Tensor dimension {d} at index {i} is out of int range.");
            result[i] = (int)d;
        }
        return result;
    }
}
