using Ort = Microsoft.ML.OnnxRuntime.Tensors;

namespace DeepSigma.AI.ONNX.Internal;

internal static class ElementTypeMap
{
    public static TensorElementType FromOrt(Ort.TensorElementType ort) => ort switch
    {
        Ort.TensorElementType.Float => TensorElementType.Float,
        Ort.TensorElementType.UInt8 => TensorElementType.UInt8,
        Ort.TensorElementType.Int8 => TensorElementType.Int8,
        Ort.TensorElementType.UInt16 => TensorElementType.UInt16,
        Ort.TensorElementType.Int16 => TensorElementType.Int16,
        Ort.TensorElementType.Int32 => TensorElementType.Int32,
        Ort.TensorElementType.Int64 => TensorElementType.Int64,
        Ort.TensorElementType.String => TensorElementType.String,
        Ort.TensorElementType.Bool => TensorElementType.Bool,
        Ort.TensorElementType.Float16 => TensorElementType.Float16,
        Ort.TensorElementType.Double => TensorElementType.Double,
        Ort.TensorElementType.UInt32 => TensorElementType.UInt32,
        Ort.TensorElementType.UInt64 => TensorElementType.UInt64,
        Ort.TensorElementType.BFloat16 => TensorElementType.BFloat16,
        _ => TensorElementType.Unknown,
    };

    public static Ort.TensorElementType ToOrt(TensorElementType value) => value switch
    {
        TensorElementType.Float => Ort.TensorElementType.Float,
        TensorElementType.UInt8 => Ort.TensorElementType.UInt8,
        TensorElementType.Int8 => Ort.TensorElementType.Int8,
        TensorElementType.UInt16 => Ort.TensorElementType.UInt16,
        TensorElementType.Int16 => Ort.TensorElementType.Int16,
        TensorElementType.Int32 => Ort.TensorElementType.Int32,
        TensorElementType.Int64 => Ort.TensorElementType.Int64,
        TensorElementType.String => Ort.TensorElementType.String,
        TensorElementType.Bool => Ort.TensorElementType.Bool,
        TensorElementType.Float16 => Ort.TensorElementType.Float16,
        TensorElementType.Double => Ort.TensorElementType.Double,
        TensorElementType.UInt32 => Ort.TensorElementType.UInt32,
        TensorElementType.UInt64 => Ort.TensorElementType.UInt64,
        TensorElementType.BFloat16 => Ort.TensorElementType.BFloat16,
        _ => throw new OnnxException($"Unsupported tensor element type: {value}"),
    };

    public static TensorElementType ForClrType<T>() where T : unmanaged
    {
        Type t = typeof(T);
        if (t == typeof(float)) return TensorElementType.Float;
        if (t == typeof(double)) return TensorElementType.Double;
        if (t == typeof(int)) return TensorElementType.Int32;
        if (t == typeof(long)) return TensorElementType.Int64;
        if (t == typeof(short)) return TensorElementType.Int16;
        if (t == typeof(sbyte)) return TensorElementType.Int8;
        if (t == typeof(byte)) return TensorElementType.UInt8;
        if (t == typeof(ushort)) return TensorElementType.UInt16;
        if (t == typeof(uint)) return TensorElementType.UInt32;
        if (t == typeof(ulong)) return TensorElementType.UInt64;
        if (t == typeof(bool)) return TensorElementType.Bool;
        throw new OnnxException($"CLR type {t.Name} has no ONNX tensor element type mapping.");
    }
}
