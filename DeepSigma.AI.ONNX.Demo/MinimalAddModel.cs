namespace DeepSigma.AI.ONNX.Demo;

/// <summary>
/// Hand-crafts a tiny ONNX model (Add op, two float[N] inputs → one float[N] output)
/// by emitting raw protobuf bytes. Used by the demo and test fixtures so we don't
/// need to ship a binary model file.
/// </summary>
internal static class MinimalAddModel
{
    public const string InputA = "a";
    public const string InputB = "b";
    public const string Output = "c";

    public static byte[] Create()
    {
        // Build the GraphProto first, then wrap in ModelProto.
        byte[] node = EncodeNode();
        byte[] inputA = EncodeValueInfo(InputA);
        byte[] inputB = EncodeValueInfo(InputB);
        byte[] outputC = EncodeValueInfo(Output);

        var graph = new ProtoWriter();
        graph.WriteLengthDelimited(1, node);          // node[]
        graph.WriteString(2, "MinimalAddGraph");      // name
        graph.WriteLengthDelimited(11, inputA);       // input[]
        graph.WriteLengthDelimited(11, inputB);       // input[]
        graph.WriteLengthDelimited(12, outputC);      // output[]

        byte[] opset = EncodeOpsetImport(string.Empty, 13);

        var model = new ProtoWriter();
        model.WriteVarint(1, 7);                       // ir_version
        model.WriteString(2, "DeepSigma.AI.ONNX");     // producer_name
        model.WriteString(3, "1.0");                   // producer_version
        model.WriteVarint(5, 1);                       // model_version
        model.WriteLengthDelimited(7, graph.ToArray()); // graph
        model.WriteLengthDelimited(8, opset);          // opset_import[]

        return model.ToArray();
    }

    private static byte[] EncodeNode()
    {
        var w = new ProtoWriter();
        w.WriteString(1, InputA);    // input[]
        w.WriteString(1, InputB);    // input[]
        w.WriteString(2, Output);    // output[]
        w.WriteString(3, "AddNode"); // name
        w.WriteString(4, "Add");     // op_type
        return w.ToArray();
    }

    private static byte[] EncodeValueInfo(string name)
    {
        // ValueInfoProto: name (1), type (2: TypeProto)
        // TypeProto: tensor_type (1: TypeProto.Tensor)
        // TypeProto.Tensor: elem_type (1) = 1 (FLOAT), shape (2: TensorShapeProto)
        // TensorShapeProto: dim (1: Dimension)
        // Dimension: dim_param (2) = "N"  (dynamic axis)
        var dim = new ProtoWriter();
        dim.WriteString(2, "N");

        var shape = new ProtoWriter();
        shape.WriteLengthDelimited(1, dim.ToArray());

        var tensor = new ProtoWriter();
        tensor.WriteVarint(1, 1);                       // FLOAT
        tensor.WriteLengthDelimited(2, shape.ToArray());

        var type = new ProtoWriter();
        type.WriteLengthDelimited(1, tensor.ToArray()); // tensor_type

        var vi = new ProtoWriter();
        vi.WriteString(1, name);
        vi.WriteLengthDelimited(2, type.ToArray());
        return vi.ToArray();
    }

    private static byte[] EncodeOpsetImport(string domain, long version)
    {
        var w = new ProtoWriter();
        if (!string.IsNullOrEmpty(domain)) w.WriteString(1, domain);
        w.WriteVarint(2, version);
        return w.ToArray();
    }

    private sealed class ProtoWriter
    {
        private readonly MemoryStream _ms = new();

        public byte[] ToArray() => _ms.ToArray();

        public void WriteVarint(int fieldNumber, long value)
        {
            WriteTag(fieldNumber, 0);
            WriteRawVarint((ulong)value);
        }

        public void WriteString(int fieldNumber, string value)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
            WriteLengthDelimited(fieldNumber, bytes);
        }

        public void WriteLengthDelimited(int fieldNumber, byte[] bytes)
        {
            WriteTag(fieldNumber, 2);
            WriteRawVarint((ulong)bytes.Length);
            _ms.Write(bytes, 0, bytes.Length);
        }

        private void WriteTag(int fieldNumber, int wireType)
        {
            WriteRawVarint(((ulong)fieldNumber << 3) | (uint)wireType);
        }

        private void WriteRawVarint(ulong value)
        {
            while (value >= 0x80)
            {
                _ms.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }
            _ms.WriteByte((byte)value);
        }
    }
}
