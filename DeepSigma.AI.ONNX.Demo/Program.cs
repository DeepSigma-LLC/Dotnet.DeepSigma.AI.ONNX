using DeepSigma.AI.ONNX;
using DeepSigma.AI.ONNX.Demo;

Console.WriteLine("=== DeepSigma.AI.ONNX — demo ===");
Console.WriteLine();
Console.WriteLine("[1] Built-in minimal Add model");
Console.WriteLine("-------------------------------");

byte[] modelBytes = MinimalAddModel.Create();
Console.WriteLine($"Generated minimal Add model: {modelBytes.Length} bytes");

using (OnnxModel addModel = OnnxModel.Load(modelBytes))
{
    Console.WriteLine($"Producer: {addModel.Metadata.ProducerName}");
    Console.WriteLine($"Graph:    {addModel.Metadata.GraphName}");

    Console.WriteLine("Inputs:");
    foreach (TensorSpec input in addModel.Inputs)
    {
        Console.WriteLine($"  - {input.Name}: {input.ElementType} [{FormatDims(input.Dimensions)}]");
    }

    Console.WriteLine("Outputs:");
    foreach (TensorSpec output in addModel.Outputs)
    {
        Console.WriteLine($"  - {output.Name}: {output.ElementType} [{FormatDims(output.Dimensions)}]");
    }

    var a = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, stackalloc int[] { 4 });
    var b = new Tensor<float>(new float[] { 10f, 20f, 30f, 40f }, stackalloc int[] { 4 });

    var inputs = new InferenceInput()
        .Add(MinimalAddModel.InputA, a)
        .Add(MinimalAddModel.InputB, b);

    using InferenceResult result = addModel.Run(inputs);
    Tensor<float> sum = result.Get<float>(MinimalAddModel.Output);
    Console.WriteLine($"a + b = [{string.Join(", ", sum.Data)}]");
}

Console.WriteLine();
Console.WriteLine("[2] Python → ONNX interop (sklearn LogisticRegression on Iris)");
Console.WriteLine("---------------------------------------------------------------");
IrisInteropDemo.Run();

Console.WriteLine();
Console.WriteLine("[3] Open-source model from the ONNX Model Zoo (MNIST CNN)");
Console.WriteLine("----------------------------------------------------------");
MnistDemo.Run();

Console.WriteLine();
Console.WriteLine("Demo complete.");

static string FormatDims(IReadOnlyList<long?> dims) =>
    string.Join(", ", dims.Select(d => d?.ToString() ?? "?"));
