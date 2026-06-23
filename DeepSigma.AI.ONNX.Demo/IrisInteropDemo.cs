using System.Text.Json;
using System.Text.Json.Serialization;

namespace DeepSigma.AI.ONNX.Demo;

internal static class IrisInteropDemo
{
    public static void Run()
    {
        string modelPath = Path.Combine(AppContext.BaseDirectory, "interop", "iris_logreg.onnx");
        string referencePath = Path.Combine(AppContext.BaseDirectory, "interop", "iris_reference.json");

        if (!File.Exists(modelPath) || !File.Exists(referencePath))
        {
            Console.WriteLine("Iris interop assets not found. Run interop/export_iris_model.py first.");
            return;
        }

        IrisReference reference = LoadReference(referencePath);

        Console.WriteLine("Loading iris_logreg.onnx (trained + exported by Python skl2onnx)...");
        using OnnxModel model = OnnxModel.Load(modelPath);
        model.Describe();

        // Build a [N, 4] float tensor from the reference samples.
        int n = reference.Samples.Count;
        var flat = new float[n * 4];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                flat[i * 4 + j] = (float)reference.Samples[i][j];
            }
        }
        var input = new Tensor<float>(flat, new[] { n, 4 });

        var inputs = new InferenceInput().Add(model.Inputs[0].Name, input);
        using InferenceResult result = model.Run(inputs);

        Tensor<long> labels = result.Get<long>("label");
        Tensor<float> probabilities = result.Get<float>("probabilities");

        Console.WriteLine();
        Console.WriteLine($"{"sample",-30} {"py-label",10} {"net-label",10} {"py-probs",-30} {"net-probs",-30} {"match"}");
        Console.WriteLine(new string('-', 130));

        int mismatches = 0;
        for (int i = 0; i < n; i++)
        {
            long netLabel = labels[i];
            int pyLabel = reference.ExpectedLabels[i];

            float[] netProbs = new float[3];
            for (int c = 0; c < 3; c++) netProbs[c] = probabilities[i, c];
            double[] pyProbs = reference.ExpectedProbabilities[i].ToArray();

            bool labelMatch = netLabel == pyLabel;
            bool probsMatch = ProbabilitiesClose(netProbs, pyProbs, tolerance: 1e-5);
            bool match = labelMatch && probsMatch;
            if (!match) mismatches++;

            string sampleStr = "[" + string.Join(", ", reference.Samples[i].Select(v => v.ToString("F2"))) + "]";
            string pyProbsStr = "[" + string.Join(", ", pyProbs.Select(p => p.ToString("F4"))) + "]";
            string netProbsStr = "[" + string.Join(", ", netProbs.Select(p => p.ToString("F4"))) + "]";
            Console.WriteLine($"{sampleStr,-30} {pyLabel,10} {netLabel,10} {pyProbsStr,-30} {netProbsStr,-30} {(match ? "OK" : "MISMATCH")}");
        }

        Console.WriteLine();
        if (mismatches == 0)
        {
            Console.WriteLine($"All {n} samples match Python predictions (labels + probabilities within 1e-5).");
        }
        else
        {
            Console.WriteLine($"{mismatches} of {n} samples disagreed with Python predictions.");
            Environment.ExitCode = 1;
        }
    }

    private static bool ProbabilitiesClose(float[] net, double[] py, double tolerance)
    {
        if (net.Length != py.Length) return false;
        for (int i = 0; i < net.Length; i++)
        {
            if (Math.Abs(net[i] - py[i]) > tolerance) return false;
        }
        return true;
    }

    private static IrisReference LoadReference(string path)
    {
        string json = File.ReadAllText(path);
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        };
        return JsonSerializer.Deserialize<IrisReference>(json, options)
            ?? throw new InvalidDataException("Failed to deserialize iris reference.");
    }

    private sealed class IrisReference
    {
        [JsonPropertyName("feature_names")]
        public List<string> FeatureNames { get; set; } = new();

        [JsonPropertyName("class_names")]
        public List<string> ClassNames { get; set; } = new();

        [JsonPropertyName("samples")]
        public List<List<double>> Samples { get; set; } = new();

        [JsonPropertyName("expected_labels")]
        public List<int> ExpectedLabels { get; set; } = new();

        [JsonPropertyName("expected_probabilities")]
        public List<List<double>> ExpectedProbabilities { get; set; } = new();
    }
}
