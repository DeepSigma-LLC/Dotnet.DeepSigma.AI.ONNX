namespace DeepSigma.AI.ONNX.Demo;

/// <summary>
/// Loads the open-source MNIST CNN from the ONNX Model Zoo (mnist-12.onnx,
/// originally trained with CNTK) and classifies hand-built 28x28 digits.
/// Demonstrates loading a real third-party model and feeding it an NCHW
/// image tensor — no image-decoding library required.
/// </summary>
internal static class MnistDemo
{
    public static void Run()
    {
        string modelPath = Path.Combine(AppContext.BaseDirectory, "interop", "mnist-12.onnx");
        if (!File.Exists(modelPath))
        {
            Console.WriteLine("mnist-12.onnx not found.");
            return;
        }

        Console.WriteLine("Loading mnist-12.onnx (CNN trained on MNIST, exported from CNTK)...");
        using OnnxModel model = OnnxModel.Load(modelPath);
        model.Describe();

        foreach ((string label, string[] artwork) in HandDrawnDigits)
        {
            Console.WriteLine();
            Console.WriteLine($"Drawn as a '{label}':");
            PrintAscii(artwork);

            Tensor<float> input = AsciiToTensor(artwork);
            using InferenceResult result = model.Run(new InferenceInput().Add(model.Inputs[0].Name, input));
            Tensor<float> logits = result.Get<float>(model.Outputs[0].Name);

            float[] probs = Softmax(logits.Data);
            int predicted = ArgMax(probs);
            Console.WriteLine($"  predicted: {predicted}  (confidence {probs[predicted]:P1})");

            int[] top3 = TopK(probs, 3);
            Console.Write("  top-3:    ");
            foreach (int idx in top3) Console.Write($" {idx}={probs[idx]:P1}");
            Console.WriteLine();
        }
    }

    // Each digit is 28 rows of 28 chars. Intensity:  ' '=0, '.'=0.25, ':'=0.5, '+'=0.75, '#'=1.0
    private static readonly (string Label, string[] Pixels)[] HandDrawnDigits =
    {
        ("1", new[]
        {
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "             .+#+           ",
            "            .+###           ",
            "           .####+           ",
            "          .#####+           ",
            "         .##:###+           ",
            "         :+. ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "             ###+           ",
            "          ::.###+::         ",
            "         +##########+       ",
            "         +##########+       ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
        }),
        ("0", new[]
        {
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "          .+####+.          ",
            "         +########+         ",
            "        +###+..+###+        ",
            "        ###      ###        ",
            "       +##.       ##+       ",
            "       ###        ###       ",
            "       ###        ###       ",
            "       ###        ###       ",
            "       ###        ###       ",
            "       ###        ###       ",
            "       ###        ###       ",
            "       ###        ###       ",
            "       +##.       ##+       ",
            "        ###      ###        ",
            "        +###+..+###+        ",
            "         +########+         ",
            "          .+####+.          ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
        }),
        ("7", new[]
        {
            "                            ",
            "                            ",
            "                            ",
            "                            ",
            "      ##################+   ",
            "      ##################+   ",
            "      ##################+   ",
            "      .............+####.   ",
            "                   ####.    ",
            "                  +####     ",
            "                 .####.     ",
            "                 ####+      ",
            "                +####       ",
            "                ####.       ",
            "               +####        ",
            "               ####+        ",
            "              .####         ",
            "              ####.         ",
            "             +####          ",
            "            .####.          ",
            "            ####+           ",
            "           +####            ",
            "           ####.            ",
            "           ###+             ",
            "                            ",
            "                            ",
            "                            ",
            "                            ",
        }),
    };

    private static Tensor<float> AsciiToTensor(string[] rows)
    {
        if (rows.Length != 28) throw new ArgumentException("Expected 28 rows.", nameof(rows));
        var pixels = new float[1 * 1 * 28 * 28];
        for (int y = 0; y < 28; y++)
        {
            string row = rows[y].PadRight(28).Substring(0, 28);
            for (int x = 0; x < 28; x++)
            {
                pixels[y * 28 + x] = row[x] switch
                {
                    ' ' => 0f,
                    '.' => 0.25f,
                    ':' => 0.5f,
                    '+' => 0.75f,
                    '#' => 1.0f,
                    _ => 0f,
                };
            }
        }
        return new Tensor<float>(pixels, new[] { 1, 1, 28, 28 });
    }

    private static void PrintAscii(string[] rows)
    {
        foreach (string row in rows) Console.WriteLine("    " + row);
    }

    private static float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        var exp = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < logits.Length; i++) exp[i] /= sum;
        return exp;
    }

    private static int ArgMax(float[] values)
    {
        int idx = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[idx]) idx = i;
        return idx;
    }

    private static int[] TopK(float[] values, int k)
    {
        return Enumerable.Range(0, values.Length)
            .OrderByDescending(i => values[i])
            .Take(k)
            .ToArray();
    }
}
